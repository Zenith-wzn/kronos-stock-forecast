"""
在本地股票 CSV 数据上运行 Kronos 预测，并导出评测结果。

脚本支持两种预测引擎：
1. kronos：调用官方 Kronos 模型做预测。
2. naive：朴素基线，把未来价格预测为历史窗口最后一天的价格。

两种引擎共用同一套数据切分、指标计算和画图逻辑，因此结果可以直接比较。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 输入 CSV 至少需要这些字段。
REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# 传给 Kronos 的特征列。amount 不存在时会用价格均值 * volume 近似生成。
PREDICT_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


@dataclass
class RunConfig:
    """保存本次运行的关键参数，最后会写入 run_config.json。"""

    stocks_dir: str
    kronos_repo: str
    output_dir: str
    engine: str
    model_name: str
    tokenizer_name: str
    device: str
    lookback: int
    pred_len: int
    max_stocks: int | None
    sample_count: int
    temperature: float
    top_p: float
    batch_size: int


def parse_args() -> argparse.Namespace:
    """保留的简易参数入口，目前主流程使用 build_parser()。"""
    base_dir = Path(__file__).resolve().parents[1]
    return argparse.ArgumentParser(
        description="Forecast local OHLCV stock CSV files with Kronos and evaluate the holdout window."
    ).parse_args()


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Forecast local OHLCV stock CSV files with Kronos and evaluate the holdout window."
    )
    parser.add_argument(
        "--stocks-dir",
        default=str(base_dir / "data" / "csi300_daily"),
        help="Folder containing stock CSV files.",
    )
    parser.add_argument(
        "--kronos-repo",
        default=str(base_dir / "third_party" / "Kronos_official_src" / "Kronos-master"),
        help="Path to the official Kronos source folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(base_dir / "kronos_outputs"),
        help="Folder for predictions, metrics, plots, and run metadata.",
    )
    parser.add_argument(
        "--engine",
        choices=["kronos", "naive"],
        default="kronos",
        help="Use 'kronos' for the official model; 'naive' only validates the local pipeline.",
    )
    parser.add_argument("--model-name", default="NeoQuasar/Kronos-small")
    parser.add_argument("--tokenizer-name", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, or another torch device.")
    parser.add_argument("--lookback", type=int, default=400)
    parser.add_argument("--pred-len", type=int, default=20)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for Kronos prediction.")
    return parser


def load_stock_csv(path: Path) -> pd.DataFrame:
    """读取单只股票 CSV，并做字段规范化、缺失值处理和日期排序。"""
    df = pd.read_csv(path)

    # 统一列名为小写，避免 Date/date、Close/close 之类大小写差异导致读取失败。
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 价格和成交量都转成数值；无法转换的内容会变成 NaN，后面统一删除。
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除关键字段缺失的行，按日期升序排列，并去掉重复日期。
    df = df.dropna(subset=REQUIRED_COLUMNS).sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # Kronos 需要 amount 字段。如果原数据没有，就用 OHLC 均价乘成交量近似。
    if "amount" not in df.columns:
        df["amount"] = df[["open", "high", "low", "close"]].mean(axis=1) * df["volume"]
    else:
        fallback_amount = df[["open", "high", "low", "close"]].mean(axis=1) * df["volume"]
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(fallback_amount)
    return df


def list_stock_files(stocks_dir: Path, max_stocks: int | None) -> list[Path]:
    """列出待预测的股票 CSV 文件，可用 max_stocks 限制数量便于调试。"""
    files = sorted(stocks_dir.glob("*.csv"))
    if max_stocks is not None:
        files = files[:max_stocks]
    if not files:
        raise FileNotFoundError(f"No CSV files found under {stocks_dir}")
    return files


def make_naive_prediction(history: pd.DataFrame, y_timestamp: pd.Series, pred_len: int) -> pd.DataFrame:
    """朴素基线：未来价格全部等于历史窗口最后一天的价格。"""
    last = history.iloc[-1]
    recent = history.tail(min(len(history), 20))

    # 建立和 Kronos 输出结构一致的 DataFrame，方便后续共用评测代码。
    pred = pd.DataFrame(index=pd.RangeIndex(pred_len), columns=PREDICT_COLUMNS, dtype=float)
    for col in ["open", "high", "low", "close"]:
        pred[col] = float(last[col])

    # volume 不直接固定最后一天，而用最近 20 天中位数，减少单日异常成交量影响。
    pred["volume"] = float(recent["volume"].median())
    pred["amount"] = pred["close"] * pred["volume"]
    pred.index = pd.to_datetime(y_timestamp).to_numpy()
    return pred


def load_kronos_predictor(args: argparse.Namespace):
    """加载官方 Kronos 源码、tokenizer 和预测模型。"""
    kronos_repo = Path(args.kronos_repo).resolve()
    if not kronos_repo.exists():
        raise FileNotFoundError(f"Kronos source folder not found: {kronos_repo}")

    # 把官方 Kronos 源码目录加入 Python 搜索路径，才能 from model import ...
    sys.path.insert(0, str(kronos_repo))

    try:
        from model import Kronos, KronosPredictor, KronosTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import Kronos. Check that PyTorch works and Kronos requirements are installed."
        ) from exc

    # tokenizer 负责把连续 OHLCV 数据离散化，model 负责自回归预测未来 token。
    tokenizer = load_hf_pytorch_model(KronosTokenizer, args.tokenizer_name)
    model = load_hf_pytorch_model(Kronos, args.model_name)
    return KronosPredictor(model, tokenizer, device=args.device, max_context=args.lookback)


def load_hf_pytorch_model(cls, repo_id: str):
    """从 Hugging Face 加载 Kronos 权重，并兼容不同 from_pretrained 实现。"""
    try:
        return cls.from_pretrained(repo_id)
    except TypeError:
        # 某些类的 from_pretrained 签名不完全兼容时，手动下载 config 和 safetensors 权重。
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_model

        config_path = hf_hub_download(repo_id, "config.json")
        weights_path = hf_hub_download(repo_id, "model.safetensors")
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        instance = cls(**config)
        load_model(instance, weights_path, strict=True)
        return instance


def predict_with_kronos(predictor, history: pd.DataFrame, y_timestamp: pd.Series, args: argparse.Namespace) -> pd.DataFrame:
    """对单只股票调用 Kronos 预测未来 pred_len 个时间点。"""
    x_df = history[PREDICT_COLUMNS].copy()
    x_timestamp = history["date"].copy()
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.pred_len,
        T=args.temperature,
        top_p=args.top_p,
        sample_count=args.sample_count,
        verbose=False,
    )
    pred_df.index = pd.to_datetime(y_timestamp).to_numpy()
    return pred_df


def predict_batch_with_kronos(predictor, histories: list[pd.DataFrame], y_timestamps: list[pd.Series], args: argparse.Namespace) -> list[pd.DataFrame]:
    """批量调用 Kronos，提高多只股票推理效率。"""
    pred_list = predictor.predict_batch(
        df_list=[history[PREDICT_COLUMNS].copy() for history in histories],
        x_timestamp_list=[history["date"].copy() for history in histories],
        y_timestamp_list=y_timestamps,
        pred_len=args.pred_len,
        T=args.temperature,
        top_p=args.top_p,
        sample_count=args.sample_count,
        verbose=False,
    )
    for pred_df, y_timestamp in zip(pred_list, y_timestamps):
        pred_df.index = pd.to_datetime(y_timestamp).to_numpy()
    return pred_list


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 MAPE，跳过真实值接近 0 的点，避免除零。"""
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 sMAPE，分母为 |真实值| + |预测值|。"""
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """用 numpy 计算 R2；真实序列没有波动时返回 NaN。"""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def directional_accuracy(history_close: float, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算方向准确率：判断预测涨跌方向是否和真实涨跌方向一致。"""
    # 第一个预测点需要和历史窗口最后一个 close 比较；之后和前一天比较。
    actual_prev = np.r_[history_close, y_true[:-1]]
    pred_prev = np.r_[history_close, y_pred[:-1]]
    actual_direction = np.sign(y_true - actual_prev)
    pred_direction = np.sign(y_pred - pred_prev)
    return float(np.mean(actual_direction == pred_direction) * 100)


def compute_metrics(stock: str, actual: pd.DataFrame, pred: pd.DataFrame, history_close: float) -> dict[str, float | str | int]:
    """计算单只股票在预测窗口内的评测指标。"""
    y_true = actual["close"].to_numpy(dtype=float)
    y_pred = pred["close"].to_numpy(dtype=float)
    err = y_pred - y_true
    abs_pct = np.abs(err / np.where(np.abs(y_true) < 1e-12, np.nan, y_true))
    return {
        "stock": stock,
        "n": int(len(y_true)),
        # 价格误差类指标，越小越好。
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "mape_pct": safe_mape(y_true, y_pred),
        "smape_pct": safe_smape(y_true, y_pred),
        "r2": r2_score_np(y_true, y_pred),
        # 方向和命中率类指标，通常越大越好。
        "directional_accuracy_pct": directional_accuracy(history_close, y_true, y_pred),
        "hit_rate_1pct": float(np.nanmean(abs_pct <= 0.01) * 100),
        "hit_rate_2pct": float(np.nanmean(abs_pct <= 0.02) * 100),
        "hit_rate_5pct": float(np.nanmean(abs_pct <= 0.05) * 100),
        # 波动相关统计，用来观察模型预测是否过平或过度波动。
        "actual_close_std": float(np.std(y_true, ddof=0)),
        "pred_close_std": float(np.std(y_pred, ddof=0)),
        "pred_close_range": float(np.max(y_pred) - np.min(y_pred)),
    }


def plot_stock(stock: str, history: pd.DataFrame, actual: pd.DataFrame, pred: pd.DataFrame, metrics: dict[str, float | str | int], output_path: Path) -> None:
    """给单只股票画预测结果图：历史、预测窗口和误差柱状图。"""
    context_plot = history.tail(120)
    pred_dates = actual["date"].reset_index(drop=True)
    pred_close = pred["close"].reset_index(drop=True)
    actual_close = actual["close"].reset_index(drop=True)
    err = pred_close - actual_close

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10), height_ratios=[1.25, 1.0, 0.75])

    # 第一张子图：历史 close、真实未来 close、预测 close 的整体走势。
    ax1.plot(context_plot["date"], context_plot["close"], label="history close", color="#2b6cb0", linewidth=1.5)
    ax1.plot(actual["date"], actual["close"], label="actual close", color="#111827", linewidth=1.8)
    ax1.plot(actual["date"], pred["close"], label="predicted close", color="#dc2626", linewidth=1.8)
    ax1.axvline(actual["date"].iloc[0], color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.8)
    ax1.set_title(
        f"{stock} Kronos close forecast | RMSE={metrics['rmse']:.4f}, "
        f"MAPE={metrics['mape_pct']:.2f}%, pred_std={metrics['pred_close_std']:.4f}"
    )
    ax1.set_ylabel("Close")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    # 第二张子图：只放大预测窗口，便于看未来 20 天的贴合情况。
    ax2.plot(pred_dates, actual_close, marker="o", label="actual close", color="#111827", linewidth=1.8)
    ax2.plot(pred_dates, pred_close, marker="o", label="predicted close", color="#dc2626", linewidth=1.8)
    ax2.set_title("Forecast window zoom")
    ax2.set_ylabel("Close")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    # 第三张子图：预测误差。红色表示预测高于真实值，蓝色表示预测低于真实值。
    colors = np.where(err >= 0, "#dc2626", "#2563eb")
    ax3.bar(pred_dates, err, color=colors, alpha=0.85)
    ax3.axhline(0, color="#111827", linewidth=1.0)
    ax3.set_title("Prediction error: predicted close - actual close")
    ax3.set_ylabel("Error")
    ax3.grid(True, axis="y", alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_summary(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """画 RMSE 最低的前 20 只股票，快速查看表现最好的样本。"""
    top = metrics_df.sort_values("rmse").head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(top["stock"], top["rmse"], color="#2563eb")
    ax.set_title("Lowest RMSE stocks in this run")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """汇总所有股票的指标，输出 mean/median/std/min/max。"""
    numeric_cols = [c for c in metrics_df.columns if c not in {"stock"}]
    summary = metrics_df[numeric_cols].agg(["mean", "median", "std", "min", "max"]).reset_index()
    return summary.rename(columns={"index": "stat"})


def save_stock_result(
    stock: str,
    args: argparse.Namespace,
    history: pd.DataFrame,
    actual: pd.DataFrame,
    pred: pd.DataFrame,
    plots_dir: Path,
    all_predictions: list[pd.DataFrame],
    all_metrics: list[dict[str, float | str | int]],
) -> None:
    """保存单只股票的预测结果、指标，并追加到总表缓存中。"""
    pred = pred.reset_index(drop=True)

    # 如果某个预测引擎没有输出完整字段，用 NaN 补齐，保持 CSV 列结构稳定。
    for col in PREDICT_COLUMNS:
        if col not in pred.columns:
            pred[col] = np.nan

    # 当前脚本主要评估 close 预测，其他 OHLCV 字段会一起保存方便后续扩展。
    metrics = compute_metrics(stock, actual, pred, float(history["close"].iloc[-1]))
    all_metrics.append(metrics)

    # 每个预测时间点一行，汇总后写入 predictions_all.csv。
    pred_out = pd.DataFrame(
        {
            "stock": stock,
            "engine": args.engine,
            "date": actual["date"],
            "actual_open": actual["open"],
            "actual_high": actual["high"],
            "actual_low": actual["low"],
            "actual_close": actual["close"],
            "actual_volume": actual["volume"],
            "pred_open": pred["open"],
            "pred_high": pred["high"],
            "pred_low": pred["low"],
            "pred_close": pred["close"],
            "pred_volume": pred["volume"],
        }
    )
    all_predictions.append(pred_out)
    plot_stock(stock, history, actual, pred, metrics, plots_dir / f"{stock}_forecast.png")
    print(
        f"[OK] {stock}: RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape_pct']:.2f}%, "
        f"pred_range={metrics['pred_close_range']:.4f}"
    )


def run(args: argparse.Namespace) -> None:
    """主流程：准备数据、运行预测、计算指标、保存所有输出。"""
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    predictor = None
    if args.engine == "kronos":
        # Kronos 模型只加载一次，后面对多只股票复用。
        predictor = load_kronos_predictor(args)

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, float | str | int]] = []
    skipped: list[dict[str, str]] = []

    prepared: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.Series]] = []
    for path in list_stock_files(Path(args.stocks_dir), args.max_stocks):
        try:
            stock = path.stem
            df = load_stock_csv(path)
            min_len = args.lookback + args.pred_len
            if len(df) < min_len:
                skipped.append({"stock": stock, "reason": f"only {len(df)} rows, need at least {min_len}"})
                continue

            # 切分方式：
            # - history：倒数第 pred_len 天之前的 lookback 条数据，作为模型输入。
            # - actual：最后 pred_len 条数据，作为真实测试窗口。
            history = df.iloc[-(args.lookback + args.pred_len) : -args.pred_len].reset_index(drop=True)
            actual = df.iloc[-args.pred_len :].reset_index(drop=True)
            y_timestamp = actual["date"]
            prepared.append((stock, history, actual, y_timestamp))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"stock": path.stem, "reason": repr(exc)})
            print(f"[SKIP] {path.stem}: {exc}")

    if args.engine == "kronos" and args.batch_size > 1:
        # Kronos 支持批量预测，多只股票一起送入模型更快。
        for start in range(0, len(prepared), args.batch_size):
            batch = prepared[start : start + args.batch_size]
            try:
                preds = predict_batch_with_kronos(
                    predictor,
                    [item[1] for item in batch],
                    [item[3] for item in batch],
                    args,
                )
                for (stock, history, actual, _), pred in zip(batch, preds):
                    save_stock_result(stock, args, history, actual, pred, plots_dir, all_predictions, all_metrics)
            except Exception as exc:  # noqa: BLE001
                # 如果批量预测失败，退回单只股票逐个预测，尽量保留可成功的样本。
                for stock, history, actual, y_timestamp in batch:
                    try:
                        pred = predict_with_kronos(predictor, history, y_timestamp, args)
                        save_stock_result(stock, args, history, actual, pred, plots_dir, all_predictions, all_metrics)
                    except Exception as single_exc:  # noqa: BLE001
                        skipped.append({"stock": stock, "reason": repr(single_exc)})
                        print(f"[SKIP] {stock}: {single_exc}")
    else:
        # naive 引擎，或者 batch_size=1 时，走单只股票循环。
        for stock, history, actual, y_timestamp in prepared:
            try:
                if args.engine == "kronos":
                    pred = predict_with_kronos(predictor, history, y_timestamp, args)
                else:
                    pred = make_naive_prediction(history, y_timestamp, args.pred_len)
                save_stock_result(stock, args, history, actual, pred, plots_dir, all_predictions, all_metrics)
            except Exception as exc:  # noqa: BLE001
                skipped.append({"stock": stock, "reason": repr(exc)})
                print(f"[SKIP] {stock}: {exc}")

    if not all_metrics:
        raise RuntimeError("No stock was successfully processed. See skipped_stocks.json for details.")

    # 拼接所有股票的预测明细、逐股票指标和整体汇总指标。
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics).sort_values("stock").reset_index(drop=True)
    summary_df = summarize_metrics(metrics_df)

    # 导出三个核心 CSV：
    # 1. predictions_all.csv：每只股票每个预测日的真实值和预测值。
    # 2. metrics_by_stock.csv：每只股票一行的指标。
    # 3. metrics_summary.csv：所有股票指标的统计汇总。
    predictions_df.to_csv(output_dir / "predictions_all.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(output_dir / "metrics_by_stock.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    plot_summary(metrics_df, output_dir / "metrics_rmse_top20.png")

    # 保存运行配置，方便之后复现实验。
    config = RunConfig(
        stocks_dir=str(Path(args.stocks_dir).resolve()),
        kronos_repo=str(Path(args.kronos_repo).resolve()),
        output_dir=str(output_dir.resolve()),
        engine=args.engine,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        lookback=args.lookback,
        pred_len=args.pred_len,
        max_stocks=args.max_stocks,
        sample_count=args.sample_count,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )
    (output_dir / "run_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存跳过的股票和原因，例如数据长度不足、字段缺失或模型推理失败。
    (output_dir / "skipped_stocks.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"  {output_dir / 'predictions_all.csv'}")
    print(f"  {output_dir / 'metrics_by_stock.csv'}")
    print(f"  {output_dir / 'metrics_summary.csv'}")
    print(f"  {plots_dir}")


def main() -> None:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

