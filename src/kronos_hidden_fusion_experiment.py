"""
Frozen Kronos hidden-state + SW-VMD gated fusion 验证实验。

路线 B：
1. 原始 OHLCV/amount 仍使用官方 Kronos tokenizer，不修改 tokenization。
2. 冻结 Kronos tokenizer 和 Kronos 主干，通过 decode_s1 提取最后时刻 hidden state。
3. SW-VMD low/mid/high 作为旁路序列，经 VMD encoder 得到 vmd embedding。
4. 使用第一种残差门控机制注入：
       h_fused = h + beta * gate([h, e]) * Wv(e)
5. 训练预测头输出未来 pred_len 天 close 收益率序列。

验证目标：
比较 Frozen-Kronos-Head 与 Frozen-Kronos-SWVMD-Gated-Fusion，判断 SW-VMD 作为
潜空间额外信息是否提升预测和横截面选股指标。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.kronos_stock_forecast import PREDICT_COLUMNS, list_stock_files, load_kronos_predictor, load_stock_csv
from src.rolling_window_comparison import (
    dataframe_to_markdown,
    portfolio_stats_from_cross_sections,
    summarize_all_signal_metrics,
    write_risk_direction_report,
)
from src.swvmd_kronos_experiment import vmd


@dataclass
class FusionSample:
    """一个 Kronos hidden fusion 监督样本。"""

    stock: str
    window_id: int
    as_of_date: pd.Timestamp
    dates: pd.Series
    current_close: float
    actual: pd.DataFrame
    raw_x: np.ndarray
    stamp_x: np.ndarray
    vmd_x: np.ndarray
    y: np.ndarray


class FusionDataset(Dataset):
    def __init__(self, samples: list[FusionSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.tensor(s.raw_x, dtype=torch.float32),
            torch.tensor(s.stamp_x, dtype=torch.float32),
            torch.tensor(s.vmd_x, dtype=torch.float32),
            torch.tensor(s.y, dtype=torch.float32),
            idx,
        )


class HiddenFusionHead(nn.Module):
    """Kronos hidden state 与 VMD embedding 的残差门控融合预测头。"""

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        use_vmd: bool,
        vmd_hidden: int = 64,
        dropout: float = 0.1,
        beta_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_vmd = use_vmd
        self.vmd_encoder = nn.LSTM(input_size=3, hidden_size=vmd_hidden, num_layers=1, batch_first=True)
        self.vmd_proj = nn.Linear(vmd_hidden, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len),
        )

    def forward(self, kronos_hidden: torch.Tensor, vmd_x: torch.Tensor) -> torch.Tensor:
        h = kronos_hidden
        if self.use_vmd:
            _, (vmd_last, _) = self.vmd_encoder(vmd_x)
            e = self.vmd_proj(vmd_last[-1])
            g = self.gate(torch.cat([h, e], dim=-1))
            h = h + self.beta * g * e
        return self.head(h)


def build_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate SW-VMD hidden-state fusion on frozen Kronos.")
    parser.add_argument("--stocks-dir", default=str(base_dir / "data" / "csi300_daily"))
    parser.add_argument("--kronos-repo", default=str(base_dir / "third_party" / "Kronos_official_src" / "Kronos-master"))
    parser.add_argument("--output-dir", default=str(base_dir / "train3" / "kronos_hidden_fusion_swvmd_recent10"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="NeoQuasar/Kronos-small")
    parser.add_argument("--tokenizer-name", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--lookback", type=int, default=400)
    parser.add_argument("--pred-len", type=int, default=20)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--max-windows-per-stock", type=int, default=10)
    parser.add_argument("--max-train-windows-per-stock", type=int, default=40)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vmd-hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--min-cross-section-stocks", type=int, default=30)
    parser.add_argument("--vmd-k", type=int, default=3)
    parser.add_argument("--vmd-alpha", type=float, default=2000.0)
    parser.add_argument("--vmd-tau", type=float, default=0.0)
    parser.add_argument("--vmd-tol", type=float, default=1e-6)
    parser.add_argument("--vmd-max-iter", type=int, default=300)
    parser.add_argument("--vmd-feature-clip", type=float, default=8.0)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_time_stamps(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "minute": d.dt.minute,
            "hour": d.dt.hour,
            "weekday": d.dt.weekday,
            "day": d.dt.day,
            "month": d.dt.month,
        }
    ).to_numpy(dtype=np.float32)


def normalize_kronos_x(history: pd.DataFrame) -> np.ndarray:
    x = history[PREDICT_COLUMNS].to_numpy(dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return np.clip((x - mean) / (std + 1e-5), -5.0, 5.0).astype(np.float32)


def vmd_features(close: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    mean = float(close.mean())
    std = float(close.std())
    if std <= 1e-8:
        return np.zeros((len(close), 3), dtype=np.float32)
    z_close = (close - mean) / std
    modes, _ = vmd(z_close, alpha=args.vmd_alpha, tau=args.vmd_tau, k=args.vmd_k, tol=args.vmd_tol, max_iter=args.vmd_max_iter)
    if modes.shape[0] < 3:
        modes = np.pad(modes, ((0, 3 - modes.shape[0]), (0, 0)))
    modes = np.nan_to_num(modes, nan=0.0, posinf=0.0, neginf=0.0)
    modes = np.clip(modes, -args.vmd_feature_clip, args.vmd_feature_clip)
    feats = modes[:3].T.astype(np.float32)
    feat_mean = feats.mean(axis=0, keepdims=True)
    feat_std = feats.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std <= 1e-6, 1.0, feat_std)
    feats = (feats - feat_mean) / feat_std
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(feats, -5.0, 5.0).astype(np.float32)


def build_samples(args: argparse.Namespace) -> tuple[list[FusionSample], list[FusionSample], list[dict[str, str]]]:
    train_samples: list[FusionSample] = []
    test_samples: list[FusionSample] = []
    skipped: list[dict[str, str]] = []

    for path in list_stock_files(Path(args.stocks_dir), args.max_stocks):
        stock = path.stem
        try:
            df = load_stock_csv(path)
            min_len = args.lookback + args.pred_len
            if len(df) < min_len:
                skipped.append({"stock": stock, "reason": f"only {len(df)} rows, need at least {min_len}"})
                continue
            end_positions = list(range(args.lookback, len(df) - args.pred_len + 1, args.step))
            test_count = len(end_positions) if args.max_windows_per_stock == 0 else min(args.max_windows_per_stock, len(end_positions))
            split = len(end_positions) - test_count
            train_position_ids = list(range(split))
            if args.max_train_windows_per_stock > 0:
                train_position_ids = train_position_ids[-args.max_train_windows_per_stock :]
            selected_position_ids = train_position_ids + list(range(split, len(end_positions)))

            for sample_id, end in enumerate(end_positions):
                if sample_id not in selected_position_ids:
                    continue
                history = df.iloc[end - args.lookback : end].reset_index(drop=True)
                actual = df.iloc[end : end + args.pred_len].reset_index(drop=True)
                current_close = float(history["close"].iloc[-1])
                y = (actual["close"].to_numpy(dtype=np.float32) / current_close - 1.0).astype(np.float32)
                sample = FusionSample(
                    stock=stock,
                    window_id=sample_id - split + 1 if sample_id >= split else 0,
                    as_of_date=history["date"].iloc[-1],
                    dates=actual["date"],
                    current_close=current_close,
                    actual=actual,
                    raw_x=normalize_kronos_x(history),
                    stamp_x=calc_time_stamps(history["date"]),
                    vmd_x=vmd_features(history["close"].to_numpy(dtype=float), args),
                    y=y,
                )
                if sample_id < split:
                    train_samples.append(sample)
                else:
                    test_samples.append(sample)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"stock": stock, "reason": repr(exc)})
            print(f"[SKIP] {stock}: {exc}")

    return train_samples, test_samples, skipped


def extract_kronos_hidden(predictor, raw_x: torch.Tensor, stamp_x: torch.Tensor) -> torch.Tensor:
    """冻结 Kronos：连续 OHLCV -> tokenizer token -> decode_s1 hidden last。"""
    tokenizer = predictor.tokenizer
    model = predictor.model
    tokens = tokenizer.encode(raw_x, half=True)
    _, context = model.decode_s1(tokens[0], tokens[1], stamp_x)
    return context[:, -1, :]


def train_one_model(
    predictor,
    train_samples: list[FusionSample],
    args: argparse.Namespace,
    use_vmd: bool,
    model_name: str,
) -> tuple[HiddenFusionHead, list[dict[str, object]]]:
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    d_model = predictor.model.d_model
    head = HiddenFusionHead(d_model, args.pred_len, use_vmd=use_vmd, vmd_hidden=args.vmd_hidden, dropout=args.dropout).to(device)
    loader = DataLoader(FusionDataset(train_samples), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    history: list[dict[str, object]] = []

    predictor.tokenizer.eval()
    predictor.model.eval()
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses: list[float] = []
        for raw_x, stamp_x, vmd_x, y, _ in loader:
            raw_x = raw_x.to(device)
            stamp_x = stamp_x.to(device)
            vmd_x = vmd_x.to(device)
            y = y.to(device)
            with torch.no_grad():
                hidden = extract_kronos_hidden(predictor, raw_x, stamp_x)
            pred = head(hidden, vmd_x)
            loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"model": model_name, "epoch": epoch, "train_loss": epoch_loss})
        print(f"[{model_name}] epoch {epoch:03d}/{args.epochs} train_loss={epoch_loss:.6f}")
    return head, history


def predict_one_model(
    predictor,
    head: HiddenFusionHead,
    test_samples: list[FusionSample],
    args: argparse.Namespace,
    model_name: str,
) -> pd.DataFrame:
    device = next(head.parameters()).device
    loader = DataLoader(FusionDataset(test_samples), batch_size=args.batch_size, shuffle=False)
    rows: list[pd.DataFrame] = []
    head.eval()
    predictor.tokenizer.eval()
    predictor.model.eval()

    with torch.no_grad():
        for raw_x, stamp_x, vmd_x, _, idx in loader:
            raw_x = raw_x.to(device)
            stamp_x = stamp_x.to(device)
            vmd_x = vmd_x.to(device)
            hidden = extract_kronos_hidden(predictor, raw_x, stamp_x)
            pred_returns = head(hidden, vmd_x).detach().cpu().numpy()
            for sample_idx, returns in zip(idx.tolist(), pred_returns):
                sample = test_samples[sample_idx]
                pred_close = sample.current_close * (1.0 + returns.astype(float))
                rows.append(
                    pd.DataFrame(
                        {
                            "model": model_name,
                            "stock": sample.stock,
                            "window_id": sample.window_id,
                            "as_of_date": sample.as_of_date,
                            "step": np.arange(1, args.pred_len + 1),
                            "date": sample.dates,
                            "current_close": sample.current_close,
                            "actual_open": sample.actual["open"],
                            "actual_high": sample.actual["high"],
                            "actual_low": sample.actual["low"],
                            "actual_close": sample.actual["close"],
                            "actual_volume": sample.actual["volume"],
                            "pred_open": pred_close,
                            "pred_high": pred_close,
                            "pred_low": pred_close,
                            "pred_close": pred_close,
                            "pred_volume": float(sample.actual["volume"].median()),
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def write_report(output_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, train_n: int, test_n: int) -> None:
    lines = [
        "# Frozen Kronos Hidden-State SW-VMD Gated Fusion 实验",
        "",
        "本实验冻结 Kronos/tokenizer，只训练潜空间预测头；对比 `Frozen-Kronos-Head` 与 `Frozen-Kronos-SWVMD-Gate`。",
        "",
        "## 1. 关键机制",
        "",
        "`h_fused = h + beta * gate([h, e]) * Wv(e)`，其中 h 为 Kronos 最后时刻 hidden state，e 为 SW-VMD encoder 输出。",
        "",
        "## 2. 实验设置",
        "",
        f"- 训练样本数：`{train_n}`",
        f"- 测试窗口数：`{test_n}`",
        f"- lookback：`{args.lookback}`",
        f"- pred_len：`{args.pred_len}`",
        f"- max_windows_per_stock：`{args.max_windows_per_stock}`",
        f"- epochs：`{args.epochs}`",
        "",
        "## 3. 汇总指标",
        "",
        dataframe_to_markdown(summary, floatfmt=".6f"),
    ]
    (output_dir / "hidden_fusion_swvmd_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples, test_samples, skipped = build_samples(args)
    if not train_samples or not test_samples:
        raise RuntimeError("No train/test samples built.")
    print(f"Built {len(train_samples)} train samples and {len(test_samples)} test samples.")

    predictor = load_kronos_predictor(args)
    for p in predictor.tokenizer.parameters():
        p.requires_grad_(False)
    for p in predictor.model.parameters():
        p.requires_grad_(False)

    base_head, hist_base = train_one_model(predictor, train_samples, args, use_vmd=False, model_name="Frozen-Kronos-Head")
    vmd_head, hist_vmd = train_one_model(predictor, train_samples, args, use_vmd=True, model_name="Frozen-Kronos-SWVMD-Gate")

    pred_base = predict_one_model(predictor, base_head, test_samples, args, "Frozen-Kronos-Head")
    pred_vmd = predict_one_model(predictor, vmd_head, test_samples, args, "Frozen-Kronos-SWVMD-Gate")
    predictions = pd.concat([pred_base, pred_vmd], ignore_index=True)
    summary, cross_sections = summarize_all_signal_metrics(predictions, min_cross_section_stocks=args.min_cross_section_stocks)
    portfolio = portfolio_stats_from_cross_sections(cross_sections)

    predictions.to_csv(output_dir / "rolling_predictions_all.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "rolling_metrics_summary_by_asof.csv", index=False, encoding="utf-8-sig")
    cross_sections.to_csv(output_dir / "rolling_cross_section_metrics_by_asof.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(output_dir / "portfolio_metrics_by_asof.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hist_base + hist_vmd).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    (output_dir / "rolling_metrics_summary_by_asof.md").write_text(dataframe_to_markdown(summary), encoding="utf-8")
    (output_dir / "skipped_stocks.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_risk_direction_report(output_dir, summary, portfolio)
    write_report(output_dir, args, summary, len(train_samples), len(test_samples))
    print(summary.to_string(index=False))
    print(f"Saved hidden fusion experiment to: {output_dir}")


if __name__ == "__main__":
    main()

