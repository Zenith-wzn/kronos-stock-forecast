"""Standalone ODEStream-style experiment on CSI300 close streams.

This script intentionally does not import the project's forecasting or
continual-learning components.  It adapts the public ODEStream architecture
to one close-only stream per stock and keeps the public method's immediate
post-prediction update rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn


DEFAULT_DATA_DIR = Path("data/csi300_daily/csi300_daily")
DEFAULT_OUTPUT_DIR = Path(
    "train3/csi300_csi800_aligned_seed100/experiments/odestream_direct_24x1"
)
TRAIN_END = pd.Timestamp("2024-12-15")
VAL_START = pd.Timestamp("2025-01-02")
VAL_END = pd.Timestamp("2025-06-14")
TEST_START = pd.Timestamp("2025-07-01")
TEST_END = pd.Timestamp("2026-06-05")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ODEF(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, latent_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, z: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        h = self.elu(self.lin1(z))
        h = self.elu(self.lin2(h))
        return self.lin3(h)


class NeuralODE(nn.Module):
    """Differentiable Euler Neural ODE used by the public ODEStream code.

    The public ODEStream solver uses an Euler step of 0.05.  This CSI300
    adapter uses the standard normalized observation time [0, 1], so one
    complete sequence receives the same 20 solver substeps.  Normalizing the
    time axis is important here because the public ETT runs are short single
    streams, while this run shares one model across hundreds of stock streams.
    """

    def __init__(self, func: ODEF, step_size: float = 0.05) -> None:
        super().__init__()
        self.func = func
        self.step_size = step_size

    def forward(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = z0
        for t0, t1 in zip(t[:-1], t[1:]):
            delta = t1 - t0
            n_steps = max(1, int(math.ceil(float(delta.detach().abs().max()) / self.step_size)))
            h = delta / n_steps
            for step in range(n_steps):
                z = z + h * self.func(z, t0 + h * step)
        return z


class RNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim + 1, hidden_dim)
        self.hid2lat = nn.Linear(hidden_dim, 2 * latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Match the public implementation: reverse the sequence and provide
        # elapsed-time information as an additional channel.
        elapsed = t.clone()
        elapsed[1:] = t[:-1] - t[1:]
        elapsed[0] = 0.0
        xt = torch.cat((x, elapsed), dim=-1)
        _, h0 = self.rnn(xt.flip((0,)))
        z0 = self.hid2lat(h0[0])
        return z0[:, : self.latent_dim], z0[:, self.latent_dim :]


class NeuralODEDecoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.ode = NeuralODE(ODEF(latent_dim, hidden_dim))
        self.l2h = nn.Linear(latent_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = self.ode(z0, t)
        return self.h2o(self.l2h(z))


class ODEVAE(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, latent_dim: int, input_dim: int) -> None:
        super().__init__()
        self.encoder = RNNEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = NeuralODEDecoder(output_dim, hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var = self.encoder(x, t)
        z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var) if sample else z_mean
        return self.decoder(z, t), z, z_mean, z_log_var


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[-1])


class CombinedModel(nn.Module):
    def __init__(self, vae: ODEVAE, lstm: LSTMModel, output_size: int) -> None:
        super().__init__()
        self.vae = vae
        self.lstm = lstm
        self.final_layer = nn.Linear(output_size * 2, output_size)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        output1, _, _, _ = self.vae(x, t, sample=sample)
        output2 = self.lstm(x)
        return self.final_layer(torch.cat((output1, output2), dim=1))


def _stock_id(path: Path) -> str:
    return path.stem.replace("_", ".")


def load_stock_streams(data_dir: Path, seed: int, stock_limit: int | None) -> list[dict[str, object]]:
    files = sorted(data_dir.glob("*.csv"))
    if stock_limit is not None:
        files = files[:stock_limit]
    if not files:
        raise FileNotFoundError(f"No stock CSV files found in {data_dir}")

    streams: list[dict[str, object]] = []
    for path in files:
        frame = pd.read_csv(path, usecols=["date", "close"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna().sort_values("date").drop_duplicates("date")
        frame = frame[frame["close"] > 0].reset_index(drop=True)
        train_values = frame.loc[frame["date"] <= TRAIN_END, "close"].to_numpy(np.float64)
        if len(train_values) < 25:
            continue
        mean = float(train_values.mean())
        std = float(train_values.std())
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        streams.append(
            {
                "stock": _stock_id(path),
                "dates": frame["date"].to_numpy(),
                "close": frame["close"].to_numpy(np.float64),
                "mean": mean,
                "std": std,
            }
        )
    if not streams:
        raise ValueError("No usable stock streams after cleaning")
    return streams


def build_split(streams: list[dict[str, object]], split: str, lag: int) -> dict[str, object]:
    if split == "train":
        lo, hi = None, TRAIN_END
    elif split == "val":
        lo, hi = VAL_START, VAL_END
    elif split == "test":
        lo, hi = TEST_START, TEST_END
    else:
        raise ValueError(split)

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    current_rows: list[float] = []
    true_rows: list[float] = []
    dates: list[str] = []
    stocks: list[str] = []
    for stream in streams:
        stock = str(stream["stock"])
        dates_arr = np.asarray(stream["dates"])
        close = np.asarray(stream["close"], dtype=np.float64)
        mean, std = float(stream["mean"]), float(stream["std"])
        for idx in range(lag, len(close)):
            target_date = pd.Timestamp(dates_arr[idx])
            if hi is not None and target_date > hi:
                break
            if lo is not None and target_date < lo:
                continue
            window = ((close[idx - lag : idx] - mean) / std).astype(np.float32)
            target = float((close[idx] - mean) / std)
            if not np.isfinite(window).all() or not np.isfinite(target):
                continue
            x_rows.append(window[:, None])
            y_rows.append(target)
            current_rows.append(float(close[idx - 1]))
            true_rows.append(float(close[idx]))
            dates.append(target_date.date().isoformat())
            stocks.append(stock)
    if not x_rows:
        raise ValueError(f"No samples generated for {split}")
    order = np.lexsort((np.asarray(stocks), np.asarray(dates)))
    return {
        "x": np.stack(x_rows, axis=0)[order],
        "y": np.asarray(y_rows, dtype=np.float32)[order, None],
        "current_close": np.asarray(current_rows, dtype=np.float64)[order],
        "true_close": np.asarray(true_rows, dtype=np.float64)[order],
        "dates": np.asarray(dates, dtype=object)[order],
        "stocks": np.asarray(stocks, dtype=object)[order],
    }


def make_time(lag: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, lag, device=device, dtype=torch.float32).view(lag, 1, 1)


def batch_iter(split: dict[str, object], batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    n = len(split["x"])
    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield indices[start : start + batch_size]


def fit_warmup(
    vae: ODEVAE,
    train: dict[str, object],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_samples: int | None,
) -> list[float]:
    optimizer = torch.optim.Adam(vae.parameters(), betas=(0.9, 0.999), lr=learning_rate)
    rng = np.random.default_rng(seed)
    n = len(train["x"])
    use_n = min(n, max_samples) if max_samples is not None else n
    losses: list[float] = []
    t = make_time(int(train["x"].shape[1]), device)
    for epoch in range(epochs):
        indices = rng.permutation(n)[:use_n]
        epoch_losses: list[float] = []
        vae.train()
        for start in range(0, len(indices), batch_size):
            ix = indices[start : start + batch_size]
            x = torch.from_numpy(np.asarray(train["x"])[ix]).to(device).transpose(0, 1)
            y = torch.from_numpy(np.asarray(train["y"])[ix]).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _, _, _ = vae(x, t.expand(-1, len(ix), -1), sample=True)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        print(f"warmup epoch={epoch + 1}/{epochs} samples={use_n} mse={mean_loss:.8f}", flush=True)
    return losses


@torch.no_grad()
def predict_one(model: CombinedModel, x: np.ndarray, t: torch.Tensor, device: torch.device) -> float:
    model.eval()
    xt = torch.from_numpy(x[None]).to(device).transpose(0, 1)
    pred = model(xt, t, sample=True)
    return float(pred[0, 0].detach().cpu())


def update_one(
    model: CombinedModel,
    x: np.ndarray,
    y: float,
    t: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    xt = torch.from_numpy(x[None]).to(device).transpose(0, 1)
    yt = torch.tensor([[y]], dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    pred = model(xt, t, sample=True)
    loss = torch.mean((pred - yt) ** 2)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def evaluate_vae(
    vae: ODEVAE,
    split: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    vae.eval()
    t = make_time(int(split["x"].shape[1]), device)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(split["x"]), batch_size):
            x = torch.from_numpy(np.asarray(split["x"])[start : start + batch_size]).to(device).transpose(0, 1)
            pred, _, _, _ = vae(x, t.expand(-1, x.shape[1], -1), sample=False)
            preds.append(pred[:, 0].detach().cpu().numpy())
    pred_norm = np.concatenate(preds)
    true_norm = np.asarray(split["y"], dtype=np.float64)[:, 0]
    current = np.asarray(split["current_close"], dtype=np.float64)
    true = np.asarray(split["true_close"], dtype=np.float64)
    # Aggregate normalized errors because stocks have different price scales.
    err = pred_norm - true_norm
    pred_direction = np.sign(pred_norm - np.asarray(split["x"])[:, -1, 0])
    true_direction = np.sign(true - current)
    return {
        "rows": int(len(err)),
        "mse_normalized": float(np.mean(err**2)),
        "mae_normalized": float(np.mean(np.abs(err))),
        "direction_accuracy": float(np.mean(pred_direction == true_direction)),
    }


def write_daily_metrics(prediction_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(prediction_path)
    grouped = []
    for date, group in frame.groupby("date", sort=True):
        err = group["pred_close"] - group["true_close"]
        grouped.append(
            {
                "date": date,
                "rows": int(len(group)),
                "mse_close": float(np.mean(err**2)),
                "mae_close": float(np.mean(np.abs(err))),
                "direction_accuracy": float((group["pred_direction"] == group["true_direction"]).mean()),
                "mean_pred_close": float(group["pred_close"].mean()),
                "mean_true_close": float(group["true_close"].mean()),
            }
        )
    pd.DataFrame(grouped).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--stock-limit", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--warmup-batch-size", type=int, default=64)
    parser.add_argument("--warmup-lr", type=float, default=1e-3)
    parser.add_argument("--online-lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lag", type=int, default=24)
    parser.add_argument("--online-batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.online_batch_size != 1:
        raise ValueError("ODEStream direct mode requires online-batch-size=1")
    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.threads))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    config = {
        "status": "running",
        "method": "ODEStream_direct",
        "input": "close_only_per_stock_stream",
        "lag": args.lag,
        "horizon": 1,
        "train_end": TRAIN_END.date().isoformat(),
        "val_start": VAL_START.date().isoformat(),
        "val_end": VAL_END.date().isoformat(),
        "test_start": TEST_START.date().isoformat(),
        "test_end": TEST_END.date().isoformat(),
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "num_layers": args.num_layers,
        "warmup_epochs": args.warmup_epochs,
        "warmup_batch_size": args.warmup_batch_size,
        "warmup_lr": args.warmup_lr,
        "online_lr": args.online_lr,
        "online_batch_size": args.online_batch_size,
        "seed": args.seed,
        "device": str(device),
        "stock_limit": args.stock_limit,
        "max_train_samples": args.max_train_samples,
        "max_val_rows": args.max_val_rows,
        "max_test_rows": args.max_test_rows,
        "forbidden_components": ["Kronos", "hidden_cache", "Adapter", "replay", "validation_gate", "ranking_head"],
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"loading raw close streams from {args.data_dir}", flush=True)
    streams = load_stock_streams(args.data_dir, args.seed, args.stock_limit)
    print(f"loaded stocks={len(streams)} device={device}", flush=True)
    stream_by_stock = {str(stream["stock"]): stream for stream in streams}
    train = build_split(streams, "train", args.lag)
    val = build_split(streams, "val", args.lag)
    test = build_split(streams, "test", args.lag)
    if args.max_val_rows is not None:
        for key in ("x", "y", "current_close", "true_close", "dates", "stocks"):
            val[key] = val[key][: args.max_val_rows]
    if args.max_test_rows is not None:
        for key in ("x", "y", "current_close", "true_close", "dates", "stocks"):
            test[key] = test[key][: args.max_test_rows]
    config.update({"stocks": len(streams), "train_rows": len(train["x"]), "val_rows": len(val["x"]), "test_rows": len(test["x"])})
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"samples train={len(train['x'])} val={len(val['x'])} test={len(test['x'])}", flush=True)

    vae = ODEVAE(1, args.hidden_dim, args.latent_dim, 1).to(device)
    warmup_losses = fit_warmup(
        vae, train, device, args.warmup_epochs, args.warmup_batch_size, args.warmup_lr,
        args.seed, args.max_train_samples,
    )
    torch.save(vae.state_dict(), args.output_dir / "odestream_warmup.pt")
    val_metrics = evaluate_vae(vae, val, device, args.warmup_batch_size)
    print(f"validation={json.dumps(val_metrics)}", flush=True)

    model = CombinedModel(vae, LSTMModel(1, args.hidden_dim, args.num_layers, 1), 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.online_lr)
    t = make_time(args.lag, device)
    prediction_path = args.output_dir / "predictions.csv"
    loss_path = args.output_dir / "online_loss.csv"
    checkpoint_path = args.output_dir / "online_checkpoint.pt"
    start_index = 0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume without {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_index = int(state["next_index"])
        print(f"resuming online stream at row={start_index}/{len(test['x'])}", flush=True)
    file_mode = "a" if args.resume else "w"
    with prediction_path.open(file_mode, newline="", encoding="utf-8") as pred_file, loss_path.open(file_mode, newline="", encoding="utf-8") as loss_file:
        pred_writer = csv.DictWriter(
            pred_file,
            fieldnames=["date", "stock", "pred_norm", "true_norm", "pred_close", "true_close", "current_close", "pred_direction", "true_direction", "update_order"],
        )
        loss_writer = csv.DictWriter(loss_file, fieldnames=["update_order", "date", "stock", "online_mse_normalized"])
        if not args.resume:
            pred_writer.writeheader()
            loss_writer.writeheader()
        model_update_order = start_index
        for i in range(start_index, len(test["x"])):
            x = np.asarray(test["x"])[i]
            y_norm = float(np.asarray(test["y"])[i, 0])
            current_close = float(np.asarray(test["current_close"])[i])
            true_close = float(np.asarray(test["true_close"])[i])
            stock = str(np.asarray(test["stocks"])[i])
            date = str(np.asarray(test["dates"])[i])
            stream = stream_by_stock[stock]
            mean, std = float(stream["mean"]), float(stream["std"])
            pred_norm = predict_one(model, x, t, device)
            pred_close = pred_norm * std + mean
            pred_direction = 1 if pred_close > current_close else (-1 if pred_close < current_close else 0)
            true_direction = 1 if true_close > current_close else (-1 if true_close < current_close else 0)
            pred_writer.writerow({
                "date": date, "stock": stock, "pred_norm": pred_norm, "true_norm": y_norm,
                "pred_close": pred_close, "true_close": true_close, "current_close": current_close,
                "pred_direction": pred_direction, "true_direction": true_direction,
                "update_order": model_update_order,
            })
            online_loss = update_one(model, x, y_norm, t, device, optimizer)
            loss_writer.writerow({"update_order": model_update_order, "date": date, "stock": stock, "online_mse_normalized": online_loss})
            model_update_order += 1
            if model_update_order % args.checkpoint_every == 0 or model_update_order == len(test["x"]):
                torch.save(
                    {"next_index": model_update_order, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    checkpoint_path,
                )
                pred_file.flush()
                loss_file.flush()
                print(f"online rows={model_update_order}/{len(test['x'])} loss={online_loss:.8f}", flush=True)

    write_daily_metrics(prediction_path, args.output_dir / "daily_metrics.csv")
    pred_frame = pd.read_csv(prediction_path)
    norm_error = pred_frame["pred_norm"] - pred_frame["true_norm"]
    summary = {
        "status": "complete",
        "method": "ODEStream_direct",
        "rows": int(pred_frame["update_order"].nunique()),
        "stocks": int(pred_frame["stock"].nunique()),
        "dates": int(pred_frame["date"].nunique()),
        "warmup_mse_normalized": warmup_losses,
        "validation": val_metrics,
        "test_mse_normalized": float(np.mean(norm_error**2)),
        "test_mae_normalized": float(np.mean(np.abs(norm_error))),
        "test_direction_accuracy": float((pred_frame["pred_direction"] == pred_frame["true_direction"]).mean()),
        "elapsed_seconds": time.time() - started,
        "output_dir": str(args.output_dir.resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    config.update({"status": "complete", "elapsed_seconds": summary["elapsed_seconds"]})
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
