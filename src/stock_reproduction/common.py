from __future__ import annotations
import hashlib, json, os, random, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "train3/csi300_csi800_aligned_seed100"
PRICE_ROOT = ROOT / "data/csi300_daily/csi300_daily"
EXPECTED_TEST_ROWS, EXPECTED_TEST_DATES = 66103, 226

def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)

def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def set_determinism(seed: int, threads: int) -> None:
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)

def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(REFERENCE / "eligible_stock_date_pairs.csv", dtype=str)
    required = {"split", "stock", "as_of_date", "input_dates", "label_dates", "label_end_date"}
    if required - set(pairs): raise ValueError(f"Reference pair file is missing columns: {sorted(required-set(pairs))}")
    test = pairs[pairs.split.eq("test")]
    if len(test) != EXPECTED_TEST_ROWS or test.as_of_date.nunique() != EXPECTED_TEST_DATES: raise ValueError("Locked test set is not 66,103 rows / 226 dates")
    if pairs.duplicated(["stock", "as_of_date"]).any(): raise ValueError("Duplicate stock/date keys")
    return pairs

def load_stock(stock: str) -> pd.DataFrame:
    path = PRICE_ROOT / f"{stock.replace('.', '_')}.csv"
    frame = pd.read_csv(path, dtype={"date": str, "code": str}).set_index("date").sort_index()
    if frame.index.duplicated().any() or set(frame.code) != {stock}: raise ValueError(f"Invalid source file: {path}")
    return frame

def target_for_row(frame: pd.DataFrame, row: pd.Series) -> float:
    signal_close = float(frame.loc[row.as_of_date, "close"])
    future = frame.loc[row.label_dates.split("|"), "close"].astype(float).to_numpy()
    return float(np.mean(future / signal_close - 1.0))

def strict_window(frame: pd.DataFrame, row: pd.Series, columns: list[str]) -> np.ndarray:
    dates = row.input_dates.split("|")
    if len(dates) != 90 or dates[-1] != row.as_of_date: raise ValueError("Input is not the locked 90-day window")
    values = frame.loc[dates, columns].astype(float).to_numpy()
    if values.shape != (90, len(columns)) or not np.isfinite(values).all(): raise ValueError("Invalid input window")
    return values

def ensure_layout(home: Path) -> None:
    for name in ("adapter", "cache", "weights", "logs", "results"): (home / name).mkdir(parents=True, exist_ok=True)

def provenance(home: Path, model: str) -> dict:
    upstream = home / "upstream"
    status = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    return {"model": model, "upstream_commit": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(), "upstream_tracked_files_clean": not status, "license": "MIT" if model == "master" and (upstream / "LICENSE").exists() else "not provided by upstream repository", "python": sys.version, "device": "cpu"}
