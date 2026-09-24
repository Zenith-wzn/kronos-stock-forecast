#!/usr/bin/env python
"""Prepare the CSI800 panel for the Kronos-paper-shifted continual-learning experiments.

This creates new artifacts only; it never edits the original zero-shot or
Adapter scripts/results.  The test dates are fixed to the prior comparison
window: 2025-07-01 through 2026-06-05 with strict 90/10 windows.
"""
from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from continual_abcs.panel import build_eligible_pairs, build_master_calendar

TEST_START = "2025-07-01"
TEST_END = "2026-06-05"
LOOKBACK = 90
PRED_LEN = 10


def date_hash(values: list[pd.Timestamp]) -> str:
    text = "|".join(pd.Timestamp(v).date().isoformat() for v in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_stock(value: object) -> str:
    """Normalize legacy sh_600000 and panel sh.600000 identifiers."""
    text = str(value).strip().lower().replace("_", ".")
    if text.startswith(("sh", "sz")) and "." not in text and len(text) >= 8:
        text = text[:2] + "." + text[2:]
    return text


def _find_rolling_predictions(path: Path) -> Path:
    if path.is_file():
        return path
    direct = path / "rolling_predictions_all.csv"
    if direct.exists():
        return direct
    files = sorted(path.rglob("rolling_predictions_all.csv"))
    if not files:
        raise FileNotFoundError(f"no rolling_predictions_all.csv under reference: {path}")
    return files[0]


def _load_reference_keys(path: Path, *, model: str | None = None) -> pd.DataFrame:
    file = _find_rolling_predictions(path)
    wanted = {"model", "stock", "code", "symbol", "as_of_date"}
    raw = pd.read_csv(file, usecols=lambda c: str(c).strip() in wanted)
    raw.columns = [str(c).strip() for c in raw.columns]
    if model is not None and "model" in raw.columns:
        raw = raw[raw["model"].astype(str).str.casefold() == model.casefold()].copy()
    stock_col = next((c for c in ("stock", "code", "symbol") if c in raw.columns), None)
    if stock_col is None or "as_of_date" not in raw.columns:
        raise ValueError(f"reference predictions need stock and as_of_date columns: {file}")
    out = pd.DataFrame({
        "stock_key": raw[stock_col].map(canonical_stock),
        "as_of_date": pd.to_datetime(raw["as_of_date"], errors="raise").dt.normalize(),
    }).drop_duplicates()
    if out.empty:
        raise ValueError(f"reference predictions contain no usable keys: {file}")
    return out


def key_hash(frame: pd.DataFrame) -> str:
    ordered = frame[["as_of_date", "stock_key"]].drop_duplicates().sort_values(["as_of_date", "stock_key"])
    text = "|".join(f"{d.date().isoformat()}:{s}" for d, s in ordered.itertuples(index=False, name=None))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_frames(stocks_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(stocks_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            continue
        if "code" not in frame.columns:
            frame["code"] = path.stem
        code_values = frame["code"].dropna().astype(str)
        stock = code_values.iloc[-1] if len(code_values) else path.stem
        frame["code"] = stock
        frames[stock] = frame
    if not frames:
        raise RuntimeError(f"no usable OHLCV CSV files found in {stocks_dir}")
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stocks-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--reference-zero-shot-dir", help="optional prior zero-shot predictions/by_date directory")
    p.add_argument("--reference-adapter-dir", help="latest Adapter output used to lock the exact recent-50 as_of_date calendar")
    p.add_argument("--calendar-min-fraction", type=float, default=0.5)
    p.add_argument("--min-cross-section", type=int, default=30)
    p.add_argument("--lookback", type=int, default=LOOKBACK)
    p.add_argument("--pred-len", type=int, default=PRED_LEN)
    p.add_argument("--test-start", default=TEST_START)
    p.add_argument("--test-end", default=TEST_END)
    p.add_argument("--train-end", default="2024-12-15")
    p.add_argument("--val-start", default="2025-01-02")
    p.add_argument("--val-end", default="2025-06-14")
    a = p.parse_args()

    if (a.test_start, a.test_end) != (TEST_START, TEST_END):
        raise ValueError(f"locked test interval must remain {TEST_START}..{TEST_END}")
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    frames = load_frames(Path(a.stocks_dir))
    calendar = build_master_calendar(frames, min_cross_section=a.min_cross_section, calendar_min_fraction=a.calendar_min_fraction)
    split_dates = {
        "train": ("1900-01-01", a.train_end),
        "val": (a.val_start, a.val_end),
        "test": (a.test_start, a.test_end),
    }
    pairs = build_eligible_pairs(frames, calendar, lookback=a.lookback, pred_len=a.pred_len,
                                 split_dates=split_dates, min_cross_section=a.min_cross_section)
    if pairs.empty:
        raise RuntimeError("eligible panel is empty")
    windows = pairs.drop_duplicates(["split", "as_of_date"]).sort_values(["split", "as_of_date"]).reset_index(drop=True)
    eligible_test_dates = pd.to_datetime(windows.loc[windows["split"] == "test", "as_of_date"]).dt.normalize().sort_values().drop_duplicates()
    if eligible_test_dates.empty:
        raise RuntimeError("no eligible test dates in locked interval")

    reference_dates: list[str] | None = None
    adapter_dates: list[str] | None = None
    if a.reference_adapter_dir:
        adapter_keys_all = _load_reference_keys(Path(a.reference_adapter_dir))
        adapter_all = adapter_keys_all["as_of_date"].drop_duplicates().sort_values()
        adapter_all = adapter_all[(adapter_all >= pd.Timestamp(a.test_start)) & (adapter_all <= pd.Timestamp(a.test_end))]
        adapter_dates = [x.date().isoformat() for x in adapter_all.tail(50)]
        if len(adapter_dates) != 50:
            raise RuntimeError(f"Adapter reference does not provide exactly 50 dates in locked interval: got {len(adapter_dates)}")
        expected = set(adapter_dates)
        actual = set(eligible_test_dates.dt.date.astype(str))
        missing = sorted(expected - actual)
        if missing:
            raise RuntimeError(f"Adapter recent-50 dates are not eligible in panel: missing={missing[:5]}")
        # Lock the exact Adapter recent-50 stock/date keys, not merely the date interval.
        keep_dates = pd.to_datetime(adapter_dates)
        adapter_test_keys = adapter_keys_all[adapter_keys_all["as_of_date"].isin(keep_dates)].drop_duplicates()
        test_mask = pairs["split"] == "test"
        pair_test = pairs.loc[test_mask, ["stock", "as_of_date"]].copy()
        pair_test["stock_key"] = pair_test["stock"].map(canonical_stock)
        pair_test["as_of_date"] = pd.to_datetime(pair_test["as_of_date"]).dt.normalize()
        available_keys = set(map(tuple, pair_test[["as_of_date", "stock_key"]].itertuples(index=False, name=None)))
        expected_keys = set(map(tuple, adapter_test_keys[["as_of_date", "stock_key"]].itertuples(index=False, name=None)))
        missing_keys = sorted(expected_keys - available_keys)
        if missing_keys:
            raise RuntimeError(f"Adapter recent-50 keys are not eligible in strict 90/10 panel: missing={missing_keys[:5]}")
        keep_key = list(zip(pair_test["as_of_date"], pair_test["stock_key"]))
        pairs.loc[test_mask, "_keep_adapter_key"] = [key in expected_keys for key in keep_key]
        pairs = pairs[(pairs["split"] != "test") | pairs["_keep_adapter_key"].fillna(False)].drop(columns="_keep_adapter_key").copy()
        windows = pairs.drop_duplicates(["split", "as_of_date"]).sort_values(["split", "as_of_date"]).reset_index(drop=True)

    test_dates = pd.to_datetime(windows.loc[windows["split"] == "test", "as_of_date"]).dt.date.astype(str).tolist()
    if not test_dates:
        raise RuntimeError("no eligible test dates in locked interval")

    zero_test_keys: pd.DataFrame | None = None
    if a.reference_zero_shot_dir:
        zero_keys_all = _load_reference_keys(Path(a.reference_zero_shot_dir), model="Kronos")
        zero_test_keys = zero_keys_all[zero_keys_all["as_of_date"].isin(pd.to_datetime(test_dates))].drop_duplicates()
        reference_dates = [x.date().isoformat() for x in zero_test_keys["as_of_date"].drop_duplicates().sort_values()]
        pair_test = pairs[pairs["split"] == "test"][["stock", "as_of_date"]].copy()
        pair_test["stock_key"] = pair_test["stock"].map(canonical_stock)
        pair_test["as_of_date"] = pd.to_datetime(pair_test["as_of_date"]).dt.normalize()
        expected_keys = set(map(tuple, pair_test[["as_of_date", "stock_key"]].itertuples(index=False, name=None)))
        actual_keys = set(map(tuple, zero_test_keys[["as_of_date", "stock_key"]].itertuples(index=False, name=None)))
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing or extra:
            raise RuntimeError(f"test keys differ from prior Kronos zero-shot: missing={missing[:5]}, extra={extra[:5]}")

    pairs.to_csv(out / "eligible_stock_date_pairs.csv", index=False)
    windows.to_csv(out / "date_windows.csv", index=False)
    pd.DataFrame({"as_of_date": calendar.date.astype(str), "calendar_position": range(len(calendar))}).to_csv(out / "master_calendar.csv", index=False)
    summary = {
        "status": "complete",
        "experiment_type": "continual_learning_ranking_head",
        "groups": ["A", "B", "C"],
        "stocks_loaded": len(frames),
        "calendar_dates": len(calendar),
        "calendar_hash": date_hash(list(calendar)),
        "test_start": TEST_START,
        "test_end": TEST_END,
        "test_dates": len(test_dates),
        "test_date_hash": date_hash([pd.Timestamp(x) for x in test_dates]),
        "lookback": a.lookback,
        "pred_len": a.pred_len,
        "min_cross_section": a.min_cross_section,
        "dates_by_split": windows.groupby("split")["as_of_date"].nunique().to_dict(),
        "rows_by_split": pairs.groupby("split").size().to_dict(),
        "reference_zero_shot_dates_checked": reference_dates is not None,
        "reference_adapter_dates_checked": adapter_dates is not None,
        "locked_test_stock_date_keys": int((pairs["split"] == "test").sum()),
        "locked_test_key_hash": key_hash(pd.DataFrame({
            "as_of_date": pd.to_datetime(pairs.loc[pairs["split"] == "test", "as_of_date"]),
            "stock_key": pairs.loc[pairs["split"] == "test", "stock"].map(canonical_stock),
        })),
        "adapter_reference_recent50_dates": adapter_dates,
    }
    (out / "panel_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()






