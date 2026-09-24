from pathlib import Path
import pandas as pd
from .common import PRICE_ROOT, REFERENCE, atomic_json, ensure_layout, load_pairs, provenance, sha256

def audit_inputs(home: Path, model: str, config: dict) -> dict:
    ensure_layout(home); pairs = load_pairs()
    fields = set(pd.read_csv(next(PRICE_ROOT.glob("*.csv")), nrows=2).columns)
    blockers, external = [], config.get("external_data", {})
    if model == "master":
        if not {"amount", "vwap"} & fields: blockers.append({"code": "stock_vwap_source_missing", "detail": "OHLCV exists, but amount/VWAP is absent; Alpha158 VWAP factors cannot be reproduced without inventing VWAP."})
        for name in ("csi300", "csi500", "csi800"):
            path = home / external.get("market_indices", {}).get(name, {}).get("path", "__missing__")
            if not path.is_file(): blockers.append({"code": f"{name}_market_series_missing", "detail": str(path)})
    else:
        path = home / external.get("point_in_time_industry", {}).get("path", "__missing__")
        if not path.is_file():
            blockers.append({"code": "industry_classification_missing", "detail": "No industry membership source is configured."})
        else:
            industry = pd.read_csv(path, dtype=str).fillna("")
            required = {"stock", "industry", "source", "snapshot_date", "classification_policy"}
            if required - set(industry): blockers.append({"code": "industry_schema_invalid", "detail": f"Missing columns: {sorted(required-set(industry))}"})
            else:
                expected = set(pd.read_csv(REFERENCE / "eligible_stock_date_pairs.csv", dtype=str).stock.unique())
                actual = set(industry.loc[industry.industry.ne(""), "stock"])
                if industry.stock.duplicated().any() or actual != expected: blockers.append({"code": "industry_coverage_invalid", "detail": f"missing={len(expected-actual)}, extra={len(actual-expected)}, duplicate={bool(industry.stock.duplicated().any())}"})
                if industry.industry.nunique() < 5: blockers.append({"code": "industry_cardinality_invalid", "detail": str(industry.industry.nunique())})
    result = {"status": "ready" if not blockers else "blocked", "model": model, "protocol": {"lookback": 90, "horizon": 10, "train_end": "2024-12-15", "validation": ["2025-01-02", "2025-06-14"], "test": ["2025-07-01", "2026-06-05"], "test_rows": int((pairs.split == "test").sum()), "test_dates": int(pairs.loc[pairs.split == "test", "as_of_date"].nunique())}, "source": {"price_root": str(PRICE_ROOT), "reference_pairs_sha256": sha256(REFERENCE / "eligible_stock_date_pairs.csv"), "available_columns": sorted(fields)}, "blockers": blockers, "disclosures": ["Current-constituent historical panel has survivorship and constituent look-ahead bias.", "Rolling features are computed inside each locked 90-day window.", "Aligned experiment; not a reproduction of paper-reported metrics."], "provenance": provenance(home, model)}
    atomic_json(home / "results/data_audit.json", result); atomic_json(home / "provenance.json", result["provenance"])
    return result
