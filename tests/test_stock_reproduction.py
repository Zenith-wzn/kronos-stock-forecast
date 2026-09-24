from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.stock_reproduction.common import strict_window, target_for_row


def sample_frame():
    dates = pd.date_range("2024-01-01", periods=105, freq="D").strftime("%Y-%m-%d")
    frame = pd.DataFrame({"close": np.arange(1, 106, dtype=float), "open": 1.0, "high": 2.0, "low": 0.5, "volume": 10.0}, index=dates)
    row = pd.Series({"as_of_date": dates[89], "input_dates": "|".join(dates[:90]), "label_dates": "|".join(dates[90:100])})
    return frame, row


def test_strict_window_ignores_outside_information():
    frame, row = sample_frame(); original = strict_window(frame, row, ["close", "volume"])
    changed = frame.copy(); changed.iloc[100:, changed.columns.get_loc("close")] = 999999
    np.testing.assert_array_equal(original, strict_window(changed, row, ["close", "volume"]))


def test_strict_window_rejects_non_90_day_window():
    frame, row = sample_frame(); row.input_dates = "|".join(row.input_dates.split("|")[1:])
    with pytest.raises(ValueError, match="90-day"): strict_window(frame, row, ["close"])


def test_path_target_is_mean_of_ten_future_returns():
    frame, row = sample_frame(); expected = np.mean(np.arange(91, 101)/90-1)
    assert target_for_row(frame, row) == pytest.approx(expected)


def test_corrected_rank_loss_orders_targets():
    torch = pytest.importorskip("torch")
    from src.stock_reproduction.models import corrected_pairwise_rank_loss, upstream_sspt_rank_loss
    target = torch.tensor([-1.0, 0.0, 1.0]); correct = target.clone(); reverse = -target
    assert corrected_pairwise_rank_loss(correct, target) == 0
    assert corrected_pairwise_rank_loss(reverse, target) > 0
    assert upstream_sspt_rank_loss(reverse, target) != corrected_pairwise_rank_loss(reverse, target)


def test_configs_lock_cpu_seed_and_protocol():
    import json
    root = Path(__file__).resolve().parents[1]
    for name in ("master", "sspt"):
        config = json.loads((root/name/"config.json").read_text(encoding="utf-8"))
        assert config["device"] == "cpu" and config["seed"] == 100 and config["threads"] == 8
        assert config["lookback"] == 90 and config["horizon"] == 10

def test_sspt_feature_layout_matches_upstream_order():
    import json
    from src.stock_reproduction.sspt_aligned import CSI300SSPT
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/"sspt/config.json").read_text(encoding="utf-8"))
    data = CSI300SSPT(root/"sspt", config, smoke=True)
    assert data.features.shape == (3, 120, 25)
    # Each OHLCV field owns five adjacent columns: MA5/10/20/30/raw.
    assert np.isfinite(data.features).all()
    assert data.features[:, :4, 0].sum() == 0
    assert data.features[:, :9, 1].sum() == 0
    assert data.features[:, :19, 2].sum() == 0
    assert data.features[:, :29, 3].sum() == 0
