"""Fixed protocol: reject configuration edits that the implementation cannot honor."""
import copy

EXPECTED = {'aer': {'annualization_days': 238,
         'close_cost': 0.0015,
         'drop_n': 5,
         'min_hold_days': 5,
         'minimum_fee': 5,
         'open_cost': 0.001,
         'top_k': 50},
 'benchmark': {'atol': 1e-05,
               'batches': [4, 8, 16],
               'fallback_batches': [2, 1],
               'rtol': 0.0001,
               'stocks_per_day': 32,
               'threads': [4, 8],
               'validation_days': 5},
 'cross_learning': False,
 'dataset': 'CSI300',
 'device': 'cpu',
 'dtype': 'float32',
 'expected_dates': 226,
 'expected_rows': 66103,
 'future_covariates': [],
 'lookback': 90,
 'model': 'chronos2',
 'ordinary_portfolio': {'cost_rate': 0.0015, 'drop_n': 5, 'min_hold_days': 5, 'top_k': 50},
 'past_covariates': ['open', 'high', 'low', 'volume'],
 'point_estimator': 'q0.5',
 'pred_len': 10,
 'quantiles': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
 'sample_count': None,
 'seed': 100,
 'target': 'close',
 'temperature': None,
 'test_end': '2026-06-05',
 'test_start': '2025-07-01',
 'timesfm_options': {'make_positive': False,
                     'padding_mode': 'none',
                     'sort_quantiles': False,
                     'use_symmetric_averaging': False,
                     'use_znorm': False},
 'top_p': None,
 'train_end': '2024-12-15',
 'training': False,
 'val_end': '2025-06-14',
 'val_start': '2025-01-02'}

def validate_config(config, model):
    expected = copy.deepcopy(EXPECTED)
    expected["model"] = model
    if config != expected:
        changed = sorted(k for k in set(config) | set(expected) if config.get(k) != expected.get(k))
        raise ValueError(f"Unsupported protocol change: {changed}; use a new audited experiment")
