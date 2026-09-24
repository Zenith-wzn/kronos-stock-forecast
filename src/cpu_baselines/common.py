from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'train3/csi300_csi800_aligned_seed100'
DATA = ROOT / 'data/csi300_daily/csi300_daily'
COVARIATES = ['open', 'high', 'low', 'volume']
VALUES = ['close', *COVARIATES]
QUANTILES = [i / 10 for i in range(1, 10)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def keyset(frame):
    keys = list(zip(frame.as_of_date.astype(str), frame.stock.astype(str)))
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate stock/date keys')
    return set(keys)


def require_keys(frame, expected):
    actual, wanted = keyset(frame), keyset(expected)
    if actual != wanted:
        raise ValueError(f'Key mismatch: missing={len(wanted-actual)}, extra={len(actual-wanted)}')


class Panel:
    def __init__(self, data=DATA, reference=REFERENCE):
        self.data, self.reference = Path(data), Path(reference)
        self.calendar = pd.read_csv(self.reference / 'master_calendar.csv').as_of_date.astype(str).tolist()
        self.positions = {d: i for i, d in enumerate(self.calendar)}
        if len(self.positions) != len(self.calendar) or self.calendar != sorted(self.calendar):
            raise ValueError('Invalid reference calendar')
        manifest = read_json(self.reference / 'dataset_manifest.json')['files']
        self.frames, self.hashes = {}, {}
        for record in manifest:
            path = self.data / record['file']
            actual_hash = sha256(path)
            if actual_hash != record['sha256']:
                raise ValueError(f'Historical data hash mismatch: {path}')
            self.hashes[record['file']] = actual_hash
            frame = pd.read_csv(path, dtype={'date': str, 'code': str})
            if frame.date.duplicated().any() or set(frame.code) != {record['stock']}:
                raise ValueError(f'Duplicate dates or wrong stock: {path}')
            self.frames[record['stock']] = frame.set_index('date').sort_index()
        windows = pd.read_csv(self.reference / 'date_windows.csv', dtype=str)
        if windows.as_of_date.duplicated().any():
            raise ValueError('Reference windows must contain one row per signal date')
        val_dates = sorted(windows.loc[windows.split == 'val', 'as_of_date'])[:5]
        validation, tests = [], []
        for chunk in pd.read_csv(self.reference / 'eligible_stock_date_pairs.csv', dtype=str, chunksize=10000):
            tests.append(chunk[chunk.split == 'test'])
            validation.append(chunk[(chunk.split == 'val') & chunk.as_of_date.isin(val_dates)])
        self.test = pd.concat(tests, ignore_index=True).sort_values(['as_of_date','stock']).reset_index(drop=True)
        self.validation = pd.concat(validation, ignore_index=True).sort_values(['as_of_date','stock']).groupby('as_of_date', sort=True).head(32).reset_index(drop=True)
        raw = pd.read_csv(self.reference / 'raw_signal.csv', dtype={'stock': str, 'as_of_date': str})
        require_keys(self.test, raw)
        if len(self.test) != 66103 or self.test.as_of_date.nunique() != 226:
            raise ValueError('Expected 66103 windows / 226 signal dates')
        if self.test.as_of_date.min() != '2025-07-01' or self.test.as_of_date.max() != '2026-06-05':
            raise ValueError('Unexpected test period')
        if len(self.validation) != 160:
            raise ValueError('Expected 160 performance examples')
        check = self.test.merge(windows.drop(columns='stock'), on='as_of_date', suffixes=('', '_reference'), validate='many_to_one')
        for col in ['split','input_dates','label_dates','input_calendar_hash','label_calendar_hash']:
            if not check[col].eq(check[col+'_reference']).all():
                raise ValueError(f'Eligibility/calendar disagreement: {col}')
        self.reference_hashes = {name: sha256(self.reference / name) for name in (
            'config.json', 'dataset_manifest.json', 'master_calendar.csv', 'date_windows.csv',
            'eligible_stock_date_pairs.csv', 'raw_signal.csv')}
        self.fingerprint = digest({'data': self.hashes, 'reference': self.reference_hashes})

    def windows(self, row):
        p = self.positions[row.as_of_date]
        past, future = self.calendar[p-89:p+1], self.calendar[p+1:p+11]
        if len(past) != 90 or len(future) != 10:
            raise ValueError('Window length mismatch')
        if past != row.input_dates.split('|') or future != row.label_dates.split('|'):
            raise ValueError(f'Calendar mismatch: {row.stock}/{row.as_of_date}')
        if past[0] != row.input_start_date or past[-1] != row.input_end_date or future[-1] != row.label_end_date or future[0] != row.label_start_date:
            raise ValueError('Window boundaries mismatch')
        for dates, column in ((past, 'input_calendar_hash'), (future, 'label_calendar_hash')):
            if hashlib.sha256('|'.join(dates).encode()).hexdigest() != getattr(row, column):
                raise ValueError('Window calendar hash mismatch')
        return past, future

    def inputs(self, rows):
        # Return only history. Labels are fetched by records(), after inference.
        result = []
        for row in rows.itertuples(index=False):
            past, _ = self.windows(row)
            x = self.frames[row.stock].loc[past, VALUES].to_numpy(dtype=np.float32).T.copy()
            if x.shape != (5, 90) or not np.isfinite(x).all() or (x[:4] <= 0).any() or (x[4] < 0).any():
                raise ValueError(f'Invalid input: {row.stock}/{row.as_of_date}')
            result.append(x)
        return result

    def records(self, rows, forecasts, model, revision):
        forecasts = np.asarray(forecasts)
        if forecasts.shape != (len(rows), 10, 9):
            raise ValueError(f'Unexpected forecast shape {forecasts.shape}')
        result = []
        for row, qs in zip(rows.itertuples(index=False), forecasts):
            _, future = self.windows(row)
            f = self.frames[row.stock]
            current = float(f.loc[row.as_of_date, 'close'])
            actual = f.loc[future, 'close'].to_numpy(float)
            if not np.isfinite(actual).all() or (actual <= 0).any():
                raise ValueError('Invalid ground truth')
            pred = qs[:, 4].astype(float)
            status = 'ok' if np.isfinite(qs).all() and (pred > 0).all() else 'invalid_prediction'
            r = {'method': model, 'seed': 100, 'stock': row.stock, 'as_of_date': row.as_of_date,
                 'label_end_date': future[-1], 'label_start_date': future[0], 'label_dates': '|'.join(future),
                 'input_start_date': row.input_start_date, 'input_end_date': row.as_of_date,
                 'current_close': current, 'model_version': revision, 'status': status,
                 'point_estimator': 'q0.5', 'actual_signal': float(np.mean(actual/current-1)),
                 'pred_signal': float(np.mean(pred/current-1)), 'next_open': float(f.loc[future[0], 'open']),
                 'next_close': float(actual[0]), 'next_date': future[0]}
            r['pred_path_mean_return'] = r['pred_signal']
            r['actual_path_mean_return'] = r['actual_signal']
            r['pred_endpoint_return'] = float(pred[-1]/current-1)
            r['actual_endpoint_return'] = float(actual[-1]/current-1)
            for h in range(10):
                r[f'label_date_{h+1}'] = future[h]
                r[f'pred_close_{h+1}'], r[f'actual_close_{h+1}'] = float(pred[h]), float(actual[h])
                r[f'pred_return_{h+1}'], r[f'actual_return_{h+1}'] = float(pred[h]/current-1), float(actual[h]/current-1)
                for j, q in enumerate(QUANTILES):
                    r[f'pred_q{q:.1f}_{h+1}'] = float(qs[h, j])
            result.append(r)
        return pd.DataFrame(result)
