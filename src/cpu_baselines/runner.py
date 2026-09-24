from __future__ import annotations
import argparse
import gc
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from .common import ROOT, Panel, atomic_csv, atomic_json, digest, read_json, require_keys, sha256


class MemoryMeter:
    def __enter__(self):
        self.stop = threading.Event(); self.peak = 0; self.low_available = psutil.virtual_memory().available
        def sample():
            process = psutil.Process()
            while not self.stop.is_set():
                self.peak = max(self.peak, process.memory_info().rss)
                self.low_available = min(self.low_available, psutil.virtual_memory().available)
                self.stop.wait(.1)
        self.thread = threading.Thread(target=sample, daemon=True); self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop.set(); self.thread.join()


def memory_error(error):
    text = str(error).lower()
    return isinstance(error, MemoryError) or any(t in text for t in ('out of memory', 'cannot allocate memory', "can't allocate memory", 'not enough memory'))


def compare_predictions(reference, actual):
    if not np.allclose(reference, actual, rtol=1e-4, atol=1e-5, equal_nan=False):
        raise ValueError(f'Batch/order inconsistency; max absolute difference={np.max(np.abs(reference-actual))}')
    return float(np.max(np.abs(reference-actual)))


def predict_rows(model, panel, rows, batch_size):
    result = []
    for start in range(0, len(rows), batch_size):
        result.append(model.predict(panel.inputs(rows.iloc[start:start+batch_size]), batch_size))
    return np.concatenate(result)


def benchmark(model, panel, home, fingerprint):
    import torch
    records = []; reference = None
    samples = panel.validation
    atomic_csv(home/'results/benchmark_samples.csv', samples)
    for threads in (4, 8):
        torch.set_num_threads(threads)
        for batch in (4, 8, 16):
            row = {'threads': threads, 'batch_size': batch, 'samples': len(samples)}
            try:
                with MemoryMeter() as memory:
                    model.predict(panel.inputs(samples.iloc[:batch]), batch)
                    start = time.perf_counter()
                    out = predict_rows(model, panel, samples, batch)
                    elapsed = time.perf_counter()-start
                if not np.isfinite(out).all() or (out[:, :, 4] <= 0).any():
                    np.save(home/f'results/benchmark_invalid_{threads}_{batch}.npy', out)
                    raise ValueError('Invalid benchmark output saved; no clipping or sample removal')
                if reference is None:
                    reference = out.copy()
                delta = compare_predictions(reference, out)
                # Verify permutation / separation of stock tasks with a bounded extra batch.
                original = panel.inputs(samples.iloc[:4])
                perm = model.predict(original[::-1], batch)[::-1]
                compare_predictions(reference[:4], perm)
                row.update(status='ok', seconds=elapsed, windows_per_second=len(samples)/elapsed,
                           estimated_test_hours=66103*elapsed/len(samples)/3600,
                           peak_rss_bytes=memory.peak, minimum_available_bytes=memory.low_available,
                           max_absolute_difference=delta)
            except Exception as error:
                row.update(status='oom' if memory_error(error) else 'error', error=repr(error))
                if not memory_error(error):
                    records.append(row); atomic_json(home/'results/benchmark.json', {'fingerprint': fingerprint, 'trials': records})
                    raise
                gc.collect()
            records.append(row)
            atomic_json(home/'results/benchmark.json', {'fingerprint': fingerprint, 'trials': records})
            print(json.dumps(row), flush=True)
    success = [r for r in records if r['status'] == 'ok']
    if not success:
        torch.set_num_threads(4)
        for batch in (2, 1):
            try:
                with MemoryMeter() as memory:
                    model.predict(panel.inputs(samples.iloc[:batch]), batch)
                    start = time.perf_counter(); out = predict_rows(model, panel, samples, batch); elapsed = time.perf_counter()-start
                if not np.isfinite(out).all() or (out[:, :, 4] <= 0).any():
                    raise ValueError('Invalid fallback predictions')
                compare_predictions(out[:4], predict_rows(model, panel, samples.iloc[:4].iloc[::-1], batch)[::-1])
                row = dict(status='ok', threads=4, batch_size=batch, samples=len(samples), seconds=elapsed,
                           windows_per_second=len(samples)/elapsed, estimated_test_hours=66103*elapsed/len(samples)/3600,
                           peak_rss_bytes=memory.peak, minimum_available_bytes=memory.low_available)
                records.append(row); success.append(row); break
            except Exception as error:
                records.append({'batch_size': batch, 'threads': 4, 'status': 'oom' if memory_error(error) else 'error', 'error': repr(error)})
                if not memory_error(error): raise
    if not success:
        raise MemoryError('CPU inference failed at batch size 1')
    best = max(success, key=lambda r: r['windows_per_second'])
    saved = {'fingerprint': fingerprint, 'trials': records, 'selected': best,
             'selection_uses_labels': False, 'tolerance': {'rtol': 1e-4, 'atol': 1e-5}}
    atomic_json(home/'results/benchmark.json', saved)
    return best


def complete_day(path, rows, fingerprint):
    meta = path.with_suffix('.meta.json')
    if not path.exists() or not meta.exists(): return False
    record = read_json(meta)
    if record['fingerprint'] != fingerprint or record['sha256'] != sha256(path):
        raise ValueError(f'Unsafe resume: config/data/content mismatch {path}')
    df = pd.read_csv(path)
    require_keys(df, rows)
    if not df.status.eq('ok').all(): raise ValueError('Prior date contains invalid predictions')
    return True


def execute(model, panel, home, fingerprint, revision, selected):
    import torch
    torch.set_num_threads(selected['threads'])
    run = home/'results'; by_date = run/'predictions/by_date'
    by_date.mkdir(parents=True, exist_ok=True)
    known_dates = set(panel.test.as_of_date)
    if any(p.stem not in known_dates for p in by_date.glob('*.csv')):
        raise ValueError('Unexpected date files in output')
    active_batch = int(selected['batch_size']); start = time.perf_counter(); done = 0
    with MemoryMeter() as memory:
        for date, rows in panel.test.sort_values(['as_of_date', 'stock']).groupby('as_of_date', sort=True):
            path = by_date/f'{date}.csv'
            if complete_day(path, rows, fingerprint):
                done += 1; continue
            t = time.perf_counter()
            while True:
                try:
                    forecasts = predict_rows(model, panel, rows, active_batch)
                    break
                except Exception as error:
                    if not memory_error(error) or active_batch == 1: raise
                    active_batch = max(1, active_batch//2); gc.collect()
                    print(f'Memory fallback to batch={active_batch}', flush=True)
            frame = panel.records(rows, forecasts, home.name, revision)
            if not frame.status.eq('ok').all():
                atomic_csv(run/'failures'/f'{date}.csv', frame)
                raise ValueError(f'Invalid predictions at {date}; raw outputs preserved')
            require_keys(frame, rows)
            atomic_csv(path, frame)
            atomic_json(path.with_suffix('.meta.json'), {'fingerprint': fingerprint, 'sha256': sha256(path),
                'rows': len(frame), 'batch_size': active_batch, 'threads': selected['threads'], 'seconds': time.perf_counter()-t})
            done += 1
            progress = {'status': 'running', 'model': home.name, 'dates_completed': done, 'dates_total': 226,
                        'last_date': date, 'elapsed_session_seconds': time.perf_counter()-start,
                        'peak_rss_bytes': memory.peak, 'batch_size': active_batch}
            atomic_json(run/'progress.json', progress)
            print(json.dumps(progress), flush=True)
        frames = [pd.read_csv(p) for p in sorted(by_date.glob('*.csv'))]
        all_predictions = pd.concat(frames, ignore_index=True)
        require_keys(all_predictions, panel.test)
        atomic_csv(run/'predictions.csv', all_predictions)
    summary = {'status': 'predictions_complete', 'model': home.name, 'rows': len(all_predictions), 'dates': done,
               'fingerprint': fingerprint, 'peak_rss_bytes': memory.peak,
               'minimum_available_bytes': memory.low_available, 'elapsed_session_seconds': time.perf_counter()-start,
               'prediction_seconds_total': sum(read_json(p)['seconds'] for p in by_date.glob('*.meta.json')),
               'note': 'Current checkpoint retrospective evaluation; current constituent survivorship bias.'}
    atomic_json(run/'progress.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', choices=['chronos2', 'timesfm3'], required=True)
    p.add_argument('--stage', choices=['preflight','benchmark','run','all'], default='all'); a = p.parse_args()
    home = ROOT/a.model; out = home/'results'; out.mkdir(parents=True, exist_ok=True)
    # A process-level lock prevents concurrent writers. OS releases it after a crash.
    lock = open(out/'run.lock', 'a+b')
    if os.name == 'nt':
        import msvcrt
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        from .protocol import validate_config
        cfg = read_json(home/'config.json')
        validate_config(cfg, a.model)
        if (out/'failure.json').exists():
            archived = out/'diagnostics'/f'failure-{time.time_ns()}.json'
            archived.parent.mkdir(exist_ok=True)
            (out/'failure.json').replace(archived)
        print('Preflight: verifying raw data, reference hashes, and all 66103 windows', flush=True)
        panel = Panel()
        atomic_json(out/'data_audit.json', {'fingerprint': panel.fingerprint, 'file_sha256': panel.hashes,
            'reference_sha256': panel.reference_hashes, 'test_rows': len(panel.test), 'test_dates': 226,
            'performance_rows': len(panel.validation), 'performance_dates': sorted(panel.validation.as_of_date.unique())})
        # Validate every history and label window without invoking either model.
        for _, rows in panel.test.groupby('as_of_date', sort=True):
            panel.inputs(rows)
        print('All historical windows verified', flush=True)
        if a.stage == 'preflight':
            print('Data preflight passed: 66103 windows / 226 dates', flush=True); return
        provenance = read_json(home/'provenance.json')
        for filename, expected in provenance['weight_sha256'].items():
            if sha256(home/'weights'/filename) != expected: raise ValueError('Weight integrity mismatch')
        code = {str(f.relative_to(ROOT)): sha256(f) for f in sorted((ROOT/'src/cpu_baselines').glob('*.py'))}
        for file in ('compute_csi800_aligned_retest_metrics.py', 'compute_kronos_csi300_cross_section_metrics.py'):
            path = ROOT/'scripts/evaluation'/file
            code[str(path.relative_to(ROOT))] = sha256(path)
        path = ROOT/'continual_abcs/portfolio.py'
        code[str(path.relative_to(ROOT))] = sha256(path)
        fingerprint = digest({'configuration': cfg, 'data': panel.fingerprint, 'provenance': provenance, 'code': code})
        identity = {'fingerprint': fingerprint, 'config': cfg, 'provenance': provenance, 'code_sha256': code}
        ident_path = out/'run_identity.json'
        if ident_path.exists() and read_json(ident_path)['fingerprint'] != fingerprint:
            raise ValueError('Run identity changed; use a separate output directory, do not reuse old predictions')
        atomic_json(ident_path, identity)
        import torch
        from .models import Forecaster
        random.seed(100); np.random.seed(100); torch.manual_seed(100)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        started = time.perf_counter()
        model = Forecaster(a.model, home/'weights')
        atomic_json(out/'model_load.json', {'seconds': time.perf_counter()-started,
                    'parameters': sum(p.numel() for p in model.pipeline.model.parameters()),
                    'cpu': True, 'dtype': 'float32', 'rss_bytes': psutil.Process().memory_info().rss})
        bp = out/'benchmark.json'
        if bp.exists() and read_json(bp).get('fingerprint') == fingerprint and read_json(bp).get('selected'):
            selected = read_json(bp)['selected']
        else:
            selected = benchmark(model, panel, home, fingerprint)
        print('SELECTED '+json.dumps(selected), flush=True)
        if a.stage == 'benchmark': return
        summary = execute(model, panel, home, fingerprint, provenance['revision'], selected)
        # Free weights before dataframe-heavy evaluation.
        del model; gc.collect()
        from .evaluate import evaluate_run
        evaluate_run(home, panel)
        summary['status'] = 'complete'
        summary['evaluation_completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        atomic_json(out/'summary.json', summary)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        atomic_json(out/'failure.json', {'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        lock.close()

if __name__ == '__main__': main()
