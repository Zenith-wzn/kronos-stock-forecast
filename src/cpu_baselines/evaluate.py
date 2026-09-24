"""Reuse historical metric functions without altering their formulas or old files."""
from __future__ import annotations
import importlib.util
import types
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, REFERENCE, atomic_csv, atomic_json, keyset, require_keys, read_json, sha256


def legacy_evaluator():
    path = ROOT/'scripts/evaluation/compute_csi800_aligned_retest_metrics.py'
    source = path.read_text(encoding='utf-8-sig')
    # Existing source has Python 3.12-only f-string quoting in an unused file finder.
    # Only replace inner quotes to make the exact same expressions valid on CPython 3.11.
    source = source.replace("stock.replace('/','_')", 'stock.replace("/","_")')
    module = types.ModuleType('historical_metrics'); module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    module.DATASET = 'CSI300'
    return module


def detailed_evaluator():
    path = ROOT/'scripts/evaluation/compute_kronos_csi300_cross_section_metrics.py'
    spec = importlib.util.spec_from_file_location('historical_ranking', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def add_execution(frame, panel):
    frame = frame.copy(); rows = []
    for row in frame[['stock', 'as_of_date']].itertuples(index=False):
        frame_stock = panel.frames[row.stock]
        p = panel.positions[row.as_of_date]
        next_date = panel.calendar[p+1]
        rows.append((float(frame_stock.loc[next_date, 'open']), float(frame_stock.loc[next_date, 'close']),
                     float(frame_stock.loc[row.as_of_date, 'close'])))
    values = np.asarray(rows)
    frame['next_open'], frame['next_close'], frame['current_close'] = values.T
    frame['execution_return_closeclose'] = values[:, 1]/values[:, 2]-1
    frame['execution_return_opentoclose'] = values[:, 1]/values[:, 0]-1
    if not np.isfinite(frame[['execution_return_closeclose','execution_return_opentoclose']]).all().all():
        raise ValueError('Missing execution prices')
    return frame


def strict_policy_portfolio(frame, signal, mode, aer, legacy):
    """Supplement historical proxy with strict holding policy, unchanged legacy costs.

    This remains a daily equal-weight return proxy, not an executable market simulator.
    Missing-universe forced exits are disclosed; they cannot obey a minimum hold.
    """
    from continual_abcs.portfolio import _target_holdings
    held, ages, records, audits = set(), {}, [], []
    retcol = 'execution_return_opentoclose' if aer else 'execution_return_closeclose'
    for date, group in frame.groupby('as_of_date', sort=True):
        ordered = group.sort_values([signal, 'stock'], ascending=[False, True])
        ranked = ordered.stock.tolist()
        chosen = _target_holdings(ranked, held, top_k=50, drop_n=5, min_holding_days=5, age=ages)
        sold, bought = held-chosen, chosen-held
        forced = held-set(ranked)
        voluntary = sold-forced
        if len(voluntary)>5 or any(ages[s]<5 for s in voluntary):
            raise ValueError('Strict holding policy violated')
        for stock in sorted(sold):
            audits.append({'as_of_date':date,'stock':stock,'holding_days':ages[stock],
                           'forced_universe_exit':stock in forced,'mode':mode})
        returns = ordered.set_index('stock')[retcol]
        top = float(returns.loc[sorted(chosen)].mean())
        bottom = float(returns.loc[ranked[-50:]].mean()); benchmark = float(returns.mean())
        turnover = (len(sold)+len(bought))/max(1,len(chosen))
        cost = ((len(bought)*legacy.AER_OPEN_COST+len(sold)*legacy.AER_CLOSE_COST)/max(1,len(chosen))
                if aer else turnover*legacy.ORDINARY_COST)
        if aer:
            cost += legacy.AER_MIN_FEE*len(bought|sold)/legacy.AER_NOTIONAL/max(1,len(chosen))
        records.append({'dataset':'CSI300','experiment':group.experiment.iloc[0],
            'path_or_endpoint':mode,'prediction_date':date,'valid_stock_count':len(group),
            'holding_count':len(chosen),'turnover':turnover,
            'q5_minus_q1':legacy.quantile_diff(ordered,signal,retcol),
            'top_raw_return':top,'bottom_return':bottom,'benchmark_return':benchmark,
            'top_excess_return':top-benchmark,'top_net_return':top-cost,
            'long_short_return':top-bottom,'transaction_cost':cost,
            'voluntary_exits':len(voluntary),'forced_exits':len(forced),
            'execution_mode':'strict policy + original daily return/cost proxy'})
        ages = {stock:ages.get(stock,0)+1 for stock in chosen}; held=chosen
    daily=pd.DataFrame(records)
    summary=legacy.portfolio_summary(daily,aer)
    for record in summary:
        record.update(metric_status='strict_policy_original_cost_proxy', min_hold_strictly_enforced=True,
                      forced_universe_exits=int(daily.forced_exits.sum()), drop_is_sale_cap=True,
                      risk_degree_applied=False)
    return daily, pd.DataFrame(audits), summary


def metrics(frame, label, panel, directory):
    directory.mkdir(parents=True, exist_ok=True)
    legacy, ranking = legacy_evaluator(), detailed_evaluator()
    require_keys(frame, panel.test)
    # Prevent canonical() from concealing duplicate or invalid rows.
    required = ['pred_signal','actual_signal','pred_endpoint_return','actual_endpoint_return']
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError('Non-finite predictions: cannot silently exclude samples')
    frame = legacy.canonical(frame, label, False)
    frame = add_execution(frame, panel)
    point, directions, cross, psum, asum, pdaily, adaily = [], [], [], [], [], [], []
    for mode, pred, actual in [('Path','pred_signal','actual_signal'),('Endpoint','pred_endpoint_return','actual_endpoint_return')]:
        point.append({'mode': mode, **legacy.scalar(frame[pred], frame[actual])})
        directions.append({'mode': mode, **legacy.direction(frame[pred], frame[actual])})
        cross.append(legacy.daily_cross(frame, mode, pred, actual))
        for aer in (False, True):
            daily, error = legacy.simulate_portfolio(frame, pred, mode, aer)
            if error or len(daily) != 226: raise ValueError(f'Portfolio incomplete: {error}')
            (adaily if aer else pdaily).append(daily)
            (asum if aer else psum).extend(legacy.portfolio_summary(daily, aer))
            strict_daily, strict_trades, strict_summary = strict_policy_portfolio(frame, pred, mode, aer, legacy)
            prefix = f'strict_{mode.lower()}_{"aer" if aer else "ordinary"}'
            atomic_csv(directory/f'{prefix}_daily.csv', strict_daily)
            atomic_csv(directory/f'{prefix}_exits.csv', strict_trades)
            atomic_json(directory/f'{prefix}_summary.json', legacy.clean(strict_summary))
    for h in range(1, 11):
        point.append({'mode': f'h{h}', **legacy.scalar(frame[f'pred_return_{h}'], frame[f'actual_return_{h}'])})
    point.append({'mode':'Path_pooled', **legacy.scalar(frame[[f'pred_return_{h}' for h in range(1,11)]].to_numpy().ravel(), frame[[f'actual_return_{h}' for h in range(1,11)]].to_numpy().ravel())})
    for record in psum + asum:
        record.update(metric_status='historical_proxy', min_hold_strictly_enforced=False,
                      drop_is_sale_cap=False, risk_degree_applied=False)
    for name, table in [('point',pd.DataFrame(point)),('direction',pd.DataFrame(directions)),
                        ('daily_cross',pd.concat(cross)),('portfolio_summary',pd.DataFrame(psum)),
                        ('aer_summary',pd.DataFrame(asum)),('portfolio_daily',pd.concat(pdaily)),('aer_daily',pd.concat(adaily))]:
        atomic_csv(directory/f'{name}.csv', table)
    days, groups = [], []
    frame['pred_path_mean_return'] = frame.pred_signal
    frame['actual_path_mean_return'] = frame.actual_signal
    for _, g in frame.groupby('as_of_date', sort=True):
        row, qs = ranking.analyze_date(g, 5, 50); days.append(row); groups.extend(qs)
    daily = pd.DataFrame(days)
    stats = {c: ranking.describe(daily[c], 9, 5000, 10, 100+i) for i, c in enumerate(c for c in daily if c not in ['as_of_date','n_stocks'])}
    atomic_csv(directory/'ranking_daily.csv', daily)
    atomic_csv(directory/'quantile_daily.csv', pd.DataFrame(groups))
    atomic_json(directory/'ranking_summary.json', legacy.clean(stats))
    # Preserve historical portfolio implementation, but do not overstate its semantics.
    notes = {
        'source_sha256': {str(p.relative_to(ROOT)):sha256(p) for p in [
            ROOT/'scripts/evaluation/compute_csi800_aligned_retest_metrics.py',
            ROOT/'scripts/evaluation/compute_kronos_csi300_cross_section_metrics.py']},
        'historical_formula_reused': True,
        'limitations': [
            'Existing portfolio implementation is a proxy, not a full execution engine.',
            'MIN_HOLD=5 is retained, but historical keep logic does not strictly enforce a five-day minimum.',
            'DROP_N=5 is a rank-buffer parameter in the historical code, not a strict five-sales-per-day cap.',
            'Historical AER is excess-return proxy; its excess-return field does not subtract costs, while top_net_return does.',
            'Supplemental strict_* files enforce min_hold=5 and voluntary sell cap=5 with original proxy costs; universe exits separately audited.',
            'Historical risk_degree=0.95 is metadata, not applied by the reused portfolio formula.',
            'Current constituent backfill has survivorship/look-ahead bias.',
            'Checkpoint pretraining overlap is unverified; retrospective evaluation is not historical deployability.'
        ]}
    atomic_json(directory/'evaluation_provenance.json', notes)
    return {'model': label, 'rows':len(frame), 'days':226,
            'path_rankic':stats['spearman_rankic_path_mean']['mean'],
            'path_ic':stats['pearson_ic_path_mean']['mean'],
            'path_rankic_nw_t':stats['spearman_rankic_path_mean']['newey_west_tstat'],
            'path_rankic_ci95':stats['spearman_rankic_path_mean']['moving_block_bootstrap_mean_ci95'],
            'point_estimator':'Kronos sampled paths' if label=='Kronos' else 'q0.5'}


def evaluate_run(home, panel):
    output = home/'results/evaluation'
    own = pd.read_csv(home/'results/predictions.csv')
    summary = metrics(own, home.name, panel, output/home.name)
    reference_dir = REFERENCE/'experiments/A_zero_shot/predictions/by_date'
    files = sorted(reference_dir.glob('*.csv'))
    if not files: raise ValueError('Kronos baseline predictions unavailable')
    legacy = legacy_evaluator()
    raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    keyset(raw)  # Reject duplicate reference keys before explicit alignment.
    wanted = panel.test[['stock','as_of_date']]
    membership = raw.merge(wanted, on=['stock','as_of_date'], how='outer', indicator=True, validate='one_to_one')
    if membership['_merge'].eq('right_only').any():
        raise ValueError('Kronos baseline missing required test keys')
    extras = membership.loc[membership['_merge'].eq('left_only'), ['stock','as_of_date']]
    atomic_csv(output/'kronos_outside_fixed_test_keys.csv', extras)
    atomic_json(output/'kronos_alignment.json', {'raw_rows':len(raw), 'required_rows':len(wanted),
        'outside_fixed_keys':len(extras), 'reason':'Explicit restriction to historical eligible_stock_date_pairs test keys; no new-model samples removed'})
    raw = raw.merge(wanted, on=['stock','as_of_date'], how='inner', validate='one_to_one')
    require_keys(raw, panel.test)
    kronos = legacy.canonical(raw, 'Kronos', True)
    joined = own.merge(kronos, on=['stock','as_of_date'], suffixes=('_new','_old'), validate='one_to_one')
    for c in ['actual_signal','actual_endpoint_return']+[f'actual_return_{h}' for h in range(1,11)]:
        if not np.allclose(joined[c+'_new'], joined[c+'_old'], rtol=1e-7, atol=1e-9):
            raise ValueError(f'Baseline labels disagree: {c}')
    baseline = metrics(kronos, 'Kronos', panel, output/'Kronos')
    rows = [summary, baseline]
    other = 'timesfm3' if home.name=='chronos2' else 'chronos2'
    other_path = ROOT/other/'results/evaluation/comparison.json'
    if other_path.exists():
        # Only merge fully completed, data-identical runs.
        audit = read_json(ROOT/other/'results/data_audit.json')
        if audit['fingerprint'] == panel.fingerprint and (ROOT/other/'results/summary.json').exists():
            rows.extend(r for r in read_json(other_path)['models'] if r['model']==other)
    atomic_json(output/'comparison.json', legacy.clean({'models':rows, 'baseline_prediction_sha256': {p.name:sha256(p) for p in files}}))
    atomic_csv(output/'comparison.csv', pd.DataFrame(rows))
    lines = ['# CSI300 CPU zero-shot comparison', '', '| Model | Rows | Days | Path RankIC | Path IC | NW t |', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r.get('path_rankic_nw_t') is None: r['path_rankic_nw_t'] = float('nan')
    lines += [f"| {r['model']} | {r['rows']} | {r['days']} | {r['path_rankic']:.6f} | {r['path_ic']:.6f} | {r['path_rankic_nw_t']:.4f} |" for r in rows]
    lines += ['', 'All methods use the same 66,103 stock/date keys. New models use raw q0.5 paths; Kronos uses its existing sampled forecast.', '', 'Portfolio results reproduce historical proxy formulas, not a corrected trading backtest. See evaluation_provenance.json for limitations.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
