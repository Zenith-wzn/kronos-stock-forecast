import json
from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parents[2]
cache = project / 'train3' / 'abcd_hidden_cache_strict_20260820'
smoke_root = project / 'train3' / 'kronos_csi300_continual_abc_recent50_smoke_single_20260824_seed100'
panel = smoke_root / 'panel_smoke'
report = {'cache': str(cache), 'panel': str(panel), 'splits': {}, 'matching_panel_summaries': []}
for split in ('train','val'):
    m = pd.read_csv(cache / f'meta_{split}.csv')
    p = pd.read_csv(panel / 'eligible_stock_date_pairs.csv', usecols=['split','as_of_date','stock'])
    p = p[p['split'].eq(split)].copy()
    for frame in (m,p):
        frame['as_of_date'] = pd.to_datetime(frame['as_of_date']).dt.strftime('%Y-%m-%d')
        frame['stock'] = frame['stock'].astype(str)
    mk = set(zip(m.as_of_date,m.stock)); pk=set(zip(p.as_of_date,p.stock))
    md=sorted(m.as_of_date.unique()); pdts=sorted(p.as_of_date.unique())
    inter_dates=sorted(set(md)&set(pdts))
    recent_dates=inter_dates[-3 if split=='train' else -2:]
    report['splits'][split] = {
        'cache_columns': list(m.columns), 'cache_rows':len(m),'panel_rows':len(p),
        'cache_dates':len(md),'panel_dates':len(pdts), 'cache_first':md[0], 'cache_last':md[-1],
        'panel_first':pdts[0], 'panel_last':pdts[-1], 'missing_keys':len(pk-mk),'extra_keys':len(mk-pk),
        'missing_dates':sorted(set(pdts)-set(md))[:20], 'extra_dates':sorted(set(md)-set(pdts))[:20],
        'intersection_dates':len(inter_dates),'recent_intersection_dates':recent_dates,
        'recent_intersection_rows':int(m[m.as_of_date.isin(recent_dates)].shape[0]),
        'recent_extra_vs_panel':len(set(zip(m[m.as_of_date.isin(recent_dates)].as_of_date,m[m.as_of_date.isin(recent_dates)].stock))-pk),
    }
for path in (project/'train3').rglob('panel_summary.json'):
    try:
        obj=json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        continue
    rows=obj.get('rows_by_split',{})
    if rows.get('train')==455166 and rows.get('val')==31575:
        report['matching_panel_summaries'].append({'path':str(path),'summary':obj})
print(json.dumps(report,ensure_ascii=False,indent=2,default=str))

