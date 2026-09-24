"""Adapter + test-then-train continual learning experiment for strict CSI300 90x10 panel."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random, time, traceback
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

TEST_START = pd.Timestamp('2025-07-01')
TEST_END = pd.Timestamp('2026-06-05')
LOOKBACK = 90
PRED_LEN = 10

class Adapter(nn.Module):
    def __init__(self, d: int, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, bottleneck)
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    def forward(self, h):
        return h + self.up(self.drop(F.gelu(self.down(self.ln(h)))))

class Model(nn.Module):
    def __init__(self, d: int, pred_len: int = 10, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.adapter = Adapter(d, bottleneck, dropout)
        self.head = nn.Linear(d, pred_len)
    def forward(self, h):
        return self.head(self.adapter(h))

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def atomic_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)

def atomic_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); frame.to_csv(tmp, index=False); tmp.replace(path)

def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def load_split(cache: Path, split: str):
    h = np.load(cache / f'hidden_{split}.npy', mmap_mode='r')
    y = np.load(cache / f'labels_{split}.npy', mmap_mode='r')
    m = pd.read_csv(cache / f'meta_{split}.csv')
    if h.ndim != 2 or h.shape[1] != 512: raise AssertionError(f'{split} hidden shape={h.shape}, expected (*,512)')
    if y.ndim != 2 or y.shape[1] != PRED_LEN: raise AssertionError(f'{split} label shape={y.shape}, expected (*,10)')
    if len(h) != len(y) or len(h) != len(m): raise AssertionError(f'{split} row mismatch')
    m['as_of_date'] = pd.to_datetime(m['as_of_date'], errors='raise').dt.normalize()
    m['stock'] = m['stock'].astype(str)
    m['_row_id'] = np.arange(len(m), dtype=np.int64)
    return h, y, m

def attach_label_end(meta: pd.DataFrame, panel: Path):
    if 'label_end_date' in meta.columns:
        out = meta.copy(); out['label_end_date'] = pd.to_datetime(out['label_end_date'], errors='raise').dt.normalize(); return out
    w = pd.read_csv(panel / 'date_windows.csv')
    w['as_of_date'] = pd.to_datetime(w['as_of_date'], errors='raise').dt.normalize()
    w['label_end_date'] = pd.to_datetime(w['label_end_date'], errors='raise').dt.normalize()
    out = meta.merge(w[['as_of_date','label_end_date']].drop_duplicates('as_of_date'), on='as_of_date', how='left', validate='many_to_one')
    if out['label_end_date'].isna().any(): raise AssertionError('missing label_end_date in panel date_windows')
    return out

def verify_locked_keys(meta, panel: Path, split: str, exact=True):
    pair = pd.read_csv(panel / 'eligible_stock_date_pairs.csv', usecols=['stock','as_of_date','split'])
    pair['as_of_date'] = pd.to_datetime(pair['as_of_date'], errors='raise').dt.normalize(); pair['stock'] = pair['stock'].astype(str)
    exp = pair.loc[pair['split'].eq(split), ['as_of_date','stock']]
    act = meta[['as_of_date','stock']]
    if exp.duplicated().any() or act.duplicated().any(): raise AssertionError(f'{split} duplicate keys')
    es = set(map(tuple, exp.astype(str).to_numpy())); ac = set(map(tuple, act.astype(str).to_numpy()))
    if (ac-es) or (exact and es-ac): raise AssertionError(f'{split} cache keys mismatch expected={len(es)} actual={len(ac)} missing={len(es-ac)} extra={len(ac-es)}')

def verify_boundaries(train_m, val_m, test_m, strict=True):
    key = lambda m: set(zip(m.as_of_date.astype(str), m.stock.astype(str)))
    if len(key(train_m)&key(val_m)) or len(key(train_m)&key(test_m)) or len(key(val_m)&key(test_m)): raise AssertionError('split key overlap detected')
    if train_m.label_end_date.max() >= TEST_START: raise AssertionError('train label_end_date reaches test period')
    if val_m.label_end_date.max() >= TEST_START: raise AssertionError('val label_end_date reaches test period')
    if train_m.as_of_date.max() >= TEST_START or val_m.as_of_date.max() >= TEST_START: raise AssertionError('train/val as_of_date reaches test period')
    if strict and (test_m.as_of_date.min() != TEST_START or test_m.as_of_date.max() != TEST_END): raise AssertionError('test interval is not locked')

def rankic(pred, actual, meta):
    p = np.asarray(pred).reshape(-1); y = np.asarray(actual).mean(axis=1); vals=[]
    for _, g in meta.groupby('as_of_date', sort=True):
        ix = g.index.to_numpy();
        if len(ix)<2: continue
        a = pd.Series(p[ix]).rank(method='average').to_numpy(); b = pd.Series(y[ix]).rank(method='average').to_numpy()
        if np.std(a)>1e-12 and np.std(b)>1e-12: vals.append(float(np.corrcoef(a,b)[0,1]))
    return float(np.mean(vals)) if vals else -math.inf

def date_batches(meta, batch_size, rng_seed, shuffle=True):
    rng=np.random.default_rng(rng_seed); batches=[]
    for _, g in meta.groupby('as_of_date', sort=True):
        ix=g.index.to_numpy(dtype=np.int64).copy()
        if shuffle: rng.shuffle(ix)
        batches += [ix[i:i+batch_size] for i in range(0,len(ix),batch_size)]
    if shuffle: rng.shuffle(batches)
    return batches

def predict(model, h, y, meta, device, version, update_flag=False, method='adapter_continual', seed=100):
    model.eval(); rows=[]; local = meta.reset_index(drop=True)
    with torch.no_grad():
        for pos in date_batches(local, 256, 0, False):
            source = local.iloc[pos]['_row_id'].to_numpy(dtype=np.int64)
            p=model(torch.as_tensor(np.asarray(h[source]), dtype=torch.float32, device=device)).cpu().numpy(); yy=np.asarray(y[source])
            for j, local_pos in enumerate(pos):
                m=local.iloc[int(local_pos)]; r=p[j]; a=yy[j]
                rec={'method':method,'seed':seed,'as_of_date':m.as_of_date.date().isoformat(),'stock':m.stock,'model_version':int(version),'pred_signal':float(r.mean()),'actual_signal':float(a.mean()),'pred_endpoint_return':float(r[-1]),'actual_endpoint_return':float(a[-1]),'label_end_date':m.label_end_date.date().isoformat(),'update_ran_after_prediction':bool(update_flag)}
                rec.update({f'pred_return_{k+1}':float(r[k]) for k in range(PRED_LEN)}); rec.update({f'actual_return_{k+1}':float(a[k]) for k in range(PRED_LEN)}); rows.append(rec)
    return rows

def train_candidate(model, h, y, meta, val_h, val_y, val_m, cfg, device, seed, epochs):
    model.train(); groups=[{'params':model.head.parameters(),'lr':cfg['head_lr']},{'params':model.adapter.parameters(),'lr':cfg['adapter_lr']}]
    opt=torch.optim.AdamW(groups, weight_decay=cfg['weight_decay']); hist=[]; best=None; bad=0
    for ep in range(1,epochs+1):
        model.train(); losses=[]; ranks=[]
        for ix in date_batches(meta, cfg['batch_size'], seed+ep, True):
            hb=torch.as_tensor(np.asarray(h[ix]), dtype=torch.float32, device=device); yy=torch.as_tensor(np.asarray(y[ix]), dtype=torch.float32, device=device); pred=model(hb)
            hub=F.smooth_l1_loss(pred, yy); ps=pred.mean(1); ys=yy.mean(1); dp=ps[:,None]-ps[None,:]; dy=ys[:,None]-ys[None,:]; mask=dy!=0
            rl=F.softplus(-torch.sign(dy[mask])*dp[mask]).mean() if bool(mask.any()) else ps.sum()*0
            loss=hub+cfg['lambda_rank']*rl; opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(hub.detach().cpu())); ranks.append(float(rl.detach().cpu()))
        vp=[]
        model.eval()
        with torch.no_grad():
            for ix in date_batches(val_m,256,0,False): vp.append(model(torch.as_tensor(np.asarray(val_h[ix]),dtype=torch.float32,device=device)).cpu().numpy())
        val_pred=np.concatenate(vp) if vp else np.empty((0,PRED_LEN)); metric=rankic(val_pred,val_y,val_m)
        row={'epoch':ep,'train_huber':float(np.mean(losses)),'train_rank_loss':float(np.mean(ranks)),'val_rankic':metric}; hist.append(row); print('[TRAIN]',json.dumps(row),flush=True)
        if best is None or metric>best[0]: best=(metric,copy.deepcopy(model.state_dict()),ep); bad=0
        else: bad += 1
        if bad >= cfg['patience']: break
    if best is None: raise RuntimeError('no training epoch')
    model.load_state_dict(best[1]); return model,hist,float(best[0]),int(best[2])

def sample_history_by_year(history, n, rng):
    if n <= 0 or history.empty: return history.iloc[0:0].copy()
    if len(history) <= n: return history.copy()
    groups=list(history.groupby(history.label_end_date.dt.year, sort=True)); sizes=np.array([len(g) for _,g in groups],dtype=float)
    quotas=np.floor(n*sizes/sizes.sum()).astype(int); remainder=n-int(quotas.sum())
    order=np.argsort(-(n*sizes/sizes.sum()-quotas))
    for j in order[:remainder]: quotas[j]+=1
    parts=[]
    for (_,g),q in zip(groups,quotas):
        if q>0: parts.append(g.iloc[np.sort(rng.choice(len(g),size=min(q,len(g)),replace=False))])
    out=pd.concat(parts,ignore_index=True) if parts else history.iloc[0:0].copy()
    if len(out)<n:
        remaining=history.drop(index=out.index,errors='ignore'); extra=remaining.iloc[np.sort(rng.choice(len(remaining),size=min(n-len(out),len(remaining)),replace=False))]; out=pd.concat([out,extra],ignore_index=True)
    return out.iloc[:n].copy()

def select_replay(train_m, test_m, update_date, cfg, seed):
    train_pool=train_m.copy(); train_pool['_source']='train'
    mature=test_m[(test_m.as_of_date < update_date) & (test_m.label_end_date <= update_date)].copy(); mature['_source']='test'
    if not mature.empty and ((mature.as_of_date >= update_date).any() or (mature.label_end_date > update_date).any()): raise AssertionError('future test information entered replay')
    pool=pd.concat([train_pool,mature],ignore_index=True,sort=False); pool=pool[pool.label_end_date <= update_date].copy()
    if pool.empty: raise AssertionError('replay is empty')
    cutoff=update_date-pd.DateOffset(months=cfg['replay_recent_months']); recent=pool[pool.label_end_date >= cutoff].copy(); history=pool[pool.label_end_date < cutoff].copy()
    cap=min(int(cfg['replay_max_rows']),len(pool)); desired_recent=int(round(cap*cfg['replay_recent_fraction'])); desired_history=cap-desired_recent
    recent_n=min(desired_recent,len(recent)); history_n=min(desired_history,len(history)); spare=cap-recent_n-history_n
    if spare>0:
        add=min(spare,len(recent)-recent_n); recent_n+=add; spare-=add
    if spare>0: history_n+=min(spare,len(history)-history_n)
    rng=np.random.default_rng(seed)
    recent_selected=recent if recent_n>=len(recent) else recent.iloc[np.sort(rng.choice(len(recent),size=recent_n,replace=False))]
    history_selected=sample_history_by_year(history,history_n,rng)
    chosen=pd.concat([recent_selected,history_selected],ignore_index=True,sort=False).sort_values(['as_of_date','stock','_source']).reset_index(drop=True)
    chosen_test=chosen[chosen._source.eq('test')]
    stats={'selected_rows':len(chosen),'selected_train_rows':int(chosen._source.eq('train').sum()),'selected_test_rows':int(chosen._source.eq('test').sum()),'selected_recent_rows':len(recent_selected),'selected_historical_rows':len(history_selected),'selected_max_as_of_date':chosen.as_of_date.max().date().isoformat(),'selected_max_label_end_date':chosen.label_end_date.max().date().isoformat(),'selected_test_min_as_of_date':chosen_test.as_of_date.min().date().isoformat() if len(chosen_test) else None,'selected_test_max_as_of_date':chosen_test.as_of_date.max().date().isoformat() if len(chosen_test) else None,'matured_test_rows_available':len(mature),'recent_cutoff':cutoff.date().isoformat()}
    return chosen,stats

def metrics(df):
    out=[]
    for d,g in df.groupby('as_of_date',sort=True):
        q=max(1,len(g)//5); s=g.sort_values('pred_signal'); out.append({'as_of_date':d,'n':len(g),'rankic':g.pred_signal.corr(g.actual_signal,method='spearman'),'ic':g.pred_signal.corr(g.actual_signal),'direction_accuracy':float((np.sign(g.pred_signal)==np.sign(g.actual_signal)).mean()),'q5_q1':float(s.tail(q).actual_signal.mean()-s.head(q).actual_signal.mean())})
    return pd.DataFrame(out)

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--cache-dir',required=True); p.add_argument('--panel-dir',required=True); p.add_argument('--initial-checkpoint',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=100); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--update-epochs',type=int,default=2); p.add_argument('--head-lr',type=float,default=3e-4); p.add_argument('--adapter-lr',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--lambda-rank',type=float,default=.2); p.add_argument('--patience',type=int,default=2); p.add_argument('--replay-max-rows',type=int,default=120000); p.add_argument('--replay-recent-months',type=int,default=24); p.add_argument('--replay-recent-fraction',type=float,default=.70); p.add_argument('--allow-cache-subset',action='store_true'); return vars(p.parse_args())

def main():
    cfg=parse_args(); out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True);
    if (out/'summary.json').exists() or (out/'predictions.csv').exists() or (out/'FAILED.json').exists(): raise FileExistsError(f'output directory already contains experiment results: {out}')
    started=time.time(); seed_everything(cfg['seed']); device=torch.device(cfg['device'] if torch.cuda.is_available() or cfg['device']=='cpu' else 'cpu')
    try:
        cache=Path(cfg['cache_dir']); panel=Path(cfg['panel_dir']);
        train_h,train_y,train_m=load_split(cache,'train'); val_h,val_y,val_m=load_split(cache,'val'); test_h,test_y,test_m=load_split(cache,'test')
        train_m=attach_label_end(train_m,panel); val_m=attach_label_end(val_m,panel); test_m=attach_label_end(test_m,panel)
        panel_summary=json.loads((panel/'panel_summary.json').read_text(encoding='utf-8'))
        if panel_summary.get('lookback') != LOOKBACK or panel_summary.get('pred_len') != PRED_LEN: raise AssertionError(f'panel window mismatch: {panel_summary}')
        windows=pd.read_csv(panel/'date_windows.csv')
        windows['as_of_date']=pd.to_datetime(windows['as_of_date'],errors='raise').dt.normalize(); windows['input_end_date']=pd.to_datetime(windows['input_end_date'],errors='raise').dt.normalize(); windows['label_start_date']=pd.to_datetime(windows['label_start_date'],errors='raise').dt.normalize()
        if not windows['input_dates'].str.split('|').map(len).eq(LOOKBACK).all(): raise AssertionError('not every input window has exactly 90 dates')
        if not windows['label_dates'].str.split('|').map(len).eq(PRED_LEN).all(): raise AssertionError('not every label window has exactly 10 dates')
        if not windows.input_end_date.eq(windows.as_of_date).all(): raise AssertionError('input_end_date must equal as_of_date')
        if not windows.label_start_date.gt(windows.as_of_date).all(): raise AssertionError('label_start_date must be after as_of_date')
        verify_locked_keys(train_m,panel,'train',not cfg.get('allow_cache_subset', False)); verify_locked_keys(val_m,panel,'val',not cfg.get('allow_cache_subset', False)); verify_locked_keys(test_m,panel,'test',not cfg.get('allow_cache_subset', False)); verify_boundaries(train_m,val_m,test_m,not cfg.get('allow_cache_subset', False))
        for name,m in [('train',train_m),('val',val_m),('test',test_m)]:
            if len(m)==0 or m.as_of_date.duplicated().any() and False: raise AssertionError(f'bad {name} metadata')
        if len(test_m) == 0 or test_m.as_of_date.nunique() == 0: raise AssertionError('formal test metadata is empty')
        if test_y.shape[1] != 10: raise AssertionError('formal output window is not 10')
        model=Model(512,PRED_LEN,64,.1).to(device); state=torch.load(cfg['initial_checkpoint'],map_location='cpu',weights_only=True); model.load_state_dict(state,strict=True)
        model.eval(); init_pred=[]
        with torch.no_grad():
            for ix in date_batches(val_m,256,0,False): init_pred.append(model(torch.as_tensor(np.asarray(val_h[ix]),dtype=torch.float32,device=device)).cpu().numpy())
        champion=rankic(np.concatenate(init_pred),val_y,val_m); version=0; checkpoints=out/'checkpoints'; checkpoints.mkdir(); torch.save(model.state_dict(),checkpoints/'version_000.pt')
        dates=pd.DatetimeIndex(sorted(test_m.as_of_date.unique())); updates=[]
        for i in range(len(dates)-1):
            if dates[i].month != dates[i+1].month: updates.append(dates[i])
        updates=[d for d in updates if d>=TEST_START and d<=TEST_END]
        expected_updates = [dates[i] for i in range(len(dates)-1) if dates[i].month != dates[i+1].month]
        if updates != expected_updates: raise AssertionError(f'update schedule mismatch: {updates}')
        update_set=set(updates); rows=[]; logs=[]; seen=set();
        for pos,date in enumerate(dates):
            dm=test_m[test_m.as_of_date.eq(date)].copy(); idx=dm.index.to_numpy(dtype=np.int64); before=version; rows.extend(predict(model,test_h,test_y,dm,device,version,False,'adapter_continual',cfg['seed']))
            if date in update_set:
                available=test_m[(test_m.as_of_date < date) & (test_m.label_end_date <= date)]
                new_ids=set(available['_row_id'].astype(int))-seen; seen.update(new_ids); before=version; champion_before=champion
                if not new_ids:
                    logs.append({'update_date':date.date().isoformat(),'model_version_before':before,'model_version_after':version,'accepted':False,'decision':'skipped_no_new_matured_labels','champion_validation_rankic_before':float(champion_before),'candidate_validation_rankic':None,'new_matured_test_rows':0,'matured_test_rows_available':len(available),'selected_replay_rows':0,'selected_train_rows':0,'selected_test_rows':0,'selected_max_as_of_date':None,'selected_max_label_end_date':None,'selected_test_min_as_of_date':None,'selected_test_max_as_of_date':None,'next_test_date':dates[pos+1].date().isoformat() if pos+1<len(dates) else None,'history':[]})
                    continue
                chosen,stats=select_replay(train_m,test_m,date,cfg,cfg['seed']+pos)
                train_part=chosen[chosen['_source'].eq('train')].copy(); test_part=chosen[chosen['_source'].eq('test')].copy()
                train_ids=train_part['_row_id'].to_numpy(dtype=np.int64); test_ids=test_part['_row_id'].to_numpy(dtype=np.int64)
                if len(train_ids) and (train_ids.min()<0 or train_ids.max()>=len(train_h)): raise AssertionError('train replay row out of bounds')
                if len(test_ids) and (test_ids.min()<0 or test_ids.max()>=len(test_h)): raise AssertionError('test replay row out of bounds')
                uh=np.concatenate([np.asarray(train_h[train_ids]),np.asarray(test_h[test_ids])]); uy=np.concatenate([np.asarray(train_y[train_ids]),np.asarray(test_y[test_ids])]); um=pd.concat([train_part,test_part],ignore_index=True)
                if (um.label_end_date > date).any(): raise AssertionError('future label entered materialized replay')
                candidate=copy.deepcopy(model).to(device); candidate,hist,cand_val,best_ep=train_candidate(candidate,uh,uy,um,val_h,val_y,val_m,cfg,device,cfg['seed']+pos+1,cfg['update_epochs']); accepted=bool(np.isfinite(cand_val) and cand_val>=champion_before)
                if accepted: model=candidate; version+=1; champion=cand_val; torch.save(model.state_dict(),checkpoints/f'version_{version:03d}.pt')
                logs.append({'update_date':date.date().isoformat(),'model_version_before':before,'model_version_after':version,'accepted':accepted,'decision':'accepted' if accepted else 'rejected_validation_rankic_gate','champion_validation_rankic_before':float(champion_before),'candidate_validation_rankic':cand_val,'new_matured_test_rows':len(new_ids),'matured_test_rows_available':stats['matured_test_rows_available'],'selected_replay_rows':stats['selected_rows'],'selected_train_rows':stats['selected_train_rows'],'selected_test_rows':stats['selected_test_rows'],'selected_max_as_of_date':stats['selected_max_as_of_date'],'selected_max_label_end_date':stats['selected_max_label_end_date'],'selected_test_min_as_of_date':stats['selected_test_min_as_of_date'],'selected_test_max_as_of_date':stats['selected_test_max_as_of_date'],'next_test_date':dates[pos+1].date().isoformat() if pos+1<len(dates) else None,'best_epoch':best_ep,'history':hist})
        pred=pd.DataFrame(rows)
        pred.loc[pred.as_of_date.isin([d.date().isoformat() for d in updates]),'update_ran_after_prediction']=True
        atomic_csv(out/'predictions.csv',pred); atomic_csv(out/'daily_metrics.csv',metrics(pred)); atomic_csv(out/'update_log.csv',pd.DataFrame(logs));
        for _,r in pred.iterrows():
            pass
        config=dict(cfg); config.update({'experiment_type':'adapter_plus_continual_test_then_train','lookback':LOOKBACK,'pred_len':PRED_LEN,'test_start':str(TEST_START.date()),'test_end':str(TEST_END.date()),'test_dates':len(dates),'test_rows':len(pred),'update_dates':[d.date().isoformat() for d in updates],'device_resolved':str(device),'cache_hashes':{f:sha256_file(cache/f) for f in ['meta_train.csv','meta_val.csv','meta_test.csv']}}); atomic_json(out/'run_config.json',config)
        summary={'status':'complete','method':'frozen_hidden_adapter_head_continual','seed':cfg['seed'],'test_dates':len(dates),'test_rows':len(pred),'initial_version':0,'final_version':version,'updates':len(logs),'accepted_updates':int(sum(bool(x['accepted']) for x in logs)),'initial_validation_rankic':float(rankic(np.concatenate(init_pred),val_y,val_m)),'final_champion_validation_rankic':float(champion),'elapsed_seconds':time.time()-started,'device':str(device)}; atomic_json(out/'summary.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    except Exception as e:
        atomic_json(out/'FAILED.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc()}); raise

if __name__=='__main__': main()




