from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .common import ROOT, REFERENCE, PRICE_ROOT, atomic_json, atomic_csv, load_pairs, load_stock, set_determinism
from .models import load_sspt, corrected_pairwise_rank_loss

class CSI300SSPT:
    """Streaming CSI300 adapter matching SSPT's 25 OHLCV features and tasks."""
    def __init__(self, home: Path, config: dict, smoke: bool = False):
        self.home, self.config = home, config; self.cache = home/'cache/sspt_aligned.npz'
        self.pairs = load_pairs(); self.stocks = sorted(self.pairs.stock.unique()); self.stock_id={s:i for i,s in enumerate(self.stocks)}
        ind = pd.read_csv(home/'external/industry_membership.csv', dtype=str).fillna('')
        self.sectors = sorted(ind.industry.unique()); self.sector_id=dict(zip(self.sectors, range(len(self.sectors))))
        self.stock_sector = {r.stock:self.sector_id[r.industry] for r in ind.itertuples()}
        self.dates = pd.read_csv(REFERENCE/'master_calendar.csv',dtype=str).as_of_date.tolist(); self.date_id={d:i for i,d in enumerate(self.dates)}
        self.frames={s:load_stock(s) for s in self.stocks}; self.features = self._load_or_build(smoke)

    def _load_or_build(self, smoke: bool):
        if self.cache.exists() and not smoke:
            z=np.load(self.cache, allow_pickle=False); return z['features']
        # Feature layout follows upstream data.py per field: MA5, MA10, MA20, MA30, raw.
        stocks = self.stocks[:3] if smoke else self.stocks
        dates = self.dates[:120] if smoke else self.dates
        raw=[]
        for stock in stocks:
            f=self.frames[stock].reindex(dates); raw.append(f[['open','high','low','close','volume']].astype(float).to_numpy())
        raw=np.asarray(raw, dtype=np.float32); valid=np.isfinite(raw)
        train_end=max(self.pairs.loc[self.pairs.split.eq('train'),'as_of_date']); train_pos=min(len(dates)-1,self.dates.index(train_end))
        maxima=np.maximum(np.nanmax(np.abs(raw[:,:train_pos+1]),axis=(0,1)),1e-6)
        scaled=raw/maxima
        out=np.full((len(stocks),len(dates),25),np.nan,dtype=np.float32)
        for field in range(5):
            for offset,window in enumerate((5,10,20,30)):
                frame=pd.DataFrame(scaled[:,:,field].T); out[:,:,field*5+offset]=frame.rolling(window,min_periods=window).mean().to_numpy().T
            out[:,:,field*5+4]=scaled[:,:,field]
        out=np.nan_to_num(out,nan=0.0)
        if smoke: return out
        self.cache.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(self.cache,features=out,maxima=maxima)
        atomic_json(self.home/'cache/feature_manifest.json',{'shape':list(out.shape),'feature_order':'MA5/10/20/30 then raw OHLCV','maxima_fit_end':train_end,'maxima':maxima.tolist(),'uses_only_train_statistics':True})
        return out

    def rows(self, split):
        return self.pairs[self.pairs.split.eq(split)].copy()

    def batch(self, date, rows):
        xs=[]; ys=[]; ids=[]; sectors=[]
        for r in rows.itertuples():
            i=self.stock_id[r.stock]; t=self.date_id[r.as_of_date]; xs.append(self.features[i,t-89:t+1]);
            frame=self.frames[r.stock]; close=float(frame.loc[r.as_of_date,'close']); future=frame.loc[r.label_dates.split('|'),'close'].astype(float).to_numpy(); ys.append(float(np.mean(future/close-1))); ids.append(i); sectors.append(self.stock_sector[r.stock])
        return torch.tensor(np.asarray(xs),dtype=torch.float32), torch.tensor(ys,dtype=torch.float32), torch.tensor(ids), torch.tensor(sectors)

    def sample_fingerprint(self, row):
        import hashlib
        i=self.stock_id[row.stock]; t=self.date_id[row.as_of_date]
        return hashlib.sha256(self.features[i,t-89:t+1].tobytes()).hexdigest()

def _resolve_device(config: dict) -> torch.device:
    requested = config.get('device', 'cpu')
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'Requested {requested}, but CUDA is unavailable')
    return torch.device(requested)


def _load_pretrain_checkpoint(model: nn.Module, path: str) -> dict:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if not isinstance(checkpoint, dict):
        raise ValueError(f'Unsupported SSPT checkpoint format: {path}')
    state = {str(k).removeprefix('module.'): v for k, v in checkpoint.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded_core = [k for k in state if k.startswith(('feature_extractor.', 'attention_layers.', 'feedforward_layers.'))]
    if not loaded_core:
        raise ValueError(f'Checkpoint does not contain SSPT backbone parameters: {path}')
    return {'path': str(path), 'loaded_core_keys': len(loaded_core), 'missing_keys': missing, 'unexpected_keys': unexpected}


def run(home: Path, config: dict):
    set_determinism(config['seed'],config['threads']); data=CSI300SSPT(home,config); device=_resolve_device(config); model=load_sspt(home,config,input_size=25).to(device)
    output_home = Path(config.get('_output_dir', home)); (output_home/'weights').mkdir(parents=True, exist_ok=True); (output_home/'results').mkdir(parents=True, exist_ok=True)
    c=config['training']; tasks=c['pretrain_tasks']; model.add_outlayer('stock',len(data.stocks),device); model.add_outlayer('sector',len(data.sectors),device); model.add_outlayer('mask_avg_price',1,device)
    checkpoint_audit = None
    checkpoint_path = c.get('pretrain_checkpoint')
    if c.get('skip_pretrain') and not checkpoint_path:
        raise ValueError('--skip-pretrain requires --pretrain-checkpoint')
    if checkpoint_path:
        checkpoint_audit = _load_pretrain_checkpoint(model, checkpoint_path)
    opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate']); history=[]; started=time.perf_counter()
    train_dates=sorted(data.rows('train').as_of_date.unique())
    # Official pretraining uses stock/sector classification and masked average price. We stream one full cross-section at a time.
    pretrain_epochs = 0 if c.get('skip_pretrain') else c['pretrain_epochs']
    for epoch in range(pretrain_epochs):
        model.train(); total=0.0; count=0
        for d in train_dates:
            rows=data.rows('train'); rows=rows[rows.as_of_date.eq(d)]
            x,y,stock,sector=data.batch(d,rows); x=x.to(device); y=y.to(device); stock=stock.to(device); sector=sector.to(device); mask_tensor=torch.ones_like(x)
            mask_count=round(x.shape[1]*c['mask_rate'])
            for sample in range(x.shape[0]):
                for feature in range(x.shape[2]):
                    chosen=torch.randperm(x.shape[1])[:mask_count]; mask_tensor[sample,chosen,feature]=0
            mask=x*mask_tensor
            losses=[]; model.pretrain_task='stock'; losses.append(nn.functional.cross_entropy(model(x),stock)); model.pretrain_task='sector'; losses.append(nn.functional.cross_entropy(model(x),sector)); model.pretrain_task='mask_avg_price'; losses.append(nn.functional.mse_loss(model(mask).squeeze(-1),x[:,:,18].mean(1)))
            loss=sum(losses); opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss); count+=1
        history.append({'phase':'pretrain','epoch':epoch+1,'loss':total/max(count,1)}); atomic_json(home/'results/progress.json',{'phase':'pretrain','epoch':epoch+1,'history':history})
    model.pretrain_task=''; model.change_finetune_mode(True,freezing=c['freezing']); opt=torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=c['learning_rate']); val_dates=sorted(data.rows('val').as_of_date.unique())
    best=-float('inf'); patience=0
    for epoch in range(c['finetune_epochs']):
        model.train(); total=0.; n=0
        for d in train_dates:
            rows=data.rows('train'); rows=rows[rows.as_of_date.eq(d)]; x,y,_,_=data.batch(d,rows); x=x.to(device); y=y.to(device); pred=model(x).squeeze(-1); loss=nn.functional.mse_loss(pred,y)+c['ranking_loss_alpha']*corrected_pairwise_rank_loss(pred,y); opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss); n+=1
        model.eval(); val_predictions=[]; val_targets=[]; val_groups=[]
        with torch.no_grad():
            for d in val_dates:
                rows=data.rows('val'); rows=rows[rows.as_of_date.eq(d)]; x,y,_,_=data.batch(d,rows); x=x.to(device); pred=model(x).squeeze(-1).cpu().numpy(); val_predictions.extend(pred.tolist()); val_targets.extend(y.numpy().tolist()); val_groups.extend([d]*len(y))
        table=pd.DataFrame({'date':val_groups,'prediction':val_predictions,'target':val_targets}); vals=[g.prediction.corr(g.target,method='spearman') for _,g in table.groupby('date')]
        score=float(np.nanmean(vals)); history.append({'phase':'finetune','epoch':epoch+1,'loss':total/max(n,1),'validation_rankic':score}); atomic_json(home/'results/progress.json',{'phase':'finetune','epoch':epoch+1,'history':history})
        if score>best: best=score; patience=0; torch.save(model.state_dict(),output_home/'weights/sspt_best.pt')
        else: patience+=1
        if patience>=c['early_stopping_patience']: break
    best_path = output_home/'weights/sspt_best.pt'
    model.load_state_dict(torch.load(best_path,map_location='cpu',weights_only=True)); model.eval(); outputs=[]; test=data.rows('test')
    with torch.no_grad():
        for d in sorted(test.as_of_date.unique()):
            rows=test[test.as_of_date.eq(d)]; x,y,_,_=data.batch(d,rows); x=x.to(device); pred=model(x).squeeze(-1).cpu().numpy()
            for r,p,a in zip(rows.itertuples(),pred,y.numpy()): outputs.append({'stock':r.stock,'as_of_date':r.as_of_date,'pred_signal':float(p),'actual_signal':float(a),'label_end_date':r.label_end_date,'model':'SSPT-Eastmoney-static-sector','seed':config['seed']})
    out=pd.DataFrame(outputs); atomic_csv(output_home/'results/predictions.csv',out); summary={'status':'complete','experiment_profile':config.get('experiment_profile','unspecified'),'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,'configured_pretrain_epochs':c['pretrain_epochs'],'configured_max_finetune_epochs':c['finetune_epochs'],'completed_pretrain_epochs':pretrain_epochs,'completed_finetune_epochs':sum(item['phase']=='finetune' for item in history),'early_stopping_patience':c['early_stopping_patience'],'selection_metric':c['selection_metric'],'official_reference':c.get('official_reference'),'rows':len(out),'dates':out.as_of_date.nunique(),'best_validation_rankic':best,'seconds':time.perf_counter()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,'feature_cache':str(data.cache),'pretraining_tasks':tasks,'checkpoint_audit':checkpoint_audit,'industry_policy':'Eastmoney current snapshot statically backfilled'}; atomic_json(output_home/'results/training_summary.json',summary)

