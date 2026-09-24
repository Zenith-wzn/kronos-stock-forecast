#!/usr/bin/env python
"""Unified evaluator for the CSI800-aligned CSI300/S&P500 retest.

Consumes the existing model-run artifacts and produces traceable Path/Endpoint
point, direction, cross-sectional, ordinary portfolio, and AER outputs.
Unavailable metrics are represented explicitly with status/reason columns.
"""
from __future__ import annotations
import argparse, json, math, os, hashlib
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

H=10
SEED=100
ANNUALIZATION_DAYS=238.0
ORDINARY_COST=0.0015
AER_OPEN_COST=0.0010
AER_CLOSE_COST=0.0015
AER_MIN_FEE=5.0
AER_NOTIONAL=1_000_000.0
TOP_K=50
DROP_N=5
MIN_HOLD=5
TEST_START='2025-07-01'
TEST_END='2026-06-05'

EXPS={
 'A':'experiments/A_zero_shot',
 'B':'experiments/B_linear_head',
 'C':'experiments/C_prediction_adapter',
 'D':'experiments/D_ranking_adapter',
 'Continual A':'experiments/continual_ABC',
 'Continual B':'experiments/continual_ABC',
 'Continual C':'experiments/continual_ABC',
 'Adapter + Continual':'experiments/adapter_continual',
}

def finite(x):
    try:
        x=float(x)
        return x if math.isfinite(x) else None
    except Exception: return None

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,float)): return finite(x)
    return x

def unavailable(reason): return {'metric_status':'无法验证','unavailable_reason':reason}

def corr(a,b,rank=False):
    x=pd.to_numeric(pd.Series(a),errors='coerce'); y=pd.to_numeric(pd.Series(b),errors='coerce')
    z=pd.DataFrame({'x':x,'y':y}).dropna()
    if len(z)<3 or z.x.nunique()<2 or z.y.nunique()<2: return None
    if rank:
        return finite(z.x.rank(method='average').corr(z.y.rank(method='average')))
    return finite(z.x.corr(z.y))

def nw_t(values,lag=9):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float); n=len(x)
    if n<3: return None
    u=x-x.mean(); lrv=float(np.dot(u,u)/n)
    for k in range(1,min(lag,n-1)+1): lrv += 2*(1-k/(lag+1))*float(np.dot(u[k:],u[:-k])/n)
    se=math.sqrt(max(lrv,0)/n)
    return finite(x.mean()/se) if se>0 else None

def scalar(pred,actual):
    p=pd.to_numeric(pd.Series(pred),errors='coerce').to_numpy(float); a=pd.to_numeric(pd.Series(actual),errors='coerce').to_numpy(float)
    m=np.isfinite(p)&np.isfinite(a); p=p[m]; a=a[m]
    if len(p)==0: return unavailable('预测值或真实标签没有可用样本')|{'n':0}
    e=p-a; mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e*e))); base=float(np.mean(np.abs(a))); sse=float(np.sum(a*a))
    return {'metric_status':'可验证','unavailable_reason':'','n':int(len(p)),'mae':mae,'rmse':rmse,'mase_vs_zero_return_persistence':finite(mae/base) if base else None,'oos_r2_vs_zero_return':finite(1-np.sum(e*e)/sse) if sse else None}

def direction(pred,actual):
    p=pd.to_numeric(pd.Series(pred),errors='coerce').to_numpy(float); a=pd.to_numeric(pd.Series(actual),errors='coerce').to_numpy(float)
    m=np.isfinite(p)&np.isfinite(a); p=p[m]>0; a=a[m]>0
    if len(p)==0: return unavailable('预测值或真实标签没有可用样本')|{'n':0}
    tp=int((p&a).sum()); tn=int((~p&~a).sum()); fp=int((p&~a).sum()); fn=int((~p&a).sum()); n=tp+tn+fp+fn
    rec=tp/(tp+fn) if tp+fn else None; spec=tn/(tn+fp) if tn+fp else None; prec=tp/(tp+fp) if tp+fp else None
    f1=2*prec*rec/(prec+rec) if prec is not None and rec is not None and prec+rec else None
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return {'metric_status':'可验证','unavailable_reason':'','n':n,'accuracy':finite((tp+tn)/n),'balanced_accuracy':finite(np.nanmean([rec,spec])),'precision':finite(prec),'recall':finite(rec),'f1':finite(f1),'specificity':finite(spec),'mcc':finite((tp*tn-fp*fn)/den) if den else None,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'positive_definition':'return > 0; zero return treated as negative','missing_label_policy':'drop rows with non-finite prediction or actual'}

def canonical(df, exp, zero=False):
    df=df.copy();
    rename={}
    for a,b in [('as_of_date','prediction_date'),('stock','stock_id')]:
        if a not in df and b in df: rename[b]=a
    if rename: df=df.rename(columns=rename)
    if 'as_of_date' not in df or 'stock' not in df: raise ValueError(f'{exp}: missing date/stock columns')
    df['as_of_date']=pd.to_datetime(df['as_of_date'],errors='coerce').dt.strftime('%Y-%m-%d'); df['stock']=df['stock'].astype(str); df=df[(df['as_of_date']>=TEST_START)&(df['as_of_date']<=TEST_END)].copy()
    if zero:
        for h in range(1,H+1):
            df[f'pred_return_{h}']=pd.to_numeric(df.get(f'pred_close_{h}'),errors='coerce')/pd.to_numeric(df.get('current_close'),errors='coerce')-1
            df[f'actual_return_{h}']=pd.to_numeric(df.get(f'actual_close_{h}'),errors='coerce')/pd.to_numeric(df.get('current_close'),errors='coerce')-1
    if 'pred_signal' not in df: df['pred_signal']=pd.to_numeric(df.get('pred_path_mean_return'),errors='coerce')
    if 'actual_signal' not in df: df['actual_signal']=pd.to_numeric(df.get('actual_path_mean_return'),errors='coerce')
    if 'pred_endpoint_return' not in df and 'pred_return_10' in df: df['pred_endpoint_return']=df['pred_return_10']
    if 'actual_endpoint_return' not in df and 'actual_return_10' in df: df['actual_endpoint_return']=df['actual_return_10']
    for c in ['pred_signal','actual_signal','pred_endpoint_return','actual_endpoint_return']+[f'pred_return_{h}' for h in range(1,H+1)]+[f'actual_return_{h}' for h in range(1,H+1)]:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    df['experiment']=exp; df['dataset']=DATASET; df['seed']=SEED
    if 'model_version' not in df: df['model_version']=0
    return df.dropna(subset=['as_of_date']).drop_duplicates(['as_of_date','stock']).sort_values(['as_of_date','stock']).reset_index(drop=True)

def load_exp(root,exp):
    p=root/EXPS[exp]
    if exp=='A':
        files=sorted((p/'predictions/by_date').glob('*.csv'))
        if not files: return None, 'zero-shot prediction files not found'
        return canonical(pd.concat([pd.read_csv(f) for f in files],ignore_index=True),exp,True), ''
    if exp.startswith('Continual'):
        g=exp[-1]; f=p/f'predictions_{g}.csv'
        if not f.exists(): f=p/'predictions.csv'
        if not f.exists(): return None, f'{f} not found'
        df=pd.read_csv(f); 
        if 'group' in df: df=df[df.group.astype(str)==g]
        return canonical(df,exp,False), ''
    f=p/'predictions.csv'
    if not f.exists(): return None, f'{f} not found'
    return canonical(pd.read_csv(f),exp,False), ''

def enrich_execution(df,stocks_dir):
    df=df.copy(); need=set(zip(df.as_of_date.astype(str),df.stock.astype(str)))
    rows=[]; stocks=sorted(df.stock.astype(str).unique())
    for stock in stocks:
        candidates=[stocks_dir/f'{stock}.csv',stocks_dir/f'us_{stock}.csv',stocks_dir/f'{stock.replace('/','_')}.csv',stocks_dir/f'us_{stock.replace('/','_')}.csv']
        fp=next((x for x in candidates if x.exists()),None)
        if fp is None:
            # fallback by code column scan is intentionally not silent
            continue
        try:
            x=pd.read_csv(fp); cols={c.lower():c for c in x.columns}; date_col=cols.get('date')
            if not date_col: continue
            x=x.rename(columns={date_col:'date'}); x['date']=pd.to_datetime(x.date,errors='coerce').dt.strftime('%Y-%m-%d'); x=x.dropna(subset=['date']).sort_values('date')
            cmap={c.lower():c for c in x.columns}
            for c in ['open','close']:
                if c not in cmap: raise ValueError(f'{fp} missing {c}')
            x['next_open']=pd.to_numeric(x[cmap['open']],errors='coerce').shift(-1); x['next_close']=pd.to_numeric(x[cmap['close']],errors='coerce').shift(-1)
            x['close']=pd.to_numeric(x[cmap['close']],errors='coerce'); x['next_date']=x.date.shift(-1)
            q=x[x.date.isin({d for d,s in need if s==stock})][['date','close','next_open','next_close','next_date']].copy(); q['stock']=stock; rows.append(q)
        except Exception: continue
    if rows:
        e=pd.concat(rows,ignore_index=True).rename(columns={'date':'as_of_date'}); df=df.merge(e,on=['as_of_date','stock'],how='left',suffixes=('','_panel'))
    else:
        for c in ['close','next_open','next_close','next_date']: df[c]=np.nan
    if 'actual_return_1' not in df: df['actual_return_1']=df['next_close']/df['close']-1
    else: df['actual_return_1']=pd.to_numeric(df['actual_return_1'],errors='coerce').fillna(df['next_close']/df['close']-1)
    df['execution_return_closeclose']=df['actual_return_1']; df['execution_return_opentoclose']=df['next_close']/df['next_open']-1
    return df

def daily_cross(df,mode,pred,actual):
    out=[]
    for d,g in df.groupby('as_of_date',sort=True):
        z=g[[pred,actual]].dropna(); ic=corr(z[pred],z[actual]); ric=corr(z[pred],z[actual],True)
        out.append({'dataset':DATASET,'experiment':g.experiment.iloc[0],'path_or_endpoint':mode,'prediction_date':d,'valid_stock_count':int(len(z)),'valid_prediction_count':int(z[pred].notna().sum()),'IC':ic,'RankIC':ric,'IC_positive':int(ic is not None and ic>0),'RankIC_positive':int(ric is not None and ric>0),'exception_reason':'' if len(z)>=3 else '横截面有效样本少于3或预测/标签常数'})
    return pd.DataFrame(out)

def quantile_diff(g,signal,ret):
    z=g[[signal,ret]].dropna();
    if len(z)<2: return None
    q=max(1,len(z)//5); s=z.sort_values(signal); return finite(s.tail(q)[ret].mean()-s.head(q)[ret].mean())

def simulate_portfolio(df,signal,mode,aer=False):
    req=[signal,'execution_return_opentoclose' if aer else 'execution_return_closeclose']
    if any(c not in df for c in req): return pd.DataFrame(), 'required execution columns missing'
    retcol=req[1]; holdings=[]; age={}; rows=[]
    for d,g in df.groupby('as_of_date',sort=True):
        z=g.dropna(subset=[signal,retcol]).copy();
        if len(z)<TOP_K: # portfolio can still be formed only if at least one stock; record actual effective k
            pass
        if z.empty: continue
        z=z.sort_values([signal,'stock'],ascending=[False,True]); ranks={s:i+1 for i,s in enumerate(z.stock)}
        desired=set(z.head(min(TOP_K,len(z))).stock)
        keep={s for s in holdings if s in ranks and ranks[s]<=TOP_K+DROP_N and (age.get(s,MIN_HOLD)>=MIN_HOLD or s in desired)}
        new=list(keep)
        for s in z.stock:
            if len(new)>=min(TOP_K,len(z)): break
            if s not in new: new.append(s)
        new=set(new); sold=set(holdings)-new; bought=new-set(holdings); turnover=(len(sold)+len(bought))/max(1,len(new))
        rr=z.set_index('stock')[retcol]
        top=float(rr[rr.index.intersection(new)].mean()) if len(new) else None
        bottom=float(rr[z.tail(min(TOP_K,len(z))).stock].mean()) if len(z) else None
        bench=float(rr.mean()); q=quantile_diff(z,signal,retcol)
        cost=(len(bought)*AER_OPEN_COST+len(sold)*AER_CLOSE_COST)/max(1,len(new)) if aer else turnover*ORDINARY_COST
        if aer:
            # fixed minimum fee on normalized notional, allocated across the day's portfolio
            cost += (AER_MIN_FEE*len(bought|sold)/AER_NOTIONAL)/max(1,len(new))
        rows.append({'dataset':DATASET,'experiment':g.experiment.iloc[0],'path_or_endpoint':mode,'prediction_date':d,'valid_stock_count':int(len(z)),'holding_count':int(len(new)),'turnover':finite(turnover),'q5_minus_q1':q,'top_raw_return':top,'bottom_return':bottom,'benchmark_return':bench,'top_excess_return':finite(top-bench) if top is not None else None,'top_net_return':finite(top-cost) if top is not None else None,'long_short_return':finite(top-bottom) if top is not None and bottom is not None else None,'transaction_cost':finite(cost),'execution_mode':'signal close -> next open -> next close' if aer else 'signal close -> next close-close proxy'})
        for s in holdings: age[s]=age.get(s,0)+1
        for s in new: age[s]=age.get(s,0)
        for s in bought: age[s]=0
        holdings=list(new)
    return pd.DataFrame(rows), ''

def portfolio_summary(daily,aer=False):
    rows=[]
    if daily.empty: return rows
    for (exp,mode),g in daily.groupby(['experiment','path_or_endpoint']):
        def mean(c): return finite(pd.to_numeric(g[c],errors='coerce').mean())
        r=pd.to_numeric(g['top_net_return'],errors='coerce').dropna().to_numpy(float); cum=np.cumprod(1+r) if len(r) else np.array([])
        mdd=None
        if len(cum): mdd=finite(np.min(cum/np.maximum.accumulate(cum)-1))
        sharpe=finite(np.mean(r)/np.std(r,ddof=1)*math.sqrt(ANNUALIZATION_DAYS)) if len(r)>1 and np.std(r,ddof=1)>0 else None
        rows.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':mode,'metric_status':'可验证','unavailable_reason':'','dates':int(len(g)),'top_k':TOP_K,'drop_n':DROP_N,'min_holding_days':MIN_HOLD,'cost_rate':AER_OPEN_COST if aer else ORDINARY_COST,'open_cost':AER_OPEN_COST if aer else None,'close_cost':AER_CLOSE_COST if aer else None,'minimum_fee':AER_MIN_FEE if aer else None,'q5_minus_q1_mean':mean('q5_minus_q1'),'top_combination_raw_return_mean':mean('top_raw_return'),'bottom_combination_return_mean':mean('bottom_return'),'top_excess_return_mean':mean('top_excess_return'),'top_net_return_mean':mean('top_net_return'),'long_short_mean':mean('long_short_return'),'sharpe':sharpe,'maximum_drawdown':mdd,'turnover_mean':mean('turnover'),'net_sharpe':sharpe,'net_maximum_drawdown':mdd,'aer':finite(mean('top_excess_return')*ANNUALIZATION_DAYS) if aer else None,'aer_vs_proxy_benchmark':finite(mean('top_excess_return')*ANNUALIZATION_DAYS) if aer else None})
    return rows

def write_df(path,df):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f'.tmp.{os.getpid()}'); df.to_csv(tmp,index=False); os.replace(tmp,path)

def main():
    global DATASET
    ap=argparse.ArgumentParser(); ap.add_argument('--market-root',required=True); ap.add_argument('--stocks-dir',required=True); ap.add_argument('--dataset',required=True); ap.add_argument('--output-root',default=None)
    a=ap.parse_args(); root=Path(a.market_root); DATASET=a.dataset; out=Path(a.output_root or root); out.mkdir(parents=True,exist_ok=True); (out/'details').mkdir(exist_ok=True)
    all_frames={}; load_errors=[]
    for exp in EXPS:
        try: df,err=load_exp(root,exp)
        except Exception as e: df,err=None,repr(e)
        if df is None: load_errors.append({'dataset':DATASET,'experiment':exp,'reason':err}); continue
        df=enrich_execution(df,Path(a.stocks_dir)); all_frames[exp]=df
    point=[]; direction_rows=[]; cross_rows=[]; portfolio_rows=[]; portfolio_summ=[]; aer_rows=[]; aer_summ=[]; pred_long=[]
    for exp,df in all_frames.items():
        for mode,pred,actual in [('Path','pred_signal','actual_signal'),('Endpoint','pred_endpoint_return','actual_endpoint_return')]:
            if pred not in df or actual not in df or df[pred].notna().sum()==0 or df[actual].notna().sum()==0:
                reason='该实验输出没有独立的Endpoint轨迹' if mode=='Endpoint' else '没有可用的Path预测/标签'
                point.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':mode,**unavailable(reason),'n':0}); direction_rows.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':mode,**unavailable(reason)}); cross_rows.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':mode,**unavailable(reason)}); portfolio_summ.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':mode,**unavailable(reason)}); aer_summ.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':mode,**unavailable(reason)}); continue
            s=scalar(df[pred],df[actual]); point.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':mode,**s}); direction_rows.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':mode,**direction(df[pred],df[actual])})
            daily=daily_cross(df,mode,pred,actual); cross_rows.extend(daily.to_dict('records'))
            for h in ([None] if mode=='Path' else [None]):
                pass
            if mode=='Path':
                for h in range(1,H+1):
                    pc=f'pred_return_{h}'; ac=f'actual_return_{h}'
                    if pc in df and ac in df:
                        point.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':f'Path_h{h}','horizon':h,**scalar(df[pc],df[ac])})
                    else: point.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':f'Path_h{h}','horizon':h,**unavailable('预测文件没有保存该horizon轨迹')})
            # pooled trajectories are also emitted explicitly
            if mode=='Path':
                cols=[(f'pred_return_{h}',f'actual_return_{h}') for h in range(1,H+1)];
                if all(x in df for pair in cols for x in pair): point.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':'Path_pooled_90to10',**scalar(np.concatenate([df[x].to_numpy(float) for x,y in cols]),np.concatenate([df[y].to_numpy(float) for x,y in cols]))})
            pdaily,reason=simulate_portfolio(df,pred,'Path' if mode=='Path' else 'Endpoint',False); 
            if not pdaily.empty: portfolio_rows.extend(pdaily.to_dict('records')); portfolio_summ.extend(portfolio_summary(pdaily,False))
            else: portfolio_summ.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':mode,**unavailable(reason or '没有形成组合逐日记录')})
            adaily,reason=simulate_portfolio(df,pred,'Path' if mode=='Path' else 'Endpoint',True)
            if not adaily.empty: aer_rows.extend(adaily.to_dict('records')); aer_summ.extend(portfolio_summary(adaily,True))
            else: aer_summ.append({'dataset':DATASET,'experiment':exp,'path_or_endpoint':mode,**unavailable(reason or '缺少next_open/next_close，无法执行AER')})
        # Long trace: preserve every stored horizon when available. Signal-only continual
        # files are represented by their explicitly available aggregates only.
        for _,r in df.iterrows():
            for h in range(1,H+1):
                pc,ac=f'pred_return_{h}',f'actual_return_{h}'
                if pc in df and ac in df:
                    pred_long.append({'dataset':DATASET,'experiment':exp,'seed':SEED,'stock_id':r.stock,'prediction_date':r.as_of_date,'horizon':h,'path_or_endpoint':'Path','prediction':finite(r.get(pc)),'actual':finite(r.get(ac)),'return_definition':f'horizon-{h} cumulative return from signal-date close','model_version':r.get('model_version',0)})
            for mode,pred,actual in [('Path','pred_signal','actual_signal'),('Endpoint','pred_endpoint_return','actual_endpoint_return')]:
                if pred in df and actual in df:
                    pred_long.append({'dataset':DATASET,'experiment':exp,'seed':SEED,'stock_id':r.stock,'prediction_date':r.as_of_date,'horizon':10,'path_or_endpoint':mode,'prediction':finite(r.get(pred)),'actual':finite(r.get(actual)),'return_definition':'Path=mean of h1..h10; Endpoint=h10','model_version':r.get('model_version',0)})
    write_df(out/'point_metrics_by_horizon.csv',pd.DataFrame(point)); write_df(out/'direction_metrics.csv',pd.DataFrame(direction_rows)); write_df(out/'daily_cross_section_metrics.csv',pd.DataFrame(cross_rows)); write_df(out/'portfolio_daily_returns.csv',pd.DataFrame(portfolio_rows)); write_df(out/'portfolio_summary.csv',pd.DataFrame(portfolio_summ)); write_df(out/'aer_daily_returns.csv',pd.DataFrame(aer_rows)); write_df(out/'aer_summary.csv',pd.DataFrame(aer_summ)); write_df(out/'predictions_long.csv',pd.DataFrame(pred_long))
    # Keep update/gate provenance separate from metric files.
    updates=[]; gates=[]
    for exp in ['Continual A','Continual B','Continual C','Adapter + Continual']:
        p=root/EXPS[exp]/('update_log.csv' if exp.startswith('Continual') else 'update_log.csv')
        if p.exists():
            x=pd.read_csv(p); x.insert(0,'experiment',exp); updates.extend(x.to_dict('records')); gates.extend(x.to_dict('records'))
        else: updates.append({'experiment':exp,'status':'无法验证','reason':'update_log.csv not found'}); gates.append({'experiment':exp,'status':'无法验证','reason':'validation gate log not found'})
    write_df(out/'continual_update_log.csv',pd.DataFrame(updates)); write_df(out/'validation_gate_log.csv',pd.DataFrame(gates))
    config={'dataset':DATASET,'market_root':str(root.resolve()),'stock_pool':'current constituents backfilled; not point-in-time membership','survivor_bias_warning':True,'current_constituent_lookahead_warning':True,'train_end':'2024-12-15','val_start':'2025-01-02','val_end':'2025-06-14','test_start':TEST_START,'test_end':TEST_END,'lookback':90,'pred_len':10,'seed':SEED,'ordinary_portfolio':{'top_k':TOP_K,'drop_n':DROP_N,'min_holding_days':MIN_HOLD,'cost_rate':ORDINARY_COST,'execution':'signal close to next close-close proxy'},'aer':{'top_k':TOP_K,'drop_n':DROP_N,'min_holding_days':MIN_HOLD,'open_cost':AER_OPEN_COST,'close_cost':AER_CLOSE_COST,'minimum_fee':AER_MIN_FEE,'risk_degree':0.95,'annualization_days':ANNUALIZATION_DAYS,'benchmark':'formal test stock pool equal-weight proxy','execution':'signal close -> next open -> next close','normalized_notional_assumption':AER_NOTIONAL},'direction_definition':'return > 0 positive; zero treated negative; non-finite dropped'}
    summary={'status':'metrics_complete','dataset':DATASET,'generated_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),'config':config,'experiments_loaded':list(all_frames),'load_errors':load_errors,'test_dates':int(pd.concat(list(all_frames.values()))['as_of_date'].nunique()) if all_frames else 0,'prediction_rows':int(sum(len(x) for x in all_frames.values())),'outputs':{'point_metrics_by_horizon':'point_metrics_by_horizon.csv','direction_metrics':'direction_metrics.csv','daily_cross_section_metrics':'daily_cross_section_metrics.csv','portfolio_summary':'portfolio_summary.csv','aer_summary':'aer_summary.csv','prediction_trace':'predictions_long.csv'}}
    (out/'summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(clean(summary),ensure_ascii=False,indent=2))
if __name__=='__main__': main()



