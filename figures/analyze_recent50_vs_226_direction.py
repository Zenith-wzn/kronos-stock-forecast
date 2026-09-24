from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'train3'/'csi300_csi800_aligned_seed100'/'experiments'
OUT=ROOT/'results'/'dashboards'/'direction_accuracy_recent50_vs_226_analysis.json'
START226,END226='2025-07-01','2026-06-05'
START50,END50='2026-05-27','2026-08-05'


def acc(df):
 p=pd.to_numeric(df.pred_signal,errors='coerce').to_numpy(float); a=pd.to_numeric(df.actual_signal,errors='coerce').to_numpy(float)
 m=np.isfinite(p)&np.isfinite(a); p=p[m]; a=a[m]
 ok=(p>0)==(a>0); n=len(ok); q=float(ok.mean())
 se=math.sqrt(q*(1-q)/n)
 return {'n':n,'accuracy':q,'correct':int(ok.sum()),'incorrect':int((~ok).sum()),'normal_ci95':[q-1.96*se,q+1.96*se],
         'actual_positive_ratio':float((a>0).mean()),'pred_positive_ratio':float((p>0).mean())}

def daily(df):
 z=[]
 for dt,g in df.groupby('as_of_date',sort=True):
  r=acc(g); z.append({'date':str(dt),'n':r['n'],'accuracy':r['accuracy'],'actual_positive_ratio':r['actual_positive_ratio'],'pred_positive_ratio':r['pred_positive_ratio']})
 return pd.DataFrame(z)

def load_full():
 b=pd.read_csv(EXP/'B_linear_head'/'predictions.csv'); keys=b[['as_of_date','stock']].drop_duplicates()
 a=pd.concat([pd.read_csv(f,usecols=['as_of_date','stock','pred_path_mean_return','actual_path_mean_return']) for f in sorted((EXP/'A_zero_shot'/'predictions'/'by_date').glob('*.csv'))],ignore_index=True).merge(keys,on=['as_of_date','stock'],how='inner')
 a=a.rename(columns={'pred_path_mean_return':'pred_signal','actual_path_mean_return':'actual_signal'})
 frames={'A':a,'B':b,'C':pd.read_csv(EXP/'C_prediction_adapter'/'predictions.csv'),'D':pd.read_csv(EXP/'D_ranking_adapter'/'predictions.csv')}
 c=pd.read_csv(EXP/'continual_ABC'/'predictions.csv')
 for g in 'ABC':frames['Continual '+g]=c[c.group.eq(g)].copy()
 frames['Adapter + Continual']=pd.read_csv(EXP/'adapter_continual'/'predictions.csv')
 return frames

def main():
 frames=load_full(); out={'full_window':[START226,END226],'recent50_old_window':[START50,END50],'models':{}}
 for name,df in frames.items():
  df=df.copy();df['as_of_date']=pd.to_datetime(df.as_of_date).dt.strftime('%Y-%m-%d')
  full=df[df.as_of_date.between(START226,END226)]; overlap=df[df.as_of_date.between(START50,END50)]
  fd=daily(full); od=daily(overlap)
  # First/last halves expose temporal instability without claiming causality.
  cut=len(fd)//2
  out['models'][name]={'full226':acc(full),'overlap_with_old_recent50':acc(overlap) if len(overlap) else None,
   'overlap_dates':int(overlap.as_of_date.nunique()),'overlap_range':[overlap.as_of_date.min() if len(overlap) else None,overlap.as_of_date.max() if len(overlap) else None],
   'daily_accuracy_std_226':float(fd.accuracy.std(ddof=1)),'daily_accuracy_min_226':float(fd.accuracy.min()),'daily_accuracy_max_226':float(fd.accuracy.max()),
   'first113_equal_day_mean_accuracy':float(fd.iloc[:cut].accuracy.mean()),'last113_equal_day_mean_accuracy':float(fd.iloc[cut:].accuracy.mean())}
 OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
