from __future__ import annotations
import hashlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.cpu_baselines.common import Panel, atomic_csv, atomic_json, require_keys, sha256
from src.cpu_baselines.runner import complete_day, compare_predictions
from src.cpu_baselines.evaluate import legacy_evaluator, detailed_evaluator


def tiny_panel():
    panel = Panel.__new__(Panel)
    panel.calendar = pd.bdate_range('2025-01-01', periods=110).strftime('%Y-%m-%d').tolist()
    panel.positions = {d:i for i,d in enumerate(panel.calendar)}
    x = np.arange(110,dtype=float)+100
    panel.frames = {'test':pd.DataFrame({'close':x,'open':x-.5,'high':x+1,'low':x-1,'volume':x*100}, index=panel.calendar)}
    past, future = panel.calendar[:90], panel.calendar[90:100]
    row = {'stock':'test','as_of_date':past[-1], 'input_start_date':past[0],'input_end_date':past[-1],
           'label_start_date':future[0],'label_end_date':future[-1],'input_dates':'|'.join(past),'label_dates':'|'.join(future),
           'input_calendar_hash':hashlib.sha256('|'.join(past).encode()).hexdigest(),
           'label_calendar_hash':hashlib.sha256('|'.join(future).encode()).hexdigest()}
    return panel,pd.DataFrame([row])


def test_exact_history_no_future():
    panel,rows=tiny_panel(); before=panel.inputs(rows)[0]
    panel.frames['test'].loc[panel.calendar[90]:,['close','open','high','low','volume']]=99999
    after=panel.inputs(rows)[0]
    np.testing.assert_array_equal(before,after)
    assert before.shape==(5,90)
    assert before[0,-1]==189


def test_window_corruption_rejected():
    panel,rows=tiny_panel(); rows.loc[0,'label_dates']=rows.loc[0,'input_dates']
    with pytest.raises(ValueError,match='Calendar mismatch'): panel.inputs(rows)


def test_price_conversion_and_quantiles():
    panel,rows=tiny_panel(); q=np.full((1,10,9),189*1.1)
    result=panel.records(rows,q,'fake','revision').iloc[0]
    assert result.pred_signal==pytest.approx(.1)
    assert result.pred_endpoint_return==pytest.approx(.1)
    assert result.actual_endpoint_return==pytest.approx(199/189-1)
    assert result['pred_q0.5_1']==pytest.approx(207.9)
    assert result.status=='ok'


def test_no_clipping_or_silent_drop():
    panel,rows=tiny_panel(); q=np.ones((1,10,9)); q[0,0,4]=-1
    result=panel.records(rows,q,'fake','revision')
    assert len(result)==1 and result.status.iloc[0]=='invalid_prediction'
    assert result.pred_close_1.iloc[0]==-1


def test_resume_hash_and_keys(tmp_path):
    _,rows=tiny_panel(); path=tmp_path/'date.csv'; frame=rows[['stock','as_of_date']].copy(); frame['status']='ok'
    assert not complete_day(path,rows,'x')
    atomic_csv(path,frame)
    assert not complete_day(path,rows,'x')
    atomic_json(path.with_suffix('.meta.json'),{'fingerprint':'x','sha256':sha256(path)})
    assert complete_day(path,rows,'x')
    with pytest.raises(ValueError,match='Unsafe resume'): complete_day(path,rows,'other')
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='Unsafe resume'): complete_day(path,rows,'x')


def test_duplicate_or_missing_keys():
    _,rows=tiny_panel()
    with pytest.raises(ValueError,match='Duplicate'): require_keys(pd.concat([rows,rows]),rows)
    with pytest.raises(ValueError,match='Key mismatch'): require_keys(rows.iloc[:0],rows)


def test_batch_comparison():
    a=np.ones((4,10,9)); assert compare_predictions(a,a.copy())==0
    with pytest.raises(ValueError,match='inconsistency'): compare_predictions(a,a+1)


def test_legacy_evaluator_regression():
    legacy=legacy_evaluator(); ranking=detailed_evaluator()
    assert legacy.scalar([0,2],[0,1])['mae']==.5
    assert legacy.direction([1,-1],[1,-1])['accuracy']==1
    x=np.arange(50,dtype=float)
    frame=pd.DataFrame({'stock':[str(i) for i in range(50)],'as_of_date':'2025-07-01',
        'pred_path_mean_return':x,'actual_path_mean_return':x,'pred_endpoint_return':x,'actual_endpoint_return':x})
    day,_=ranking.analyze_date(frame,5,50)
    assert day['spearman_rankic_path_mean']==pytest.approx(1)


def test_chronos_adapter_covariates_and_shape():
    import torch
    from src.cpu_baselines.models import Forecaster
    captured={}
    class Pipeline:
        def predict_quantiles(self,tasks,**kwargs):
            captured.update(tasks=tasks,kwargs=kwargs)
            return [torch.ones((1,10,9)) for _ in tasks],[]
    m=Forecaster.__new__(Forecaster); m.name='chronos2'; m.pipeline=Pipeline()
    x=np.arange(450,dtype=np.float32).reshape(5,90)
    assert m.predict([x],4).shape==(1,10,9)
    assert captured['kwargs']['cross_learning'] is False
    assert captured['kwargs']['batch_size']==20
    assert set(captured['tasks'][0]['past_covariates'])=={'open','high','low','volume'}
    assert 'future_covariates' not in captured['tasks'][0]
    np.testing.assert_array_equal(captured['tasks'][0]['target'],x[0])


def test_timesfm_adapter_disables_clamping():
    from src.cpu_baselines.models import Forecaster
    captured={}
    class Pipeline:
        config=SimpleNamespace(per_core_batch_size=1)
        def predict_batch(self,**kwargs):
            captured.update(kwargs)
            return [SimpleNamespace(quantiles=np.ones((10,9)))]
    m=Forecaster.__new__(Forecaster); m.name='timesfm3'; m.pipeline=Pipeline()
    x=np.ones((5,90),dtype=np.float32)
    assert m.predict([x],4).shape==(1,10,9)
    assert captured['make_positive'] is False and captured['sort_quantiles'] is False
    assert captured['past_future_covariates'] is None
    assert captured['past_only_covariates'][0].shape==(4,90)


def test_protocol_rejects_ignored_configuration():
    from src.cpu_baselines.protocol import EXPECTED, validate_config
    import copy
    cfg=copy.deepcopy(EXPECTED)
    validate_config(cfg, "chronos2")
    cfg["lookback"]=89
    with pytest.raises(ValueError, match="Unsupported protocol"):
        validate_config(cfg,"chronos2")


def test_strict_policy_keeps_young_positions_and_limits_sales():
    from src.cpu_baselines.evaluate import strict_policy_portfolio
    legacy=legacy_evaluator()
    rows=[]
    for day in range(7):
        for stock in range(100):
            rows.append(dict(as_of_date=f'2025-07-{day+1:02d}', stock=f'{stock:03d}',
                pred_signal=float(stock if day==0 else -stock), experiment='test',
                execution_return_closeclose=.01, execution_return_opentoclose=.01))
    daily, exits, summary=strict_policy_portfolio(pd.DataFrame(rows),'pred_signal','Path',False,legacy)
    assert (daily.voluntary_exits<=5).all()
    assert daily.voluntary_exits.iloc[1:5].sum()==0
    assert daily.voluntary_exits.iloc[5]==5
    assert exits.holding_days.min()>=5
    assert daily.transaction_cost.iloc[0]==pytest.approx(.0015)
