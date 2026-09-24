"""Build an offline comparison from audited daily returns, preserving the old page."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/continual_seed100_mae_gate_audit'
report = json.loads((BASE / 'strict_comparison.json').read_text(encoding='utf-8'))
template = (ROOT / 'results/cumulative_return_curves/continual_multiseed_cumulative_return_curve.html').read_text(encoding='utf-8')
names = {'rankic_gate': 'RankIC 门控', 'mae_gate_tol001': 'MAE 容忍度 0.001', 'mae_gate_strict': '严格 MAE'}
colors = ['#64748b', '#2563eb', '#0891b2', '#a16207', '#db2777', '#15803d', '#dc2626']
series = {}
period_rows = []
for run, name in names.items():
    for group in ('A', 'B', 'C'):
        if group == 'A' and series:
            continue
        rows = list(csv.DictReader((BASE / f'{run}_{group}_daily_returns.csv').open(encoding='utf-8')))
        p = report[run]['groups'][group]['portfolio']
        wealth = 1.0
        values = [0.0]
        for row in rows:
            wealth *= 1 + float(row['net_return'])
            assert math.isclose(wealth * 1e6, float(row['portfolio_value']), rel_tol=1e-10)
            values.append(wealth - 1)
        assert len(rows) == 226
        assert math.isclose(values[-1], p['net_cumulative_return'], abs_tol=1e-10)
        series[f'{run}_{group}'] = dict(
            label='A 原始信号（三次一致）' if group == 'A' else name + (' · B 冻结头' if group == 'B' else ' · C 持续学习'),
            color=colors[len(series)], dash='dash' if group == 'B' else 'solid',
            dates=['2025-06-30'] + [r['as_of_date'] for r in rows], values=values,
            test_start=rows[0]['as_of_date'], test_end=rows[-1]['as_of_date'], n_returns=len(rows),
            final_cumulative_return=p['net_cumulative_return'], annualized_return_252=p['annualized_net_return'],
            max_drawdown=p['maximum_drawdown'], mean_turnover=p['turnover_mean'],
            total_transaction_cost=p['cumulative_transaction_cost'], trade_count=p['trade_count'])
        if run == 'mae_gate_strict' and group in ('B', 'C'):
            for lo, hi in [('2025-07-01','2025-07-31'), ('2025-08-01','2025-11-28'), ('2025-12-01','2026-06-05')]:
                net = math.prod(1 + float(r['net_return']) for r in rows if lo <= r['as_of_date'] <= hi) - 1
                period_rows.append(f'<tr><td>严格 MAE · {group}</td><td>{lo} 至 {hi}</td><td>{net:.2%}</td></tr>')
data = dict(series=series, rules=[
    'CSI300 / seed100 / 226 个交易日 / 每组 66,103 个股票—日期样本。',
    'Top30、Drop3、最短持有5日、等权多头、初始资金100万元。',
    '净收益 = 下一交易日 close-to-close 组合收益 − 单边权重换手率 × 0.15%。',
    '横轴为信号日期；收益归属该信号日至下一交易日。不把10日 Path 标签当作每日资金收益。',
    '严格 MAE 在2025-07-31和2025-11-28收盘预测后接受更新，分别从2025-08-01和2025-12-01生效。'
], notes=[
    '本次 seed100 的三种门控，C 净累计收益均低于各自 B，尚无持续学习提升测试收益的证据。单个 seed 和单段测试不能证明所有持续学习方法无效。',
    '三次运行的冻结 B 不一致：净收益分别为29.67%、22.38%、23.66%。跨运行不是纯门控单变量对照，不能把 C 的差异全部归因于门控；优先比较每次运行内部的 C 与 B。',
    '验证 MAE 从0.0308392降至0.0302406，但验证 RankIC 从0.0441543降至0.0337701。MAE改善不保证排序或交易收益改善。',
    '严格 MAE 第二阶段的相对恢复不构成第二次更新有效的因果证据，尚缺同一起点且不接受该次更新的反事实对照。',
    '采用信号收盘至下一收盘的回顾性收益代理，未证明可在信号当日收盘成交。当前成分股回填存在幸存者及成分股前视偏差，不能等同可执行实盘收益。'
])
start = template.index('const DATA=') + len('const DATA=')
end = template.index(';const chart=', start)
page = template[:start] + json.dumps(data, ensure_ascii=False, allow_nan=False) + template[end:]
page = page.replace('CSI300 Continual 多 seed 累计收益率', 'CSI300 Seed100 三种门控收益对比')
page = page.replace('沪深300（CSI300）Continual 多 seed 累计收益率', 'CSI300 · Seed100 持续学习收益对比')
page = page.replace('9条 seed×组明细与3条跨 seed 均值曲线；累计收益使用下一交易日 close-to-close 实现收益。', 'RankIC / MAE 0.001 / 严格 MAE · 2025-07-01 至 2026-06-05')
page = page.replace('Continual 多 seed 累计收益率曲线', 'Seed100 三种门控累计收益率曲线')
analysis = '<section class="card"><h2>结论：三种门控均未超过各自冻结基线</h2><div class="table-wrap"><table><thead><tr><th>门控</th><th>接受更新</th><th>B 净累计收益</th><th>C 净累计收益</th><th>C − B（百分点）</th><th>B / C Sharpe</th><th>B / C 成本（元）</th></tr></thead><tbody>'
for run, name in names.items():
    r = report[run]
    b, c = [r['groups'][g]['portfolio'] for g in ('B', 'C')]
    analysis += f'<tr><td>{name}</td><td>{r["accepted_updates"]}</td><td>{b["net_cumulative_return"]:.2%}</td><td>{c["net_cumulative_return"]:.2%}</td><td>{100*(c["net_cumulative_return"]-b["net_cumulative_return"]):+.2f}</td><td>{b["sharpe"]:.3f} / {c["sharpe"]:.3f}</td><td>{b["cumulative_transaction_cost_amount"]:,.0f} / {c["cumulative_transaction_cost_amount"]:,.0f}</td></tr>'
analysis += '</tbody></table></div><p>严格 MAE：C 收益19.26%，低于 B 的23.66%，差4.40个百分点；最大回撤分别为12.56%和12.54%。旧 RankIC 门控虽收益低于 B，但回撤较小，因此收益结论不代表所有风险指标均恶化。</p><h2>严格 MAE 分阶段净收益</h2><div class="table-wrap"><table><thead><tr><th>组合</th><th>信号日期区间</th><th>区间复利收益</th></tr></thead><tbody>' + ''.join(period_rows) + '</tbody></table></div><p>阶段收益从该阶段重新复利，持仓沿用完整回测，不在阶段边界重新建仓。2025-08-01至11-28，C落后B约7.57个百分点；12-01起相对恢复约2.92个百分点，仍未弥补前期差距。</p></section>'
page = page.replace('<div class="card"><h2>交易规则与口径</h2>', analysis + '<div class="card"><h2>交易规则与口径</h2>')
page = page.replace('</style>', '.card{border-radius:8px}.badge{white-space:normal}*{letter-spacing:0}h1{overflow-wrap:anywhere}@media(max-width:650px){.wrap{padding:12px}.head{display:block}h1{font-size:22px}#chart{height:410px}.card{padding:12px}} </style>')
page = page.replace("(e.clientX+12)+'px'", "Math.max(4,Math.min(e.clientX+12,innerWidth-tip.offsetWidth-8))+'px'")
page = page.replace("(e.clientY+12)+'px'", "Math.max(4,Math.min(e.clientY+12,innerHeight-tip.offsetHeight-8))+'px'")
out = ROOT / 'results/cumulative_return_curves/seed100_gate_comparison.html'
out.write_text(page, encoding='utf-8')
print(out)
print('Verified 7 curves, 226 returns each; daily compounding matches audited portfolio values.')
