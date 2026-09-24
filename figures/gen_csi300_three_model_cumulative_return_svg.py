from pathlib import Path
import json, math

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "cumulative_return_curves" / "csi300_three_model_cumulative_return_data.json"
OUT = ROOT / "results" / "cumulative_return_curves" / "csi300_three_model_cumulative_return_curve.svg"

data = json.loads(DATA.read_text(encoding="utf-8"))
series = data["series"]
width, height = 1200, 720
left, right, top, bottom = 105, 45, 72, 105
pw, ph = width-left-right, height-top-bottom
all_values = [v for s in series.values() for v in s["values"] if v is not None]
ymin = min(min(all_values), 0.0)
ymax = max(max(all_values), 0.0)
pad = max((ymax-ymin)*0.08, 0.01)
ymin, ymax = ymin-pad, ymax+pad
n = max(len(s["values"]) for s in series.values())

def xcoord(i): return left + (i/(n-1))*pw
def ycoord(v): return top + (ymax-v)/(ymax-ymin)*ph
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def nice_ticks(lo, hi, count=7):
    raw = (hi-lo)/count
    mag = 10 ** math.floor(math.log10(raw))
    step = min((1,2,2.5,5,10), key=lambda z: abs(z*mag-raw))*mag
    start = math.floor(lo/step)*step
    out=[]; v=start
    while v <= hi+step*0.5:
        if v >= lo-step*0.1: out.append(v)
        v += step
    return out

parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
'<rect width="100%" height="100%" fill="white"/>',
'<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#172033}.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#667085}.axis{font-size:12px;fill:#475467}.legend{font-size:14px;font-weight:600}.grid{stroke:#e5eaf2;stroke-width:1}.frame{stroke:#98a2b3;stroke-width:1;fill:none}</style>',
f'<text x="{left}" y="32" class="title">CSI300 三模型累计收益率</text>',
f'<text x="{left}" y="55" class="sub">Top30 等权 · 每日最多调出3只 · 最短持有5个交易日 · 单边权重换手成本0.15% · 2025-07-01—2026-06-05</text>']
for t in nice_ticks(ymin,ymax):
    y=ycoord(t)
    parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+pw}" y2="{y:.2f}" class="grid"/>')
    parts.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" class="axis">{t*100:.0f}%</text>')
# Date tick labels are read from the common date vector.
dates=next(iter(series.values()))["dates"]
for idx in [0, round((n-1)*.25), round((n-1)*.5), round((n-1)*.75), n-1]:
    x=xcoord(idx)
    parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+ph}" class="grid"/>')
    parts.append(f'<text x="{x:.2f}" y="{top+ph+27}" text-anchor="middle" class="axis">{esc(dates[idx])}</text>')
parts.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" class="frame"/>')
if ymin <= 0 <= ymax:
    y=ycoord(0)
    parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+pw}" y2="{y:.2f}" stroke="#667085" stroke-width="1.3"/>')
for s in series.values():
    pts=' '.join(f'{xcoord(i):.2f},{ycoord(v):.2f}' for i,v in enumerate(s["values"]) if v is not None)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
# Legend at bottom, including final return.
legend_y=height-35
slot=pw/len(series)
for i,s in enumerate(series.values()):
    x=left+i*slot
    parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x+34}" y2="{legend_y}" stroke="{s["color"]}" stroke-width="4"/>')
    label=f'{s["label"]}  {s["final_cumulative_return"]*100:.2f}%'
    parts.append(f'<text x="{x+44}" y="{legend_y+5}" class="legend">{esc(label)}</text>')
parts.append(f'<text transform="translate(26 {top+ph/2}) rotate(-90)" text-anchor="middle" class="axis">累计收益率</text>')
parts.append('</svg>')
OUT.write_text('\n'.join(parts), encoding='utf-8')
print(OUT)

