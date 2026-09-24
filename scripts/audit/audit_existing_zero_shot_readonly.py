import csv, json, math, os, re
from pathlib import Path
from collections import Counter, defaultdict

DESKTOP = Path(__file__).resolve().parents[2]
script_hits = list(DESKTOP.rglob("run_continual_csi300_abc.ps1"))
if not script_hits:
    raise SystemExit("PROJECT_NOT_FOUND")
project = script_hits[0].parent
train = project / "train3"
adapter_dir = train / "kronos_adapter_recent10_train20_ep5_seed20260521_rerun_20260817_182753"
adapter_csv = adapter_dir / "rolling_predictions_all.csv"

def norm_stock(x):
    s = str(x or "").strip().lower().replace("_", ".")
    if s.startswith(("sh.", "sz.", "bj.")):
        p, n = s.split(".", 1)
        n = re.sub(r"\.0$", "", n)
        if n.isdigit(): n = n.zfill(6)
        return p + "." + n
    s = re.sub(r"\.0$", "", s)
    digits = re.sub(r"\D", "", s)
    if digits:
        digits = digits.zfill(6)[-6:]
        ex = "sh" if digits[0] in "569" else ("bj" if digits[0] in "48" else "sz")
        return ex + "." + digits
    return s

def choose(fields, names):
    low = {f.lower(): f for f in fields}
    for n in names:
        if n in low: return low[n]
    return None

def load_config_near(base):
    found = []
    roots = [base]
    if base.name == "by_date": roots += [base.parent, base.parent.parent]
    else: roots += [base.parent]
    seen = set()
    for r in roots:
        if not r.exists() or r in seen: continue
        seen.add(r)
        for p in sorted(r.glob("*.json")):
            if p.stat().st_size > 2_000_000: continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            vals = {}
            def walk(v):
                if isinstance(v, dict):
                    for k, x in v.items():
                        if str(k).lower() in {"lookback","pred_len","sample_count","test_samples","seed","model_name","tokenizer_name","start_date","end_date","max_windows_per_stock","step"}:
                            vals[str(k)] = x
                        walk(x)
                elif isinstance(v, list):
                    for x in v: walk(x)
            walk(obj)
            if vals: found.append({"file": str(p), "values": vals})
    return found

def summarize_csv(p, target_dates=None, target_keys=None, collect_keys=True):
    out = {"path": str(p), "size": p.stat().st_size}
    dates=set(); keys=set(); models=set(); fieldnames=[]; rows=0
    stock_counts=Counter(); signal_nonfinite=Counter(); signal_seen=Counter()
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        rd=csv.DictReader(f); fieldnames=rd.fieldnames or []
        dcol=choose(fieldnames,["as_of_date","forecast_origin","date"])
        scol=choose(fieldnames,["stock","stock_code","symbol","code","ticker"])
        mcol=choose(fieldnames,["model"])
        signal_fields=[x for x in fieldnames if x.lower() in {"pred_path_mean_return","signal_pred_mean_return","pred_endpoint_return","pred_close_mean","pred_close"}]
        out.update({"date_col":dcol,"stock_col":scol,"fields":fieldnames,"signal_fields":signal_fields})
        for row in rd:
            rows += 1
            d=(row.get(dcol) or "")[:10] if dcol else ""
            s=norm_stock(row.get(scol)) if scol else ""
            if d: dates.add(d)
            if d and s:
                keys.add((d,s)); stock_counts[d]+=1
            if mcol and len(models)<20: models.add((row.get(mcol) or "").strip())
            for sf in signal_fields:
                v=(row.get(sf) or "").strip()
                if v:
                    signal_seen[sf]+=1
                    try:
                        if not math.isfinite(float(v)): signal_nonfinite[sf]+=1
                    except Exception: signal_nonfinite[sf]+=1
    out.update({"rows":rows,"unique_dates":len(dates),"first":min(dates) if dates else None,"last":max(dates) if dates else None,"unique_keys":len(keys),"models":sorted(models),"signal_seen":dict(signal_seen),"signal_nonfinite":dict(signal_nonfinite),"unique_stocks_per_date_min":min((len({s for d2,s in keys if d2==d}) for d in dates), default=0),"unique_stocks_per_date_max":max((len({s for d2,s in keys if d2==d}) for d in dates), default=0)})
    if target_dates is not None:
        out["exact_dates"] = dates == target_dates
        out["overlap_dates"] = len(dates & target_dates)
        out["missing_dates"] = sorted(target_dates-dates)
        out["extra_dates_count"] = len(dates-target_dates)
        out["extra_dates_head"] = sorted(dates-target_dates)[:8]
    if target_keys is not None and dates == target_dates:
        out["missing_keys"] = len(target_keys-keys)
        out["extra_keys"] = len(keys-target_keys)
        out["missing_keys_head"] = sorted(target_keys-keys)[:8]
        out["extra_keys_head"] = sorted(keys-target_keys)[:8]
    out["configs"] = load_config_near(p.parent)
    return out, dates, keys

adapter, adapter_dates_all, adapter_keys_all = summarize_csv(adapter_csv)
adapter_recent50 = set(sorted(adapter_dates_all)[-50:])
# Re-read to retain only exact recent50 keys.
adapter_recent50_keys=set(); adapter_counts=Counter()
with adapter_csv.open("r", encoding="utf-8-sig", newline="") as f:
    rd=csv.DictReader(f); dcol=choose(rd.fieldnames or [],["as_of_date"]); scol=choose(rd.fieldnames or [],["stock"])
    for row in rd:
        d=(row.get(dcol) or "")[:10]
        if d in adapter_recent50:
            s=norm_stock(row.get(scol)); adapter_recent50_keys.add((d,s)); adapter_counts[d]+=1

print(json.dumps({"section":"adapter_reference","project":str(project),"adapter":adapter,"recent50_count":len(adapter_recent50),"recent50_first":min(adapter_recent50),"recent50_last":max(adapter_recent50),"recent50_dates":sorted(adapter_recent50),"recent50_unique_keys":len(adapter_recent50_keys),"recent50_unique_stocks_per_date_min":min(Counter(d for d,s in adapter_recent50_keys).values()),"recent50_unique_stocks_per_date_max":max(Counter(d for d,s in adapter_recent50_keys).values())},ensure_ascii=False))

rolling=[]
for p in sorted(train.rglob("rolling_predictions_all.csv")):
    if p.resolve()==adapter_csv.resolve(): continue
    try:
        s,_,_=summarize_csv(p,adapter_recent50,adapter_recent50_keys)
        rolling.append(s)
        print(json.dumps({"section":"rolling_candidate",**s},ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"section":"rolling_error","path":str(p),"error":repr(e)},ensure_ascii=False))

raw=[]
for p in sorted(train.rglob("*.csv")):
    n=p.name.lower()
    if p.name=="rolling_predictions_all.csv": continue
    if ("raw" in n and "signal" in n) or ("zero" in n and "shot" in n) or ("baseline" in n):
        try:
            s,_,_=summarize_csv(p,adapter_recent50,adapter_recent50_keys)
            raw.append(s)
            print(json.dumps({"section":"raw_candidate",**s},ensure_ascii=False))
        except Exception as e:
            print(json.dumps({"section":"raw_error","path":str(p),"error":repr(e)},ensure_ascii=False))

for bd in sorted([p for p in train.rglob("by_date") if p.is_dir() and p.parent.name=="predictions"]):
    files=sorted(bd.glob("*.csv")); stems={p.stem[:10] for p in files if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*",p.stem)}
    item={"section":"by_date_candidate","path":str(bd),"file_count":len(files),"date_count":len(stems),"first":min(stems) if stems else None,"last":max(stems) if stems else None,"exact_dates":stems==adapter_recent50,"overlap_dates":len(stems&adapter_recent50),"missing_dates":sorted(adapter_recent50-stems),"extra_dates_count":len(stems-adapter_recent50),"configs":load_config_near(bd)}
    if files:
        with files[0].open("r",encoding="utf-8-sig",newline="") as f:
            rd=csv.DictReader(f); item["fields"]=rd.fieldnames or []
        if stems==adapter_recent50:
            keys=set(); dup=0; rows=0; seeds=set(); models=set(); signal_bad=0
            for fp in files:
                with fp.open("r",encoding="utf-8-sig",newline="") as f:
                    rd=csv.DictReader(f); fields=rd.fieldnames or []; dcol=choose(fields,["as_of_date"]); scol=choose(fields,["stock"]); seedcol=choose(fields,["seed"]); mcol=choose(fields,["model"]); sig=choose(fields,["pred_path_mean_return","signal_pred_mean_return"])
                    for row in rd:
                        rows+=1; d=(row.get(dcol) or fp.stem)[:10]; s=norm_stock(row.get(scol)); k=(d,s)
                        if k in keys: dup+=1
                        keys.add(k)
                        if seedcol: seeds.add((row.get(seedcol) or "").strip())
                        if mcol: models.add((row.get(mcol) or "").strip())
                        if sig:
                            try:
                                if not math.isfinite(float(row.get(sig))): signal_bad+=1
                            except Exception: signal_bad+=1
            item.update({"rows":rows,"unique_keys":len(keys),"duplicate_keys":dup,"missing_keys":len(adapter_recent50_keys-keys),"extra_keys":len(keys-adapter_recent50_keys),"missing_keys_head":sorted(adapter_recent50_keys-keys)[:8],"extra_keys_head":sorted(keys-adapter_recent50_keys)[:8],"seeds":sorted(seeds),"models":sorted(models),"signal_nonfinite":signal_bad})
    print(json.dumps(item,ensure_ascii=False))

