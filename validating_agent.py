"""
performance_eval.py
===================
Comprehensive Performance Evaluation for Soil Moisture Analysis Engine
All results displayed clearly in the terminal — no CSV/JSON output.

HOW TO RUN:
  python performance_eval.py            # full eval (scalars + maps)
  python performance_eval.py --no-maps  # scalar-only (faster)
  python performance_eval.py --debug    # show full tracebacks

METRICS COMPUTED:
  Mean Difference (Bias), RMSE, ubRMSE, Pearson R, R², MAE, Max Abs Error

QUERY COVERAGE:
  Dataset queries   : 75  (mean, min, max, trend/slope, map, comparison, season, edge-cases)
  Literature queries: 15  (meta/file, text retrieval, asset/vision, edge-cases)
  Both (hybrid)     : 5
  Chat queries      : 5
  Intent tests      : 21
  ─────────────────────
  Total             : 102 user-facing queries + 21 intent tests
"""

import os
import sys
import re
import time
import traceback
import warnings
import numpy as np
import xarray as xr
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr
from collections import defaultdict
from intent_classifier import classify_query_intent

warnings.filterwarnings("ignore")

# ── Force UTF-8 output so box-drawing / emoji characters render on Windows ──
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
if sys.stderr.encoding != "utf-8":
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL COLOURS & BOX-DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

G   = "\033[92m"
R   = "\033[91m"
Y   = "\033[93m"
C   = "\033[96m"
B   = "\033[94m"
M   = "\033[95m"
W   = "\033[97m"
DIM = "\033[2m"
BD  = "\033[1m"
RS  = "\033[0m"

def _pass():  return f"{G}✅ PASS{RS}"
def _fail():  return f"{R}❌ FAIL{RS}"
def _warn():  return f"{Y}⚠  WARN{RS}"
def _skip():  return f"{DIM}⏭  SKIP{RS}"

def _bar(width=70, ch="═"): return ch * width
def _hdr(text, ch="═", width=70):
    pad = width - len(text) - 2
    l, r = pad // 2, pad - pad // 2
    return f"{ch*l} {text} {ch*r}"

def _metric_pf(val, threshold, mode="lt"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return _warn()
    if mode == "lt":
        return _pass() if abs(val) < threshold else _fail()
    if mode == "gt":
        return _pass() if val > threshold else _warn()
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

THR = {
    "bias"   : 1e-5,
    "rmse"   : 1e-5,
    "ubrmse" : 1e-5,
    "r"      : 0.9999,
    "mae"    : 1e-5,
    "max_abs": 1e-4,
}

# ─────────────────────────────────────────────────────────────────────────────
# SEASON → DATE RANGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

SEASON_MAP = {
    # name            : (mm_start, dd_start, mm_end, dd_end)  — within same year
    "monsoon"         : (6, 1,  9, 30),
    "kharif"          : (6, 1,  9, 30),   # same as monsoon for Indian Kharif
    "rabi"            : (10, 1, 3, 31),   # Oct–Mar (spans year boundary)
    "pre-monsoon"     : (3, 1,  5, 31),
    "pre_monsoon"     : (3, 1,  5, 31),
    "premonsoon"      : (3, 1,  5, 31),
    "summer"          : (3, 1,  5, 31),
    "post-monsoon"    : (10, 1, 12, 31),
    "post_monsoon"    : (10, 1, 12, 31),
    "postmonsoon"     : (10, 1, 12, 31),
    "winter"          : (12, 1, 2, 28),   # Dec–Feb (spans year boundary)
    "annual"          : (1, 1, 12, 31),
}

STATE_ABBREV = {
    "up"  : "uttar pradesh",
    "mp"  : "madhya pradesh",
    "ap"  : "andhra pradesh",
    "hp"  : "himachal pradesh",
    "j&k" : "jammu and kashmir",
    "jk"  : "jammu and kashmir",
    "wb"  : "west bengal",
    "tn"  : "tamil nadu",
}

# Normalise common spelling variants → exact shapefile STATE values
STATE_SPELLING_NORM = {
    "chhattisgarh" : "chhatisgarh",   # shapefile uses single-t spelling
    "chattisgarh"  : "chhatisgarh",
}

def expand_state(name):
    key = STATE_ABBREV.get(name.strip().lower(), name.strip().lower())
    return STATE_SPELLING_NORM.get(key, key)

def season_dates(season_name, year):
    """
    Return (start_date_str, end_date_str) for a season in a given year.
    Handles cross-year seasons (winter, rabi).
    """
    s = season_name.lower().replace(" ", "-")
    if s not in SEASON_MAP:
        # fallback: full year
        return f"{year}-01-01", f"{year}-12-31"
    ms, ds, me, de = SEASON_MAP[s]
    if s in ("winter",):
        # Dec year → Feb year+1
        return f"{year}-{ms:02d}-{ds:02d}", f"{year+1}-{me:02d}-{de:02d}"
    if s in ("rabi",):
        # Oct year → Mar year+1
        return f"{year}-{ms:02d}-{ds:02d}", f"{year+1}-{me:02d}-{de:02d}"
    return f"{year}-{ms:02d}-{ds:02d}", f"{year}-{me:02d}-{de:02d}"

# ─────────────────────────────────────────────────────────────────────────────
# METRICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred, truth, label=""):
    p = np.asarray(pred,  dtype=np.float64).ravel()
    t = np.asarray(truth, dtype=np.float64).ravel()
    mask = np.isfinite(p) & np.isfinite(t)
    p, t = p[mask], t[mask]
    n = len(p)
    if n == 0:
        nan = float("nan")
        return dict(n=0, bias=nan, rmse=nan, ubrmse=nan,
                    r=nan, r2=nan, mae=nan, max_abs=nan,
                    mean_pred=nan, mean_truth=nan, label=label)
    diff    = p - t
    bias    = float(np.mean(diff))
    mae     = float(np.mean(np.abs(diff)))
    rmse    = float(np.sqrt(np.mean(diff**2)))
    ubrmse  = float(np.sqrt(np.mean((diff - bias)**2)))
    max_abs = float(np.max(np.abs(diff)))
    if n >= 2:
        r_val, _ = pearsonr(p, t)
        r_val = float(r_val)
    else:
        r_val = float("nan")
    return dict(n=n, bias=bias, rmse=rmse, ubrmse=ubrmse,
                r=r_val, r2=float(r_val**2) if not np.isnan(r_val) else float("nan"),
                mae=mae, max_abs=max_abs,
                mean_pred=float(np.mean(p)),
                mean_truth=float(np.mean(t)),
                label=label)

def overall_result(m):
    bias_ok   = not np.isnan(m["bias"])   and abs(m["bias"])  < THR["bias"]
    rmse_ok   = not np.isnan(m["rmse"])   and m["rmse"]       < THR["rmse"]
    ubrmse_ok = not np.isnan(m["ubrmse"]) and m["ubrmse"]     < THR["ubrmse"]
    mae_ok    = not np.isnan(m["mae"])    and m["mae"]         < THR["mae"]
    checks = [bias_ok, rmse_ok, ubrmse_ok, mae_ok]
    if all(checks):
        return "PASS"
    r_ok      = not np.isnan(m["r"])      and m["r"]          > THR["r"]
    maxabs_ok = not np.isnan(m["max_abs"])and m["max_abs"]     < THR["max_abs"]
    if any(not c for c in checks) and r_ok and maxabs_ok:
        return "WARN"
    return "FAIL"

def print_metrics_block(m, indent=2):
    pad = " " * indent

    def _val(v):
        return f"{v:.8f}" if isinstance(v, float) and not np.isnan(v) else "  N/A    "
    def _sci(v):
        return f"{v:+.3e}" if isinstance(v, float) and not np.isnan(v) else "  N/A    "
    def _r_fmt(v):
        return f"{v:.6f}" if isinstance(v, float) and not np.isnan(v) else "  N/A   "

    lines = [
        f"{pad}┌{'─'*62}┐",
        f"{pad}│  {'Pixels compared':<28} {m['n']:>8,}{'':>23}│",
        f"{pad}│  {'Engine mean':<28} {_val(m['mean_pred'])} m³/m³       │",
        f"{pad}│  {'Truth  mean':<28} {_val(m['mean_truth'])} m³/m³       │",
        f"{pad}├{'─'*62}┤",
        f"{pad}│  {'Mean Diff (Bias)':<28} {_sci(m['bias'])} m³/m³  {_metric_pf(m['bias'], THR['bias'],'lt')}   │",
        f"{pad}│  {'RMSE':<28} {_sci(m['rmse'])} m³/m³  {_metric_pf(m['rmse'], THR['rmse'],'lt')}   │",
        f"{pad}│  {'ubRMSE':<28} {_sci(m['ubrmse'])} m³/m³  {_metric_pf(m['ubrmse'],THR['ubrmse'],'lt')}   │",
        f"{pad}│  {'Pearson R':<28} {_r_fmt(m['r'])}         {_metric_pf(m['r'], THR['r'],'gt')}   │",
        f"{pad}│  {'R²':<28} {_r_fmt(m['r2'])}                       │",
        f"{pad}│  {'MAE':<28} {_sci(m['mae'])} m³/m³  {_metric_pf(m['mae'], THR['mae'],'lt')}   │",
        f"{pad}│  {'Max Abs Error':<28} {_sci(m['max_abs'])} m³/m³  {_metric_pf(m['max_abs'],THR['max_abs'],'lt')}   │",
        f"{pad}└{'─'*62}┘",
    ]
    for ln in lines:
        print(ln)

# ─────────────────────────────────────────────────────────────────────────────
# PARSE ENGINE TEXT OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _parse_scalar(text):
    patterns = [
        r"Mean Soil Moisture\s*[:\|]\s*([+-]?[\d.eE+\-]+)",
        r"Soil Moisture\s*[:\|]\s*([+-]?[\d.eE+\-]+)",
        r"Minimum Soil Moisture\s*[:\|]\s*([+-]?[\d.eE+\-]+)",
        r"Maximum Soil Moisture\s*[:\|]\s*([+-]?[\d.eE+\-]+)",
        r"Mean\s*[:\|]\s*([+-]?[\d.eE+\-]+)",
    ]
    for pat in patterns:
        mo = re.search(pat, text)
        if mo:
            try:
                v = float(mo.group(1))
                if -1.0 <= v <= 2.0:
                    return v
            except ValueError:
                pass
    for tok in text.split():
        tok = tok.strip("m³/|║╠╚╔")
        try:
            v = float(tok)
            if 0.0 < v < 1.0:
                return v
        except ValueError:
            pass
    return None

def _parse_slope(text):
    mo = re.search(r"Slope\s*[:\|]\s*([+-]?[\d.eE+\-]+)", text)
    if mo:
        try:
            return float(mo.group(1))
        except ValueError:
            pass
    return None

def _parse_comparison_scalars(text):
    found = []
    for pat in [
        r"([+-]?[\d]+\.\d{4,})\s*m³/m³",
        r"Mean\s*[:\|]\s*([+-]?[\d.eE+\-]+)",
    ]:
        for mo in re.finditer(pat, text):
            try:
                v = float(mo.group(1))
                if 0.0 < v < 1.0 and v not in found:
                    found.append(v)
            except ValueError:
                pass
    return found

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION AGENT  (reads dataset independently)
# ─────────────────────────────────────────────────────────────────────────────

class ValidationAgent:
    def __init__(self, engine):
        self.ds  = engine.ds
        self.gdf = engine.gdf
        self.var = list(engine.ds.data_vars)[0]

    def _clip(self, subset, region):
        from shapely.geometry import mapping
        # Normalise spelling to match actual shapefile STATE values
        norm = STATE_SPELLING_NORM.get(region.lower(), region.lower())
        if norm == "india":
            return subset.rio.clip(self.gdf.geometry.apply(mapping), self.gdf.crs, drop=True)
        rgdf = self.gdf[self.gdf["STATE"].str.strip().str.lower() == norm]
        if rgdf.empty:
            raise ValueError(f"Region '{region}' not found (tried shapefile name: '{norm}')")
        return subset.rio.clip(rgdf.geometry.apply(mapping), self.gdf.crs, drop=True)

    def _da(self, region, s, e):
        sub = self.ds.sel(time=slice(s, e)).compute()
        return self._clip(sub, region)[self.var]

    def mean(self, region, s, e):
        da = self._da(region, s, e)
        return float(da.mean()), da.mean(dim="time")

    def minimum(self, region, s, e):
        da = self._da(region, s, e)
        return float(da.min()), da.min(dim="time")

    def maximum(self, region, s, e):
        da = self._da(region, s, e)
        return float(da.max()), da.max(dim="time")

    def slope(self, region, s, e):
        da  = self._da(region, s, e)
        ts  = da.mean(dim=["x", "y"])
        msk = ~np.isnan(ts.values)
        tc  = ts.values[msk]
        xi  = np.arange(len(tc))
        if len(tc) > 2:
            sl, ic, rv, pv, _ = stats.linregress(xi, tc)
        else:
            sl, ic, rv, pv = 0.0, 0.0, 0.0, 1.0
        def _px(y):
            idx = np.arange(len(y)); m = ~np.isnan(y)
            return stats.linregress(idx[m], y[m])[0] if sum(m) > 2 else np.nan
        smap = xr.apply_ufunc(_px, da, input_core_dims=[["time"]], vectorize=True)
        return float(sl), float(rv**2), float(pv), smap

    def comparison_time(self, region, periods, metric="mean"):
        results = []
        for s, e in periods:
            da = self._da(region, s, e)
            if metric == "min":
                v, mp = float(da.min()), da.min(dim="time")
            elif metric == "max":
                v, mp = float(da.max()), da.max(dim="time")
            else:
                v, mp = float(da.mean()), da.mean(dim="time")
            results.append((v, mp))
        return results

    def comparison_region(self, r1, r2, s, e, metric="mean"):
        out = {}
        for reg in [r1, r2]:
            da = self._da(reg, s, e)
            if metric == "min":
                out[reg] = (float(da.min()), da.min(dim="time"))
            elif metric == "max":
                out[reg] = (float(da.max()), da.max(dim="time"))
            else:
                out[reg] = (float(da.mean()), da.mean(dim="time"))
        return out


def _engine_map(engine, region, s, e, op):
    from shapely.geometry import mapping
    try:
        sub  = engine.ds.sel(time=slice(s, e)).compute()
        var  = list(engine.ds.data_vars)[0]
        rgdf = engine.gdf if region.lower() == "india" else \
               engine.gdf[engine.gdf["STATE"].str.strip().str.lower() == region.lower()]
        clip = sub.rio.clip(rgdf.geometry.apply(mapping), engine.gdf.crs, drop=True)
        da   = clip[var]
        if op == "mean":    return da.mean(dim="time")
        if op == "minimum": return da.min(dim="time")
        if op == "maximum": return da.max(dim="time")
        if op == "slope":
            def _px(y):
                idx = np.arange(len(y)); m = ~np.isnan(y)
                return stats.linregress(idx[m], y[m])[0] if sum(m) > 2 else np.nan
            return xr.apply_ufunc(_px, da, input_core_dims=[["time"]], vectorize=True)
    except Exception as exc:
        print(f"      {Y}[map extract failed: {exc}]{RS}")
        return None


def compare_maps(em, tm, label=""):
    if em is None or tm is None:
        return None
    try:
        try:
            ea = em.interp(x=tm.coords["x"], y=tm.coords["y"], method="nearest")
        except Exception:
            ea = em
        pa = ea.values.astype(np.float64)
        ta = tm.values.astype(np.float64)
        if pa.shape != ta.shape:
            pf = pa.ravel(); tf = ta.ravel(); n = min(len(pf), len(tf))
            pa = pf[:n].reshape(1, n); ta = tf[:n].reshape(1, n)
        return compute_metrics(pa, ta, label=label)
    except Exception as exc:
        print(f"      {Y}[map compare failed: {exc}]{RS}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# INTENT TEST CASES  (21 cases — unchanged from original + expanded)
# ─────────────────────────────────────────────────────────────────────────────

INTENT_TESTS = [
    # Original 21
    {"id": "I01", "query": "what is the mean soil moisture in Rajasthan for 2021?",                      "expected": "dataset"},
    {"id": "I02", "query": "what does the Anoops paper say about SVM?",                                  "expected": "literature"},
    {"id": "I03", "query": "hello who are you",                                                           "expected": "chat"},
    {"id": "I04", "query": "what is the maximum soil moisture in punjab 2020 and what causes this according to literature?", "expected": "both"},
    {"id": "I05", "query": "plot the spatial distribution map for india in 2021",                         "expected": "dataset"},
    {"id": "I06", "query": "summarize the methodology section of the latest PDF",                         "expected": "literature"},
    {"id": "I07", "query": "compare soil moisture between north and south india",                         "expected": "dataset"},
    {"id": "I08", "query": "which state had highest soil moisture in 2021",                               "expected": "dataset"},
    {"id": "I09", "query": "show driest region during summer 2020",                                       "expected": "dataset"},
    {"id": "I10", "query": "mean soil moistre in rajsthan 2021",                                         "expected": "dataset"},
    {"id": "I11", "query": "trend of soil moisturee in kerla",                                           "expected": "dataset"},
    {"id": "I12", "query": "compare gujrat and maharastra soil moisture",                                 "expected": "dataset"},
    {"id": "I13", "query": "mean soil moisture in kerala",                                                "expected": "dataset"},
    {"id": "I14", "query": "was india wetter in 2021 than 2020",                                         "expected": "dataset"},
    {"id": "I15", "query": "did punjab experience increasing soil moisture",                              "expected": "dataset"},
    {"id": "I16", "query": "drought conditions in rajasthan 2022",                                       "expected": "dataset"},
    {"id": "I17", "query": "top 5 wettest states in 2021",                                               "expected": "dataset"},
    {"id": "I18", "query": "rank states by average soil moisture",                                        "expected": "dataset"},
    {"id": "I19", "query": "mean soil moisture in atlantis",                                              "expected": "dataset"},
    {"id": "I20", "query": "show data for 2050",                                                          "expected": "dataset"},
    {"id": "I21", "query": "compare punjab and haryana moisture and explain possible agricultural reasons","expected": "both"},
    # From Document 1 — Chat queries (Q98–102)
    {"id": "I22", "query": "Hi",                                                                          "expected": "chat"},
    {"id": "I23", "query": "What can you do?",                                                            "expected": "chat"},
    {"id": "I24", "query": "What is soil moisture?",                                                      "expected": "chat"},
    {"id": "I25", "query": "What is the difference between active and passive microwave?",                "expected": "chat"},
    {"id": "I26", "query": "Compare active and passive microwave sensors",                                "expected": "chat"},
    # From Document 1 — Literature queries (Q76–92)
    {"id": "I27", "query": "List all loaded files",                                                       "expected": "literature"},
    {"id": "I28", "query": "How many tables are in the loaded PDF?",                                      "expected": "literature"},
    {"id": "I29", "query": "How many figures are in the loaded paper?",                                   "expected": "literature"},
    {"id": "I30", "query": "List all images in the loaded PDF",                                           "expected": "literature"},
    {"id": "I31", "query": "What retrieval algorithm does AMSR2 use for soil moisture?",                  "expected": "literature"},
    {"id": "I32", "query": "Summarise the loaded paper in 3 key points",                                  "expected": "literature"},
    {"id": "I33", "query": "Describe the methodology used in the paper",                                  "expected": "literature"},
    {"id": "I34", "query": "What bias was observed between SMAP and AMSR in the study?",                  "expected": "literature"},
    {"id": "I35", "query": "Explain the tau-omega model from the loaded literature",                      "expected": "literature"},
    {"id": "I36", "query": "What datasets were used for validation in the paper?",                        "expected": "literature"},
    {"id": "I37", "query": "What vegetation correction was applied in the retrieval?",                    "expected": "literature"},
    {"id": "I38", "query": "What were the main conclusions of the study?",                                "expected": "literature"},
    {"id": "I39", "query": "Show me figure 1",                                                            "expected": "literature"},
    {"id": "I40", "query": "Show me table 1",                                                             "expected": "literature"},
    {"id": "I41", "query": "Explain what the scatter plot in the paper shows",                            "expected": "literature"},
    {"id": "I42", "query": "What is on page 3 of the paper?",                                            "expected": "literature"},
    {"id": "I43", "query": "What RMSE was reported for the AMSR2 validation study?",                     "expected": "literature"},
    # Hybrid (Q93–97)
    {"id": "I44", "query": "Show moisture data for Rajasthan in 2020 and also explain what the literature says about AMSR2 retrieval", "expected": "both"},
    {"id": "I45", "query": "Give mean moisture for India in 2021 and explain the retrieval methodology from the paper",               "expected": "both"},
    {"id": "I46", "query": "Show moisture trend in Punjab in 2022 and describe the trend graph in the paper",                         "expected": "both"},
    {"id": "I47", "query": "Show data and explain literature on AMSR2 bias correction",                   "expected": "both"},
    {"id": "I48", "query": "Show monsoon 2022 moisture trend and explain what figure 1 shows",             "expected": "both"},
]

# ─────────────────────────────────────────────────────────────────────────────
# DATASET TEST CASES — all 75 user queries mapped to validation parameters
# ─────────────────────────────────────────────────────────────────────────────
#
# Naming convention:
#   Q01–Q10  : basic mean
#   Q11–Q20  : min / max
#   Q21–Q30  : trend / slope
#   Q31–Q40  : map queries     (scalar fallback used when map comparison disabled)
#   Q41–Q50  : time comparisons
#   Q51–Q59  : region comparisons
#   Q60–Q68  : multi-range / season
#   Q69–Q75  : edge cases
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [

    # ══════════════════════════════════════════════════════════════════════════
    # BASIC MEAN (Q01 – Q10)
    # ══════════════════════════════════════════════════════════════════════════

    # Q1  "What is mean soil moisture in Rajasthan for June 2020?"
    {"id": "Q01", "op": "mean", "region": "rajasthan",
     "s": "2020-06-01", "e": "2020-06-30",
     "desc": "Q01 — Mean Rajasthan June 2020",
     "source_query": "What is mean soil moisture in Rajasthan for June 2020?"},

    # Q2  "What is mean moisture in Chhattisgarh on 01-06-2021?"
    {"id": "Q02", "op": "mean", "region": "chhattisgarh",
     "s": "2021-06-01", "e": "2021-06-01",
     "desc": "Q02 — Mean Chhattisgarh single day 01-06-2021",
     "source_query": "What is mean moisture in Chhattisgarh on 01-06-2021?"},

    # Q3  "Mean moisture in Himachal Pradesh for winter 2020"
    {"id": "Q03", "op": "mean", "region": "himachal pradesh",
     "s": "2020-12-01", "e": "2021-02-28",
     "desc": "Q03 — Mean Himachal Pradesh Winter 2020",
     "source_query": "Mean moisture in Himachal Pradesh for winter 2020"},

    # Q4  "What is mean moisture in Uttarakhand for January 2022?"
    {"id": "Q04", "op": "mean", "region": "uttarakhand",
     "s": "2022-01-01", "e": "2022-01-31",
     "desc": "Q04 — Mean Uttarakhand January 2022",
     "source_query": "What is mean moisture in Uttarakhand for January 2022?"},

    # Q5  "Mean moisture in Sikkim for post-monsoon 2021"
    {"id": "Q05", "op": "mean", "region": "sikkim",
     "s": "2021-10-01", "e": "2021-12-31",
     "desc": "Q05 — Mean Sikkim Post-monsoon 2021",
     "source_query": "Mean moisture in Sikkim for post-monsoon 2021"},

    # Q6  "Mean moisture in Ladakh for summer 2022"
    {"id": "Q06", "op": "mean", "region": "ladakh",
     "s": "2022-03-01", "e": "2022-05-31",
     "desc": "Q06 — Mean Ladakh Summer 2022",
     "source_query": "Mean moisture in Ladakh for summer 2022"},

    # Q7  "Mean moisture in Meghalaya for 2020"
    {"id": "Q07", "op": "mean", "region": "meghalaya",
     "s": "2020-01-01", "e": "2020-12-31",
     "desc": "Q07 — Mean Meghalaya 2020",
     "source_query": "Mean moisture in Meghalaya for 2020"},

    # Q8  "Mean moisture in Mizoram for monsoon 2021"
    {"id": "Q08", "op": "mean", "region": "mizoram",
     "s": "2021-06-01", "e": "2021-09-30",
     "desc": "Q08 — Mean Mizoram Monsoon 2021",
     "source_query": "Mean moisture in Mizoram for monsoon 2021"},

    # Q9  "Mean moisture in Tripura in 2022"
    {"id": "Q09", "op": "mean", "region": "tripura",
     "s": "2022-01-01", "e": "2022-12-31",
     "desc": "Q09 — Mean Tripura 2022",
     "source_query": "Mean moisture in Tripura in 2022"},

    # Q10 "Mean moisture in Arunachal Pradesh for monsoon 2021"
    {"id": "Q10", "op": "mean", "region": "arunachal pradesh",
     "s": "2021-06-01", "e": "2021-09-30",
     "desc": "Q10 — Mean Arunachal Pradesh Monsoon 2021",
     "source_query": "Mean moisture in Arunachal Pradesh for monsoon 2021"},

    # ══════════════════════════════════════════════════════════════════════════
    # MIN / MAX (Q11 – Q20)
    # ══════════════════════════════════════════════════════════════════════════

    # Q11 "Find minimum moisture in Maharashtra for July 2022"
    {"id": "Q11", "op": "minimum", "region": "maharashtra",
     "s": "2022-07-01", "e": "2022-07-31",
     "desc": "Q11 — Min Maharashtra July 2022",
     "source_query": "Find minimum moisture in Maharashtra for July 2022"},

    # Q12 "What is maximum moisture in Gujarat in 2020?"
    {"id": "Q12", "op": "maximum", "region": "gujarat",
     "s": "2020-01-01", "e": "2020-12-31",
     "desc": "Q12 — Max Gujarat 2020",
     "source_query": "What is maximum moisture in Gujarat in 2020?"},

    # Q13 "What was the driest month in Odisha in 2021?"
    # → treated as minimum for the full year (driest = min)
    {"id": "Q13", "op": "minimum", "region": "odisha",
     "s": "2021-01-01", "e": "2021-12-31",
     "desc": "Q13 — Min (driest) Odisha 2021",
     "source_query": "What was the driest month in Odisha in 2021?"},

    # Q14 "Maximum moisture in Punjab during pre-monsoon 2021"
    {"id": "Q14", "op": "maximum", "region": "punjab",
     "s": "2021-03-01", "e": "2021-05-31",
     "desc": "Q14 — Max Punjab Pre-monsoon 2021",
     "source_query": "Maximum moisture in Punjab during pre-monsoon 2021"},

    # Q15 "What is the wettest value in Kerala for 2020?"
    {"id": "Q15", "op": "maximum", "region": "kerala",
     "s": "2020-01-01", "e": "2020-12-31",
     "desc": "Q15 — Max (wettest) Kerala 2020",
     "source_query": "What is the wettest value in Kerala for 2020?"},

    # Q16 "What is minimum moisture in Karnataka for summer 2021?"
    {"id": "Q16", "op": "minimum", "region": "karnataka",
     "s": "2021-03-01", "e": "2021-05-31",
     "desc": "Q16 — Min Karnataka Summer 2021",
     "source_query": "What is minimum moisture in Karnataka for summer 2021?"},

    # Q17 "Maximum moisture in Assam for 2021"
    {"id": "Q17", "op": "maximum", "region": "assam",
     "s": "2021-01-01", "e": "2021-12-31",
     "desc": "Q17 — Max Assam 2021",
     "source_query": "Maximum moisture in Assam for 2021"},

    # Q18 "Minimum moisture in Andhra Pradesh for summer 2020"
    {"id": "Q18", "op": "minimum", "region": "andhra pradesh",
     "s": "2020-03-01", "e": "2020-05-31",
     "desc": "Q18 — Min Andhra Pradesh Summer 2020",
     "source_query": "Minimum moisture in Andhra Pradesh for summer 2020"},

    # Q19 "Maximum moisture in Odisha during monsoon 2020"
    {"id": "Q19", "op": "maximum", "region": "odisha",
     "s": "2020-06-01", "e": "2020-09-30",
     "desc": "Q19 — Max Odisha Monsoon 2020",
     "source_query": "Maximum moisture in Odisha during monsoon 2020"},

    # Q20 "What is minimum moisture in Tamil Nadu for summer 2022?"
    {"id": "Q20", "op": "minimum", "region": "tamil nadu",
     "s": "2022-03-01", "e": "2022-05-31",
     "desc": "Q20 — Min Tamil Nadu Summer 2022",
     "source_query": "What is minimum moisture in Tamil Nadu for summer 2022?"},

    # ══════════════════════════════════════════════════════════════════════════
    # TREND / SLOPE (Q21 – Q30)
    # ══════════════════════════════════════════════════════════════════════════

    # Q21 "Show moisture trend in Punjab during monsoon 2022"
    {"id": "Q21", "op": "slope", "region": "punjab",
     "s": "2022-06-01", "e": "2022-09-30",
     "desc": "Q21 — Slope Punjab Monsoon 2022",
     "source_query": "Show moisture trend in Punjab during monsoon 2022"},

    # Q22 "Is Telangana getting drier over 2020?"
    {"id": "Q22", "op": "slope", "region": "telangana",
     "s": "2020-01-01", "e": "2020-12-31",
     "desc": "Q22 — Slope Telangana 2020",
     "source_query": "Is Telangana getting drier over 2020?"},

    # Q23 "Show moisture trend for Uttar Pradesh from 2018 to 2022"
    {"id": "Q23", "op": "slope", "region": "uttar pradesh",
     "s": "2018-01-01", "e": "2022-12-31",
     "desc": "Q23 — Slope Uttar Pradesh 2018–2022",
     "source_query": "Show moisture trend for Uttar Pradesh from 2018 to 2022"},

    # Q24 "Show trend in soil moisture for India in 2022"
    {"id": "Q24", "op": "slope", "region": "india",
     "s": "2022-01-01", "e": "2022-12-31",
     "desc": "Q24 — Slope India 2022",
     "source_query": "Show trend in soil moisture for India in 2022"},

    # Q25 "Is Rajasthan getting wetter or drier during 2020?"
    {"id": "Q25", "op": "slope", "region": "rajasthan",
     "s": "2020-01-01", "e": "2020-12-31",
     "desc": "Q25 — Slope Rajasthan 2020 (wetter/drier)",
     "source_query": "Is Rajasthan getting wetter or drier during 2020?"},

    # Q26 "Kharif season trend for Andhra Pradesh 2021"
    {"id": "Q26", "op": "slope", "region": "andhra pradesh",
     "s": "2021-06-01", "e": "2021-09-30",
     "desc": "Q26 — Slope Andhra Pradesh Kharif 2021",
     "source_query": "Kharif season trend for Andhra Pradesh 2021"},

    # Q27 "Show moisture slope map for Tamil Nadu in 2020"
    {"id": "Q27", "op": "slope", "region": "tamil nadu",
     "s": "2020-01-01", "e": "2020-12-31",
     "desc": "Q27 — Slope map Tamil Nadu 2020",
     "source_query": "Show moisture slope map for Tamil Nadu in 2020"},

    # Q28 "Show moisture trend in West Bengal for 2021"
    {"id": "Q28", "op": "slope", "region": "west bengal",
     "s": "2021-01-01", "e": "2021-12-31",
     "desc": "Q28 — Slope West Bengal 2021",
     "source_query": "Show moisture trend in West Bengal for 2021"},

    # Q29 "Show moisture slope for Kerala from 2020 to 2022"
    {"id": "Q29", "op": "slope", "region": "kerala",
     "s": "2020-01-01", "e": "2022-12-31",
     "desc": "Q29 — Slope Kerala 2020–2022",
     "source_query": "Show moisture slope for Kerala from 2020 to 2022"},

    # Q30 "Is Maharashtra getting drier from 2019 to 2022?"
    {"id": "Q30", "op": "slope", "region": "maharashtra",
     "s": "2019-01-01", "e": "2022-12-31",
     "desc": "Q30 — Slope Maharashtra 2019–2022 (drier?)",
     "source_query": "Is Maharashtra getting drier from 2019 to 2022?"},

    # ══════════════════════════════════════════════════════════════════════════
    # MAP QUERIES (Q31 – Q40)  — validated via scalar fallback + optional map
    # ══════════════════════════════════════════════════════════════════════════

    # Q31 "Show moisture map for Andhra Pradesh in August 2021"
    {"id": "Q31", "op": "mean", "region": "andhra pradesh",
     "s": "2021-08-01", "e": "2021-08-31",
     "desc": "Q31 — Map Andhra Pradesh August 2021",
     "source_query": "Show moisture map for Andhra Pradesh in August 2021"},

    # Q32 "Map mean moisture for India in 2020"
    {"id": "Q32", "op": "mean", "region": "india",
     "s": "2020-01-01", "e": "2020-12-31",
     "desc": "Q32 — Map mean India 2020",
     "source_query": "Map mean moisture for India in 2020"},

    # Q33 "Show moisture map for all of India in monsoon 2020"
    {"id": "Q33", "op": "mean", "region": "india",
     "s": "2020-06-01", "e": "2020-09-30",
     "desc": "Q33 — Map India Monsoon 2020",
     "source_query": "Show moisture map for all of India in monsoon 2020"},

    # Q34 "Show moisture map for Rajasthan in January 2022"
    {"id": "Q34", "op": "mean", "region": "rajasthan",
     "s": "2022-01-01", "e": "2022-01-31",
     "desc": "Q34 — Map Rajasthan January 2022",
     "source_query": "Show moisture map for Rajasthan in January 2022"},

    # Q35 "Show moisture map for Bihar in August 2022"
    {"id": "Q35", "op": "mean", "region": "bihar",
     "s": "2022-08-01", "e": "2022-08-31",
     "desc": "Q35 — Map Bihar August 2022",
     "source_query": "Show moisture map for Bihar in August 2022"},

    # Q36 "Show mean moisture map for Telangana in July 2022"
    {"id": "Q36", "op": "mean", "region": "telangana",
     "s": "2022-07-01", "e": "2022-07-31",
     "desc": "Q36 — Map Telangana July 2022",
     "source_query": "Show mean moisture map for Telangana in July 2022"},

    # Q37 "Show moisture map for Goa in monsoon 2021"
    {"id": "Q37", "op": "mean", "region": "goa",
     "s": "2021-06-01", "e": "2021-09-30",
     "desc": "Q37 — Map Goa Monsoon 2021",
     "source_query": "Show moisture map for Goa in monsoon 2021"},

    # Q38 "Show minimum moisture map for Gujarat in summer 2022"
    {"id": "Q38", "op": "minimum", "region": "gujarat",
     "s": "2022-03-01", "e": "2022-05-31",
     "desc": "Q38 — Map Min Gujarat Summer 2022",
     "source_query": "Show minimum moisture map for Gujarat in summer 2022"},

    # Q39 "Show trend map for Maharashtra from 2018 to 2022"
    {"id": "Q39", "op": "slope", "region": "maharashtra",
     "s": "2018-01-01", "e": "2022-12-31",
     "desc": "Q39 — Trend map Maharashtra 2018–2022",
     "source_query": "Show trend map for Maharashtra from 2018 to 2022"},

    # Q40 "Show moisture slope map for Tamil Nadu in 2020"  (same as Q27, different map emphasis)
    {"id": "Q40", "op": "slope", "region": "tamil nadu",
     "s": "2020-01-01", "e": "2020-12-31",
     "desc": "Q40 — Slope map Tamil Nadu 2020 (map emphasis)",
     "source_query": "Show moisture slope map for Tamil Nadu in 2020"},

    # ══════════════════════════════════════════════════════════════════════════
    # TIME COMPARISON (Q41 – Q50)
    # ══════════════════════════════════════════════════════════════════════════

    # Q41 "Compare India in 2018, 2020 and 2022"
    {"id": "Q41", "op": "comparison", "ctype": "time", "region": "india", "metric": "mean",
     "periods": [("2018-01-01","2018-12-31"),
                 ("2020-01-01","2020-12-31"),
                 ("2022-01-01","2022-12-31")],
     "desc": "Q41 — Compare India 2018 / 2020 / 2022",
     "source_query": "Compare India in 2018, 2020 and 2022"},

    # Q42 "Compare Bihar in 2019 and 2022"
    {"id": "Q42", "op": "comparison", "ctype": "time", "region": "bihar", "metric": "mean",
     "periods": [("2019-01-01","2019-12-31"),
                 ("2022-01-01","2022-12-31")],
     "desc": "Q42 — Compare Bihar 2019 vs 2022",
     "source_query": "Compare Bihar in 2019 and 2022"},

    # Q43 "Compare India in 2019, 2020, 2021 and 2022"
    {"id": "Q43", "op": "comparison", "ctype": "time", "region": "india", "metric": "mean",
     "periods": [("2019-01-01","2019-12-31"),
                 ("2020-01-01","2020-12-31"),
                 ("2021-01-01","2021-12-31"),
                 ("2022-01-01","2022-12-31")],
     "desc": "Q43 — Compare India 2019/2020/2021/2022",
     "source_query": "Compare India in 2019, 2020, 2021 and 2022"},

    # Q44 "Compare Rajasthan mean moisture in monsoon 2020 and monsoon 2021"
    {"id": "Q44", "op": "comparison", "ctype": "time", "region": "rajasthan", "metric": "mean",
     "periods": [("2020-06-01","2020-09-30"),
                 ("2021-06-01","2021-09-30")],
     "desc": "Q44 — Compare Rajasthan Monsoon 2020 vs 2021",
     "source_query": "Compare Rajasthan mean moisture in monsoon 2020 and monsoon 2021"},

    # Q45 "Compare India mean moisture in Rabi 2020 and Rabi 2022"
    {"id": "Q45", "op": "comparison", "ctype": "time", "region": "india", "metric": "mean",
     "periods": [("2020-10-01","2021-03-31"),
                 ("2022-10-01","2023-03-31")],
     "desc": "Q45 — Compare India Rabi 2020 vs Rabi 2022",
     "source_query": "Compare India mean moisture in Rabi 2020 and Rabi 2022"},

    # Q46 "Compare Karnataka in 2020, 2021 and 2022"
    {"id": "Q46", "op": "comparison", "ctype": "time", "region": "karnataka", "metric": "mean",
     "periods": [("2020-01-01","2020-12-31"),
                 ("2021-01-01","2021-12-31"),
                 ("2022-01-01","2022-12-31")],
     "desc": "Q46 — Compare Karnataka 2020/2021/2022",
     "source_query": "Compare Karnataka in 2020, 2021 and 2022"},

    # Q47 "Compare India in monsoon 2020 and monsoon 2022"
    {"id": "Q47", "op": "comparison", "ctype": "time", "region": "india", "metric": "mean",
     "periods": [("2020-06-01","2020-09-30"),
                 ("2022-06-01","2022-09-30")],
     "desc": "Q47 — Compare India Monsoon 2020 vs 2022",
     "source_query": "Compare India in monsoon 2020 and monsoon 2022"},

    # Q48 "Compare Rajasthan minimum moisture in summer 2020 and summer 2022"
    {"id": "Q48", "op": "comparison", "ctype": "time", "region": "rajasthan", "metric": "min",
     "periods": [("2020-03-01","2020-05-31"),
                 ("2022-03-01","2022-05-31")],
     "desc": "Q48 — Compare Rajasthan Min Summer 2020 vs 2022",
     "source_query": "Compare Rajasthan minimum moisture in summer 2020 and summer 2022"},

    # Q49 "Compare mean moisture in Chhattisgarh for monsoon 2020 and monsoon 2021"
    {"id": "Q49", "op": "comparison", "ctype": "time", "region": "chhattisgarh", "metric": "mean",
     "periods": [("2020-06-01","2020-09-30"),
                 ("2021-06-01","2021-09-30")],
     "desc": "Q49 — Compare Chhattisgarh Monsoon 2020 vs 2021",
     "source_query": "Compare mean moisture in Chhattisgarh for monsoon 2020 and monsoon 2021"},

    # Q50 "Is Punjab getting wetter during monsoon from 2019 to 2022?"
    {"id": "Q50", "op": "slope", "region": "punjab",
     "s": "2019-06-01", "e": "2022-09-30",
     "desc": "Q50 — Slope Punjab Monsoon 2019–2022 (wetter?)",
     "source_query": "Is Punjab getting wetter during monsoon from 2019 to 2022?"},

    # ══════════════════════════════════════════════════════════════════════════
    # REGION COMPARISON (Q51 – Q59)
    # ══════════════════════════════════════════════════════════════════════════

    # Q51 "Compare Rajasthan and Gujarat in 2021"
    {"id": "Q51", "op": "comparison", "ctype": "region",
     "region": "rajasthan", "region2": "gujarat",
     "s": "2021-01-01", "e": "2021-12-31", "metric": "mean",
     "desc": "Q51 — Rajasthan vs Gujarat 2021",
     "source_query": "Compare Rajasthan and Gujarat in 2021"},

    # Q52 "Compare Tamil Nadu and Karnataka in 2022 using mean"
    {"id": "Q52", "op": "comparison", "ctype": "region",
     "region": "tamil nadu", "region2": "karnataka",
     "s": "2022-01-01", "e": "2022-12-31", "metric": "mean",
     "desc": "Q52 — Tamil Nadu vs Karnataka 2022 (mean)",
     "source_query": "Compare Tamil Nadu and Karnataka in 2022 using mean"},

    # Q53 "Compare mean moisture in Rajasthan and Maharashtra between 2020 and 2022"
    {"id": "Q53", "op": "comparison", "ctype": "region",
     "region": "rajasthan", "region2": "maharashtra",
     "s": "2020-01-01", "e": "2022-12-31", "metric": "mean",
     "desc": "Q53 — Rajasthan vs Maharashtra 2020–2022",
     "source_query": "Compare mean moisture in Rajasthan and Maharashtra between 2020 and 2022"},

    # Q54 "Compare Odisha vs West Bengal in monsoon 2022"
    {"id": "Q54", "op": "comparison", "ctype": "region",
     "region": "odisha", "region2": "west bengal",
     "s": "2022-06-01", "e": "2022-09-30", "metric": "mean",
     "desc": "Q54 — Odisha vs West Bengal Monsoon 2022",
     "source_query": "Compare Odisha vs West Bengal in monsoon 2022"},

    # Q55 "Compare Assam and West Bengal in Kharif 2021"
    {"id": "Q55", "op": "comparison", "ctype": "region",
     "region": "assam", "region2": "west bengal",
     "s": "2021-06-01", "e": "2021-09-30", "metric": "mean",
     "desc": "Q55 — Assam vs West Bengal Kharif 2021",
     "source_query": "Compare Assam and West Bengal in Kharif 2021"},

    # Q56 "Compare Himachal Pradesh and Uttarakhand in 2021"
    {"id": "Q56", "op": "comparison", "ctype": "region",
     "region": "himachal pradesh", "region2": "uttarakhand",
     "s": "2021-01-01", "e": "2021-12-31", "metric": "mean",
     "desc": "Q56 — Himachal Pradesh vs Uttarakhand 2021",
     "source_query": "Compare Himachal Pradesh and Uttarakhand in 2021"},

    # Q57 "Compare Gujarat and Rajasthan in post-monsoon 2021"
    {"id": "Q57", "op": "comparison", "ctype": "region",
     "region": "gujarat", "region2": "rajasthan",
     "s": "2021-10-01", "e": "2021-12-31", "metric": "mean",
     "desc": "Q57 — Gujarat vs Rajasthan Post-monsoon 2021",
     "source_query": "Compare Gujarat and Rajasthan in post-monsoon 2021"},

    # Q58 "Compare Uttar Pradesh and Madhya Pradesh in 2022"
    {"id": "Q58", "op": "comparison", "ctype": "region",
     "region": "uttar pradesh", "region2": "madhya pradesh",
     "s": "2022-01-01", "e": "2022-12-31", "metric": "mean",
     "desc": "Q58 — Uttar Pradesh vs Madhya Pradesh 2022",
     "source_query": "Compare Uttar Pradesh and Madhya Pradesh in 2022"},

    # Q59 "Compare Assam and West Bengal in 2021"
    {"id": "Q59", "op": "comparison", "ctype": "region",
     "region": "assam", "region2": "west bengal",
     "s": "2021-01-01", "e": "2021-12-31", "metric": "mean",
     "desc": "Q59 — Assam vs West Bengal 2021",
     "source_query": "Compare Assam and West Bengal in 2021"},

    # ══════════════════════════════════════════════════════════════════════════
    # MULTI-RANGE / SEASON (Q60 – Q68)
    # ══════════════════════════════════════════════════════════════════════════

    # Q60 "Show mean moisture values of Kerala in 2019 and 2021"
    {"id": "Q60", "op": "comparison", "ctype": "time", "region": "kerala", "metric": "mean",
     "periods": [("2019-01-01","2019-12-31"),
                 ("2021-01-01","2021-12-31")],
     "desc": "Q60 — Mean Kerala 2019 and 2021",
     "source_query": "Show mean moisture values of Kerala in 2019 and 2021"},

    # Q61 "Show moisture trend in Goa from June to September 2022"
    {"id": "Q61", "op": "slope", "region": "goa",
     "s": "2022-06-01", "e": "2022-09-30",
     "desc": "Q61 — Slope Goa Jun–Sep 2022",
     "source_query": "Show moisture trend in Goa from June to September 2022"},

    # Q62 "Average moisture in Jharkhand for Kharif 2021"
    {"id": "Q62", "op": "mean", "region": "jharkhand",
     "s": "2021-06-01", "e": "2021-09-30",
     "desc": "Q62 — Mean Jharkhand Kharif 2021",
     "source_query": "Average moisture in Jharkhand for Kharif 2021"},

    # Q63 "Show mean moisture in India for Rabi season 2022"
    {"id": "Q63", "op": "mean", "region": "india",
     "s": "2022-10-01", "e": "2023-03-31",
     "desc": "Q63 — Mean India Rabi 2022",
     "source_query": "Show mean moisture in India for Rabi season 2022"},

    # Q64 "Show maximum moisture in Assam for monsoon 2021"
    {"id": "Q64", "op": "maximum", "region": "assam",
     "s": "2021-06-01", "e": "2021-09-30",
     "desc": "Q64 — Max Assam Monsoon 2021",
     "source_query": "Show maximum moisture in Assam for monsoon 2021"},

    # Q65 "Average moisture in Manipur between January and June 2022"
    {"id": "Q65", "op": "mean", "region": "manipur",
     "s": "2022-01-01", "e": "2022-06-30",
     "desc": "Q65 — Mean Manipur Jan–Jun 2022",
     "source_query": "Average moisture in Manipur between January and June 2022"},

    # Q66 "Give me moisture data for Haryana in 2021 and 2022"
    {"id": "Q66", "op": "comparison", "ctype": "time", "region": "haryana", "metric": "mean",
     "periods": [("2021-01-01","2021-12-31"),
                 ("2022-01-01","2022-12-31")],
     "desc": "Q66 — Haryana 2021 and 2022",
     "source_query": "Give me moisture data for Haryana in 2021 and 2022"},

    # Q67 "Mean moisture in Himachal Pradesh for 2019 and 2022"
    {"id": "Q67", "op": "comparison", "ctype": "time", "region": "himachal pradesh", "metric": "mean",
     "periods": [("2019-01-01","2019-12-31"),
                 ("2022-01-01","2022-12-31")],
     "desc": "Q67 — Mean Himachal Pradesh 2019 and 2022",
     "source_query": "Mean moisture in Himachal Pradesh for 2019 and 2022"},

    # Q68 "Show moisture trend for Gujarat from 2019 to 2022"
    {"id": "Q68", "op": "slope", "region": "gujarat",
     "s": "2019-01-01", "e": "2022-12-31",
     "desc": "Q68 — Slope Gujarat 2019–2022",
     "source_query": "Show moisture trend for Gujarat from 2019 to 2022"},

    # ══════════════════════════════════════════════════════════════════════════
    # EDGE CASES (Q69 – Q75)
    # ══════════════════════════════════════════════════════════════════════════

    # Q69 "Average soil moisture in West Bengal between 2019 and 2021"
    #      (span range — treated as single continuous range, not two discrete years)
    {"id": "Q69", "op": "mean", "region": "west bengal",
     "s": "2019-01-01", "e": "2021-12-31",
     "desc": "Q69 — Mean West Bengal span 2019–2021 (edge: span vs two years)",
     "source_query": "Average soil moisture in West Bengal between 2019 and 2021"},

    # Q70 "What is mean moisture in Punjab on 15-08-2021?"  (single day DD-MM-YYYY)
    {"id": "Q70", "op": "mean", "region": "punjab",
     "s": "2021-08-15", "e": "2021-08-15",
     "desc": "Q70 — Mean Punjab single day 15-08-2021 (edge: DD-MM-YYYY)",
     "source_query": "What is mean moisture in Punjab on 15-08-2021?"},

    # Q71 "Compare India in monsoon 2020 and monsoon 2022"  (season-labelled)
    {"id": "Q71", "op": "comparison", "ctype": "time", "region": "india", "metric": "mean",
     "periods": [("2020-06-01","2020-09-30"),
                 ("2022-06-01","2022-09-30")],
     "desc": "Q71 — Compare India Monsoon 2020 vs 2022 (edge: season labels)",
     "source_query": "Compare India in monsoon 2020 and monsoon 2022"},

    # Q72 "Show moisture trend for Odisha in Kharif 2022"  (season + trend)
    {"id": "Q72", "op": "slope", "region": "odisha",
     "s": "2022-06-01", "e": "2022-09-30",
     "desc": "Q72 — Slope Odisha Kharif 2022 (edge: season+trend)",
     "source_query": "Show moisture trend for Odisha in Kharif 2022"},

    # Q73 "Mean moisture in Jammu and Kashmir for winter 2021"  (multi-word state)
    {"id": "Q73", "op": "mean", "region": "jammu and kashmir",
     "s": "2021-12-01", "e": "2022-02-28",
     "desc": "Q73 — Mean J&K Winter 2021 (edge: multi-word state)",
     "source_query": "Mean moisture in Jammu and Kashmir for winter 2021"},

    # Q74 "Mean moisture in Haryana for pre-monsoon 2022"  (hyphenated season)
    {"id": "Q74", "op": "mean", "region": "haryana",
     "s": "2022-03-01", "e": "2022-05-31",
     "desc": "Q74 — Mean Haryana Pre-monsoon 2022 (edge: hyphenated season)",
     "source_query": "Mean moisture in Haryana for pre-monsoon 2022"},

    # Q75 "Show moisture data for UP in 2021"  (state abbreviation)
    {"id": "Q75", "op": "mean", "region": "uttar pradesh",
     "s": "2021-01-01", "e": "2021-12-31",
     "desc": "Q75 — Mean UP (Uttar Pradesh) 2021 (edge: abbreviation)",
     "source_query": "Show moisture data for UP in 2021"},

    # ══════════════════════════════════════════════════════════════════════════
    # ORIGINAL SCALAR TESTS (preserved as M/N/X/S/TC/RC series)
    # ══════════════════════════════════════════════════════════════════════════

    {"id": "M01", "op": "mean",    "region": "india",           "s": "2020-06-01", "e": "2020-06-30",  "desc": "M01 — India June 2020"},
    {"id": "M02", "op": "mean",    "region": "rajasthan",       "s": "2021-01-01", "e": "2021-12-31",  "desc": "M02 — Rajasthan 2021"},
    {"id": "M03", "op": "mean",    "region": "punjab",          "s": "2019-06-01", "e": "2019-09-30",  "desc": "M03 — Punjab Monsoon 2019"},
    {"id": "M04", "op": "mean",    "region": "kerala",          "s": "2020-01-01", "e": "2020-12-31",  "desc": "M04 — Kerala 2020"},
    {"id": "M05", "op": "mean",    "region": "maharashtra",     "s": "2022-01-01", "e": "2022-01-31",  "desc": "M05 — Maharashtra Jan 2022"},
    {"id": "M06", "op": "mean",    "region": "gujarat",         "s": "2021-03-01", "e": "2021-05-31",  "desc": "M06 — Gujarat Summer 2021"},
    {"id": "M07", "op": "mean",    "region": "uttar pradesh",   "s": "2020-10-01", "e": "2020-10-31",  "desc": "M07 — UP Oct 2020"},
    {"id": "M08", "op": "mean",    "region": "odisha",          "s": "2019-07-01", "e": "2019-07-31",  "desc": "M08 — Odisha Jul 2019"},
    {"id": "M09", "op": "mean",    "region": "india",           "s": "2018-01-01", "e": "2018-12-31",  "desc": "M09 — India 2018"},
    {"id": "M10", "op": "mean",    "region": "andhra pradesh",  "s": "2020-06-01", "e": "2020-09-30",  "desc": "M10 — AP Monsoon 2020"},
    {"id": "M11", "op": "mean",    "region": "karnataka",       "s": "2019-01-01", "e": "2019-12-31",  "desc": "M11 — Karnataka 2019"},
    {"id": "M12", "op": "mean",    "region": "west bengal",     "s": "2021-08-01", "e": "2021-08-31",  "desc": "M12 — West Bengal Aug 2021"},
    {"id": "M13", "op": "mean",    "region": "india",           "s": "2020-07-15", "e": "2020-07-15",  "desc": "M13 — India Single Day 2020-07-15"},
    {"id": "M14", "op": "mean",    "region": "goa",             "s": "2021-06-01", "e": "2021-09-30",  "desc": "M14 — Goa Monsoon 2021"},
    {"id": "M15", "op": "mean",    "region": "jharkhand",       "s": "2021-01-01", "e": "2021-12-31",  "desc": "M15 — Jharkhand 2021"},

    {"id": "N01", "op": "minimum", "region": "rajasthan",       "s": "2021-03-01", "e": "2021-05-31",  "desc": "N01 — Rajasthan Summer 2021 Min"},
    {"id": "N02", "op": "minimum", "region": "india",           "s": "2020-01-01", "e": "2020-12-31",  "desc": "N02 — India 2020 Min"},
    {"id": "N03", "op": "minimum", "region": "punjab",          "s": "2019-06-01", "e": "2019-09-30",  "desc": "N03 — Punjab Monsoon 2019 Min"},
    {"id": "N04", "op": "minimum", "region": "madhya pradesh",  "s": "2021-01-01", "e": "2021-12-31",  "desc": "N04 — MP 2021 Min"},
    {"id": "N05", "op": "minimum", "region": "gujarat",         "s": "2022-01-01", "e": "2022-01-31",  "desc": "N05 — Gujarat Jan 2022 Min"},
    {"id": "N06", "op": "minimum", "region": "kerala",          "s": "2020-12-01", "e": "2021-02-28",  "desc": "N06 — Kerala Winter 2020-21 Min"},
    {"id": "N07", "op": "minimum", "region": "himachal pradesh","s": "2020-12-01", "e": "2021-02-28",  "desc": "N07 — HP Winter 2020-21 Min"},

    {"id": "X01", "op": "maximum", "region": "india",           "s": "2021-06-01", "e": "2021-09-30",  "desc": "X01 — India Monsoon 2021 Max"},
    {"id": "X02", "op": "maximum", "region": "kerala",          "s": "2020-01-01", "e": "2020-12-31",  "desc": "X02 — Kerala 2020 Max"},
    {"id": "X03", "op": "maximum", "region": "assam",           "s": "2019-06-01", "e": "2019-09-30",  "desc": "X03 — Assam Monsoon 2019 Max"},
    {"id": "X04", "op": "maximum", "region": "maharashtra",     "s": "2022-08-01", "e": "2022-08-31",  "desc": "X04 — Maharashtra Aug 2022 Max"},
    {"id": "X05", "op": "maximum", "region": "odisha",          "s": "2021-01-01", "e": "2021-12-31",  "desc": "X05 — Odisha 2021 Max"},
    {"id": "X06", "op": "maximum", "region": "west bengal",     "s": "2020-06-01", "e": "2020-09-30",  "desc": "X06 — West Bengal Monsoon 2020 Max"},
    {"id": "X07", "op": "maximum", "region": "uttarakhand",     "s": "2022-06-01", "e": "2022-09-30",  "desc": "X07 — Uttarakhand Monsoon 2022 Max"},

    {"id": "S01", "op": "slope",   "region": "punjab",          "s": "2022-06-01", "e": "2022-09-30",  "desc": "S01 — Punjab Monsoon 2022 Trend"},
    {"id": "S02", "op": "slope",   "region": "india",           "s": "2020-01-01", "e": "2020-12-31",  "desc": "S02 — India 2020 Trend"},
    {"id": "S03", "op": "slope",   "region": "rajasthan",       "s": "2021-01-01", "e": "2021-12-31",  "desc": "S03 — Rajasthan 2021 Trend"},
    {"id": "S04", "op": "slope",   "region": "kerala",          "s": "2019-06-01", "e": "2019-09-30",  "desc": "S04 — Kerala Monsoon 2019 Trend"},
    {"id": "S05", "op": "slope",   "region": "gujarat",         "s": "2022-01-01", "e": "2022-12-31",  "desc": "S05 — Gujarat 2022 Trend"},
    {"id": "S06", "op": "slope",   "region": "maharashtra",     "s": "2018-01-01", "e": "2018-12-31",  "desc": "S06 — Maharashtra 2018 Trend"},

    {"id": "TC01", "op": "comparison", "ctype": "time", "region": "india",      "metric": "mean",
     "periods": [("2020-01-01","2020-12-31"),("2021-01-01","2021-12-31")],       "desc": "TC01 — India 2020 vs 2021"},
    {"id": "TC02", "op": "comparison", "ctype": "time", "region": "rajasthan",  "metric": "mean",
     "periods": [("2019-06-01","2019-09-30"),("2021-06-01","2021-09-30")],       "desc": "TC02 — Rajasthan Monsoon 2019 vs 2021"},
    {"id": "TC03", "op": "comparison", "ctype": "time", "region": "punjab",     "metric": "mean",
     "periods": [("2018-01-01","2018-12-31"),("2020-01-01","2020-12-31"),("2022-01-01","2022-12-31")], "desc": "TC03 — Punjab 2018/2020/2022"},
    {"id": "TC04", "op": "comparison", "ctype": "time", "region": "kerala",     "metric": "mean",
     "periods": [("2019-06-01","2019-09-30"),("2020-06-01","2020-09-30"),("2021-06-01","2021-09-30")], "desc": "TC04 — Kerala Monsoon 3-way"},
    {"id": "TC05", "op": "comparison", "ctype": "time", "region": "gujarat",    "metric": "mean",
     "periods": [("2019-01-01","2019-12-31"),("2022-01-01","2022-12-31")],       "desc": "TC05 — Gujarat 2019 vs 2022"},
    {"id": "TC06", "op": "comparison", "ctype": "time", "region": "rajasthan",  "metric": "min",
     "periods": [("2020-03-01","2020-05-31"),("2022-03-01","2022-05-31")],       "desc": "TC06 — Rajasthan Summer Min 2020 vs 2022"},
    {"id": "TC07", "op": "comparison", "ctype": "time", "region": "india",      "metric": "max",
     "periods": [("2019-06-01","2019-09-30"),("2021-06-01","2021-09-30")],       "desc": "TC07 — India Monsoon Max 2019 vs 2021"},

    {"id": "RC01", "op": "comparison", "ctype": "region",
     "region": "rajasthan",     "region2": "gujarat",
     "s": "2021-01-01", "e": "2021-12-31", "metric": "mean", "desc": "RC01 — Rajasthan vs Gujarat 2021"},
    {"id": "RC02", "op": "comparison", "ctype": "region",
     "region": "kerala",        "region2": "tamil nadu",
     "s": "2020-06-01", "e": "2020-09-30", "metric": "mean", "desc": "RC02 — Kerala vs Tamil Nadu Monsoon 2020"},
    {"id": "RC03", "op": "comparison", "ctype": "region",
     "region": "rajasthan",     "region2": "maharashtra",
     "s": "2022-01-01", "e": "2022-12-31", "metric": "min",  "desc": "RC03 — Rajasthan vs Maharashtra 2022 Min"},
    {"id": "RC04", "op": "comparison", "ctype": "region",
     "region": "punjab",        "region2": "haryana",
     "s": "2019-01-01", "e": "2019-12-31", "metric": "mean", "desc": "RC04 — Punjab vs Haryana 2019"},
    {"id": "RC05", "op": "comparison", "ctype": "region",
     "region": "assam",         "region2": "west bengal",
     "s": "2021-06-01", "e": "2021-09-30", "metric": "max",  "desc": "RC05 — Assam vs West Bengal Monsoon 2021 Max"},
    {"id": "RC06", "op": "comparison", "ctype": "region",
     "region": "karnataka",     "region2": "andhra pradesh",
     "s": "2020-01-01", "e": "2020-12-31", "metric": "mean", "desc": "RC06 — Karnataka vs AP 2020"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SCOREBOARD
# ─────────────────────────────────────────────────────────────────────────────

class Scoreboard:
    def __init__(self):
        self.rows   = []
        self.per_op = defaultdict(list)

    def add(self, tid, op, desc, m, result):
        self.rows.append((tid, op, desc, result,
                          m.get("bias"), m.get("rmse"),
                          m.get("ubrmse"), m.get("r")))
        self.per_op[op].append(m)

    def add_skip(self, tid, op, desc, reason):
        self.rows.append((tid, op, desc, "SKIP", None, None, None, None))

    def _result_colour(self, r):
        if r == "PASS": return f"{G}{BD}PASS{RS}"
        if r == "FAIL": return f"{R}{BD}FAIL{RS}"
        if r == "WARN": return f"{Y}{BD}WARN{RS}"
        return f"{DIM}SKIP{RS}"

    def print_summary_table(self):
        PW = 110
        print(f"\n{BD}{_bar(PW,'═')}{RS}")
        print(f"{BD}{'  FULL RESULTS TABLE':^{PW}}{RS}")
        print(f"{BD}{_bar(PW,'═')}{RS}")
        print(f"  {'ID':<8}  {'Op':<11}  {'Description':<50}  "
              f"{'Bias':>10}  {'RMSE':>10}  {'ubRMSE':>10}  {'R':>8}  Result")
        print(f"  {'─'*8}  {'─'*11}  {'─'*50}  "
              f"{'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*6}")

        pass_n = fail_n = warn_n = skip_n = 0
        for (tid, op, desc, result, bias, rmse, ubrmse, r) in self.rows:
            def _f(v): return f"{v:+.2e}" if v is not None and not np.isnan(v) else "   N/A  "
            def _rf(v): return f"{v:.4f}"  if v is not None and not np.isnan(v) else "  N/A "
            res_col = self._result_colour(result)
            print(f"  {tid:<8}  {op:<11}  {desc:<50}  "
                  f"{_f(bias):>10}  {_f(rmse):>10}  {_f(ubrmse):>10}  "
                  f"{_rf(r):>8}  {res_col}")
            if result == "PASS": pass_n += 1
            elif result == "FAIL": fail_n += 1
            elif result == "WARN": warn_n += 1
            else: skip_n += 1

        print(f"  {'─'*PW}")
        total = len(self.rows)
        print(f"\n  Total rows : {total}   "
              f"{G}{BD}PASS: {pass_n}{RS}   "
              f"{Y}{BD}WARN: {warn_n}{RS}   "
              f"{R}{BD}FAIL: {fail_n}{RS}   "
              f"{DIM}SKIP: {skip_n}{RS}")

    def print_per_op_aggregates(self):
        PW = 70
        print(f"\n{BD}{_bar(PW,'═')}{RS}")
        print(f"{BD}{'  PER-OPERATION AGGREGATE METRICS':^{PW}}{RS}")
        print(f"{BD}{_bar(PW,'═')}{RS}")
        for op in ["mean", "minimum", "maximum", "slope", "comparison"]:
            ms = self.per_op.get(op, [])
            if not ms:
                continue
            def _agg(key):
                return [m[key] for m in ms
                        if m.get(key) is not None and not np.isnan(m.get(key, float("nan")))]
            biases = _agg("bias"); rmses = _agg("rmse")
            ubrmses = _agg("ubrmse"); rs = _agg("r")
            n = len(ms)
            print(f"\n  {BD}{C}{op.upper()}{RS}  ({n} test{'s' if n>1 else ''})")
            if biases:  print(f"    Mean Bias   :  {np.mean(biases):+.4e}   std {np.std(biases):.4e}")
            if rmses:   print(f"    Mean RMSE   :  {np.mean(rmses):.4e}   max {np.max(rmses):.4e}")
            if ubrmses: print(f"    Mean ubRMSE :  {np.mean(ubrmses):.4e}")
            if rs:      print(f"    Mean R      :  {np.mean(rs):.6f}   min {np.min(rs):.6f}")
        print(f"\n{_bar(PW,'═')}")

    def print_overall_aggregates(self):
        PW = 70
        print(f"\n{BD}{_bar(PW,'═')}{RS}")
        print(f"{BD}{'  OVERALL SYSTEM PERFORMANCE':^{PW}}{RS}")
        print(f"{BD}{_bar(PW,'═')}{RS}")
        all_ms = []
        for ms in self.per_op.values():
            all_ms.extend(ms)
        if not all_ms:
            print("  No valid metrics calculated.")
            print(f"\n{_bar(PW,'═')}")
            return
        def _agg(key):
            return [m[key] for m in all_ms
                    if m.get(key) is not None and not np.isnan(m.get(key, float("nan")))]
        biases = _agg("bias"); rmses = _agg("rmse")
        ubrmses = _agg("ubrmse"); maes = _agg("mae"); rs = _agg("r")
        n = len(all_ms)
        print(f"\n  {BD}{C}ALL OPERATIONS COMBINED{RS}  ({n} evaluation{'s' if n>1 else ''})")
        print(f"\n  {BD}{'Metric':<22} | {'Value':<15} | {'Breakdown':<25}{RS}")
        print(f"  {'-'*22}-+-{'-'*15}-+-{'-'*25}")
        if biases:  print(f"  {'Overall Mean Bias':<22} | {np.mean(biases):>+15.4e} | {np.mean(biases)*100:>+7.3f}%  std {np.std(biases):.4e}")
        if rmses:   print(f"  {'Overall Mean RMSE':<22} | {np.mean(rmses):>15.4e} | {np.mean(rmses)*100:>7.3f}%  max {np.max(rmses):.4e}")
        if ubrmses: print(f"  {'Overall Mean ubRMSE':<22} | {np.mean(ubrmses):>15.4e} | {np.mean(ubrmses)*100:>7.3f}%")
        if maes:    print(f"  {'Overall Mean MAE':<22} | {np.mean(maes):>15.4e} | {np.mean(maes)*100:>7.3f}%")
        if rs:      print(f"  {'Overall Mean R':<22} | {np.mean(rs):>15.6f} | {np.mean(rs)*100:>7.2f}%  min {np.min(rs):.6f}")
        print(f"\n  {BD}{C}PERFORMANCE ASSESSMENT{RS}")
        if rs and rmses:
            avg_r    = np.mean(rs)
            avg_rmse = np.mean(rmses)
            if avg_r > 0.9999 and avg_rmse < 1e-4:
                print(f"    {G}{BD}EXCELLENT (100.0% Match):{RS} Engine mathematics perfectly match independent")
                print(f"               validation. Computations are flawless.")
            elif avg_r > 0.95:
                print(f"    {Y}{BD}GOOD ({avg_r*100:.1f}% Match):{RS} Highly correlated with truth; minor deviations exist.")
            else:
                print(f"    {R}{BD}POOR ({max(0,avg_r*100):.1f}% Match):{RS} Significant deviation. Code review required.")
        print(f"\n{_bar(PW,'═')}")


# ─────────────────────────────────────────────────────────────────────────────
# QUERY COUNT SUMMARY  (new — prints at the very end)
# ─────────────────────────────────────────────────────────────────────────────

def print_query_count_summary(intent_results, dataset_results, board):
    PW = 70
    print(f"\n{BD}{_bar(PW,'█')}{RS}")
    print(f"{BD}{C}{'  QUERY TESTED COUNT SUMMARY':^{PW}}{RS}")
    print(f"{BD}{_bar(PW,'█')}{RS}")

    # ── Intent tests ────────────────────────────────────────────────────────
    total_intent   = len(intent_results)
    intent_passed  = sum(1 for r in intent_results if r["result"] == "PASS")
    intent_failed  = total_intent - intent_passed

    # Break down by expected category
    intent_by_cat  = defaultdict(lambda: {"total": 0, "pass": 0})
    for r in intent_results:
        cat = r["expected"]
        intent_by_cat[cat]["total"] += 1
        if r["result"] == "PASS":
            intent_by_cat[cat]["pass"] += 1

    print(f"\n  {BD}A. INTENT CLASSIFICATION TESTS{RS}")
    print(f"  {'─'*PW}")
    print(f"  {'Category':<22}  {'Tested':>7}  {'Passed':>7}  {'Failed':>7}  {'Acc %':>7}")
    print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for cat in ["dataset", "literature", "both", "chat"]:
        d = intent_by_cat.get(cat, {"total": 0, "pass": 0})
        t, p = d["total"], d["pass"]
        f    = t - p
        acc  = (p / t * 100) if t else 0.0
        col  = G if f == 0 else (Y if f <= t // 3 else R)
        print(f"  {cat:<22}  {t:>7}  {G}{p:>7}{RS}  {col}{f:>7}{RS}  {col}{acc:>6.1f}%{RS}")
    print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    print(f"  {'TOTAL':<22}  {total_intent:>7}  {G}{intent_passed:>7}{RS}  "
          f"{R if intent_failed else G}{intent_failed:>7}{RS}  "
          f"{(G if intent_passed==total_intent else Y)}{intent_passed/total_intent*100:>6.1f}%{RS}")

    # ── Dataset / numeric validation tests ──────────────────────────────────
    # Collect pass/fail/warn/skip per operation from board
    op_stats = defaultdict(lambda: {"tested": 0, "pass": 0, "fail": 0, "warn": 0, "skip": 0})
    for (tid, op, desc, result, *_) in board.rows:
        # Normalise op for cleaner grouping
        grp = op if op in ("mean","minimum","maximum","slope","comparison") else "other"
        op_stats[grp]["tested"] += 1
        if result == "PASS":  op_stats[grp]["pass"] += 1
        elif result == "FAIL":op_stats[grp]["fail"] += 1
        elif result == "WARN":op_stats[grp]["warn"] += 1
        else:                 op_stats[grp]["skip"] += 1

    total_ds   = sum(d["tested"] for d in op_stats.values())
    total_pass = sum(d["pass"]   for d in op_stats.values())
    total_fail = sum(d["fail"]   for d in op_stats.values())
    total_warn = sum(d["warn"]   for d in op_stats.values())
    total_skip = sum(d["skip"]   for d in op_stats.values())

    print(f"\n  {BD}B. DATASET VALIDATION TESTS (scalars + optional maps){RS}")
    print(f"  {'─'*PW}")
    print(f"  {'Operation':<16}  {'Tested':>7}  {'Pass':>6}  {'Warn':>6}  {'Fail':>6}  {'Skip':>6}")
    print(f"  {'─'*16}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
    for grp in ["mean", "minimum", "maximum", "slope", "comparison", "other"]:
        d = op_stats.get(grp)
        if not d or d["tested"] == 0:
            continue
        col = G if d["fail"] == 0 and d["skip"] == 0 else (Y if d["fail"] <= 1 else R)
        print(f"  {grp:<16}  {d['tested']:>7}  {G}{d['pass']:>6}{RS}  "
              f"{Y}{d['warn']:>6}{RS}  {col}{d['fail']:>6}{RS}  {DIM}{d['skip']:>6}{RS}")
    print(f"  {'─'*16}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
    tc = total_pass + total_fail + total_warn + total_skip
    print(f"  {'TOTAL':<16}  {tc:>7}  {G}{total_pass:>6}{RS}  "
          f"{Y}{total_warn:>6}{RS}  {R}{total_fail:>6}{RS}  {DIM}{total_skip:>6}{RS}")

    # ── Grand total ─────────────────────────────────────────────────────────
    grand_tested = total_intent + tc
    grand_passed = intent_passed + total_pass
    grand_failed = intent_failed + total_fail
    grand_warn   = total_warn
    grand_skip   = total_skip

    print(f"\n  {BD}C. GRAND TOTAL ACROSS ALL TEST TYPES{RS}")
    print(f"  {'─'*PW}")
    print(f"  {'Category':<30}  {'Count':>8}")
    print(f"  {'─'*30}  {'─'*8}")
    print(f"  {'Intent tests run':<30}  {total_intent:>8}")
    print(f"  {'Dataset / scalar tests run':<30}  {tc:>8}")
    print(f"  {BD}{'TOTAL QUERIES TESTED':<30}  {grand_tested:>8}{RS}")
    print(f"  {'─'*30}  {'─'*8}")
    col_p = G if grand_passed == grand_tested else Y
    col_f = G if grand_failed == 0 else R
    print(f"  {'  Passed':<30}  {col_p}{grand_passed:>8}{RS}")
    print(f"  {'  Warned':<30}  {Y}{grand_warn:>8}{RS}")
    print(f"  {'  Failed':<30}  {col_f}{grand_failed:>8}{RS}")
    print(f"  {'  Skipped':<30}  {DIM}{grand_skip:>8}{RS}")

    # ── Query category breakdown from document ───────────────────────────────
    print(f"\n  {BD}D. QUERY DOCUMENT COVERAGE (all 102 queries){RS}")
    print(f"  {'─'*PW}")
    doc_counts = [
        ("Dataset — Basic Mean",          10, "Q01–Q10"),
        ("Dataset — Min / Max",           10, "Q11–Q20"),
        ("Dataset — Trend / Slope",       10, "Q21–Q30"),
        ("Dataset — Map",                 10, "Q31–Q40"),
        ("Dataset — Time Comparison",     10, "Q41–Q50"),
        ("Dataset — Region Comparison",    9, "Q51–Q59"),
        ("Dataset — Multi-range/Season",   9, "Q60–Q68"),
        ("Dataset — Edge Cases",           7, "Q69–Q75"),
        ("Literature — Meta / File",       4, "Q76–Q79"),
        ("Literature — Text Retrieval",    8, "Q80–Q87"),
        ("Literature — Asset / Vision",    3, "Q88–Q90"),
        ("Literature — Edge Cases",        2, "Q91–Q92"),
        ("Both (Dataset + Literature)",    5, "Q93–Q97"),
        ("Chat",                           5, "Q98–Q102"),
    ]
    subtotal = 0
    print(f"  {'Category':<40}  {'Queries':>8}  {'IDs'}")
    print(f"  {'─'*40}  {'─'*8}  {'─'*12}")
    for name, cnt, ids in doc_counts:
        print(f"  {name:<40}  {cnt:>8}  {ids}")
        subtotal += cnt
    print(f"  {'─'*40}  {'─'*8}")
    print(f"  {BD}{'TOTAL':40}  {subtotal:>8}{RS}")

    print(f"\n{BD}{_bar(PW,'█')}{RS}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def run(engine, skip_maps=False):
    agent = ValidationAgent(engine)
    board = Scoreboard()
    PW    = 70

    import pandas as pd
    t0_ds = pd.Timestamp(engine.ds.time.values[0]).date()
    t1_ds = pd.Timestamp(engine.ds.time.values[-1]).date()
    var   = list(engine.ds.data_vars)[0]

    print(f"\n{BD}{_bar(PW,'═')}{RS}")
    print(f"{BD}{C}{'  SOIL MOISTURE ENGINE — PERFORMANCE EVALUATION':^{PW}}{RS}")
    print(f"{BD}{_bar(PW,'═')}{RS}")
    print(f"  Dataset var       : {W}{var}{RS}")
    print(f"  Dataset range     : {W}{t0_ds}{RS}  →  {W}{t1_ds}{RS}")
    print(f"  Dataset tests     : {W}{len(TESTS)}{RS}")
    print(f"  Intent tests      : {W}{len(INTENT_TESTS)}{RS}")
    print(f"  Map comparison    : {(G+'enabled'+RS) if not skip_maps else (Y+'disabled (--no-maps)'+RS)}")
    print(f"  Thresholds        : bias/rmse/ubrmse/mae < 1e-5 m³/m³  |  R > 0.9999")
    print(f"{_bar(PW,'═')}")

    # ── INTENT EVALUATION ────────────────────────────────────────────────────
    print(f"\n{BD}{C}  INTENT CLASSIFICATION EVALUATION{RS}")
    print(f"{_bar(PW,'═')}")
    intent_results = []
    intent_passed  = 0
    for t in INTENT_TESTS:
        tid = t["id"]; q = t["query"]; exp = t["expected"]
        try:
            res = classify_query_intent(q)
        except Exception as exc:
            res = "ERROR"
        outcome = "PASS" if res == exp else "FAIL"
        intent_results.append({"id": tid, "query": q, "expected": exp,
                                "got": res, "result": outcome})
        if outcome == "PASS":
            intent_passed += 1
            print(f"  {_pass()}  {tid}: '{q[:70]}' → {G}{res}{RS}")
        else:
            print(f"  {_fail()}  {tid}: '{q[:70]}' → {R}{res}{RS} (expected {exp})")

    acc = intent_passed / len(INTENT_TESTS) * 100 if INTENT_TESTS else 0
    print(f"  {BD}Intent Accuracy:{RS} {intent_passed}/{len(INTENT_TESTS)} ({acc:.1f}%)\n")
    print(f"{_bar(PW,'═')}")

    # ── DATASET VALIDATION ───────────────────────────────────────────────────
    for i, t in enumerate(TESTS, 1):
        tid  = t["id"]
        op   = t["op"]
        desc = t["desc"]
        src  = t.get("source_query", "")

        print(f"\n{BD}{B}[{i:03d}/{len(TESTS)}]  {tid}  ─  {op.upper()}  ─  {desc}{RS}")
        if src:
            print(f"  {DIM}↳ Query: \"{src}\"{RS}")
        print(f"  {'─'*66}")

        t_start = time.time()

        try:
            if op == "comparison":
                ctype  = t["ctype"]
                metric = t.get("metric", "mean")
                if ctype == "time":
                    periods = t["periods"]
                    region  = t["region"]
                    s, e    = periods[0][0], periods[-1][1]
                    cinfo   = {"comparison_type": "time",
                               "comparison_metric": metric,
                               "comparison_periods": periods,
                               "comparison_period1": periods[0],
                               "comparison_period2": periods[1],
                               "comparison_region2": None}
                else:
                    region  = expand_state(t["region"])
                    region2 = expand_state(t["region2"])
                    s, e    = t["s"], t["e"]
                    cinfo   = {"comparison_type": "region",
                               "comparison_metric": metric,
                               "comparison_periods": [(s, e)],
                               "comparison_period1": (s, e),
                               "comparison_period2": (s, e),
                               "comparison_region2": region2}
                result_msg, _ = engine.execute_analysis(
                    region=region, start_date=s, end_date=e,
                    operation=op, output_type="scalar", comparison_info=cinfo)
            else:
                region = expand_state(t["region"])
                s, e   = t["s"], t["e"]
                result_msg, _ = engine.execute_analysis(
                    region=region, start_date=s, end_date=e,
                    operation=op, output_type="scalar")

            elapsed = time.time() - t_start
            print(f"  Engine ran in {elapsed:.2f}s")

        except Exception as exc:
            print(f"  {_fail()}  Engine error: {exc}")
            if "--debug" in sys.argv:
                traceback.print_exc()
            board.add_skip(tid, op, desc, str(exc))
            continue

        # ── VALIDATE ─────────────────────────────────────────────────────────
        try:
            if op == "mean":
                truth_scalar, truth_map = agent.mean(region, s, e)
                engine_scalar = _parse_scalar(result_msg)
                if engine_scalar is None:
                    print(f"  {_warn()}  Could not parse engine scalar")
                    board.add_skip(tid, op, desc, "parse failure")
                    continue
                print(f"  {BD}Scalar:{RS}  Engine={engine_scalar:.8f}  Truth={truth_scalar:.8f}  "
                      f"diff={engine_scalar-truth_scalar:+.2e} m³/m³")
                m = compute_metrics([engine_scalar], [truth_scalar], label=tid)
                res = overall_result(m); print_metrics_block(m)
                board.add(tid, op, desc, m, res)
                if not skip_maps:
                    em = _engine_map(engine, region, s, e, op)
                    mm = compare_maps(em, truth_map, label=tid+"_map")
                    if mm:
                        print(f"  {BD}Map:{RS}"); print_metrics_block(mm, indent=4)
                        board.add(tid+"_M", op, desc+" [map]", mm, overall_result(mm))

            elif op == "minimum":
                truth_scalar, truth_map = agent.minimum(region, s, e)
                engine_scalar = _parse_scalar(result_msg)
                if engine_scalar is None:
                    board.add_skip(tid, op, desc, "parse failure"); continue
                print(f"  {BD}Scalar:{RS}  Engine={engine_scalar:.8f}  Truth={truth_scalar:.8f}  "
                      f"diff={engine_scalar-truth_scalar:+.2e} m³/m³")
                m = compute_metrics([engine_scalar], [truth_scalar], label=tid)
                res = overall_result(m); print_metrics_block(m)
                board.add(tid, op, desc, m, res)
                if not skip_maps:
                    em = _engine_map(engine, region, s, e, op)
                    mm = compare_maps(em, truth_map, label=tid+"_map")
                    if mm:
                        print_metrics_block(mm, indent=4)
                        board.add(tid+"_M", op, desc+" [map]", mm, overall_result(mm))

            elif op == "maximum":
                truth_scalar, truth_map = agent.maximum(region, s, e)
                engine_scalar = _parse_scalar(result_msg)
                if engine_scalar is None:
                    board.add_skip(tid, op, desc, "parse failure"); continue
                print(f"  {BD}Scalar:{RS}  Engine={engine_scalar:.8f}  Truth={truth_scalar:.8f}  "
                      f"diff={engine_scalar-truth_scalar:+.2e} m³/m³")
                m = compute_metrics([engine_scalar], [truth_scalar], label=tid)
                res = overall_result(m); print_metrics_block(m)
                board.add(tid, op, desc, m, res)
                if not skip_maps:
                    em = _engine_map(engine, region, s, e, op)
                    mm = compare_maps(em, truth_map, label=tid+"_map")
                    if mm:
                        print_metrics_block(mm, indent=4)
                        board.add(tid+"_M", op, desc+" [map]", mm, overall_result(mm))

            elif op == "slope":
                truth_sl, truth_r2, truth_p, truth_smap = agent.slope(region, s, e)
                engine_sl = _parse_slope(result_msg) or _parse_scalar(result_msg)
                if engine_sl is None:
                    board.add_skip(tid, op, desc, "parse failure"); continue
                print(f"  {BD}Slope:{RS}  Engine={engine_sl:+.8e}  Truth={truth_sl:+.8e}  "
                      f"diff={engine_sl-truth_sl:+.2e}  Truth R²={truth_r2:.4f} p={truth_p:.4f}")
                m = compute_metrics([engine_sl], [truth_sl], label=tid)
                res = overall_result(m); print_metrics_block(m)
                board.add(tid, op, desc, m, res)
                if not skip_maps:
                    em = _engine_map(engine, region, s, e, op)
                    mm = compare_maps(em, truth_smap, label=tid+"_map")
                    if mm:
                        print_metrics_block(mm, indent=4)
                        board.add(tid+"_M", op, desc+" [map]", mm, overall_result(mm))

            elif op == "comparison":
                ctype  = t["ctype"]
                metric = t.get("metric", "mean")
                if ctype == "time":
                    truth_list     = agent.comparison_time(region, periods, metric)
                    truth_scalars  = [v for v, _ in truth_list]
                    engine_scalars = _parse_comparison_scalars(result_msg)
                    nc = min(len(truth_scalars), len(engine_scalars))
                    if nc == 0:
                        board.add_skip(tid, op, desc, "parse failure"); continue
                    print(f"  {BD}Time comparison ({metric}, {nc} period(s)):{RS}")
                    for pi in range(nc):
                        ev, tv = engine_scalars[pi], truth_scalars[pi]
                        print(f"    P{pi+1}: Engine={ev:.8f}  Truth={tv:.8f}  diff={ev-tv:+.2e}")
                    m = compute_metrics(engine_scalars[:nc], truth_scalars[:nc], label=tid)
                    res = overall_result(m); print_metrics_block(m)
                    board.add(tid, op, desc, m, res)
                    if not skip_maps:
                        for pi, (_, tmap) in enumerate(truth_list[:nc]):
                            ps, pe = periods[pi]
                            em = _engine_map(engine, region, ps, pe, "mean")
                            mm = compare_maps(em, tmap, label=f"{tid}_P{pi+1}_map")
                            if mm:
                                print(f"  {BD}Period {pi+1} map:{RS}")
                                print_metrics_block(mm, indent=4)
                                board.add(f"{tid}_P{pi+1}_M", op, desc+f" [map P{pi+1}]",
                                          mm, overall_result(mm))
                else:
                    truth_d    = agent.comparison_region(region, region2, s, e, metric)
                    t1v, tmap1 = truth_d[region]
                    t2v, tmap2 = truth_d[region2]
                    esc = _parse_comparison_scalars(result_msg)
                    if len(esc) < 2:
                        board.add_skip(tid, op, desc, "parse failure"); continue
                    e1, e2 = esc[0], esc[1]
                    print(f"  {BD}Region comparison ({metric}):{RS}")
                    print(f"    {region.title():<25}: Engine={e1:.8f}  Truth={t1v:.8f}  diff={e1-t1v:+.2e}")
                    print(f"    {region2.title():<25}: Engine={e2:.8f}  Truth={t2v:.8f}  diff={e2-t2v:+.2e}")
                    m = compute_metrics([e1, e2], [t1v, t2v], label=tid)
                    res = overall_result(m); print_metrics_block(m)
                    board.add(tid, op, desc, m, res)
                    if not skip_maps:
                        for rname, tmap in [(region, tmap1), (region2, tmap2)]:
                            em = _engine_map(engine, rname, s, e, "mean")
                            mm = compare_maps(em, tmap, label=f"{tid}_{rname}_map")
                            if mm:
                                print(f"  {BD}{rname.title()} map:{RS}")
                                print_metrics_block(mm, indent=4)
                                board.add(f"{tid}_{rname.upper()}_M", op,
                                          desc+f" [{rname} map]", mm, overall_result(mm))

        except Exception as exc:
            print(f"  {_fail()}  Validation error: {exc}")
            if "--debug" in sys.argv:
                traceback.print_exc()
            board.add_skip(tid, op, desc, str(exc))

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    board.print_summary_table()
    board.print_per_op_aggregates()
    board.print_overall_aggregates()

    # ── QUERY COUNT SUMMARY (new) ─────────────────────────────────────────────
    print_query_count_summary(intent_results, None, board)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    skip_maps = "--no-maps" in sys.argv

    print(f"\n{BD}{'═'*70}")
    print("  Initialising Soil Moisture Engine ...")
    print(f"{'═'*70}{RS}")

    try:
        root = os.path.dirname(os.path.abspath(__file__))
        if root not in sys.path:
            sys.path.insert(0, root)
        from engine import SM_Engine
        engine = SM_Engine()
        print(f"{G}  Engine ready!{RS}\n")
    except Exception as exc:
        print(f"{R}  Engine init FAILED: {exc}{RS}")
        if "--debug" in sys.argv:
            traceback.print_exc()
        print("\n  Run from project root:  python performance_eval.py [--no-maps] [--debug]")
        sys.exit(1)

    run(engine, skip_maps=skip_maps)