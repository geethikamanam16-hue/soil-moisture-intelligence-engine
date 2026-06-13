"""
main.py
=======
Main Application - Soil Moisture Analysis Engine
Advanced NLP-powered analysis with flexible output control

UPDATES v2.5-v2.7: see prior changelog in original file.

UPDATES v2.8 (IMAGE DISPLAY FIX):
  ✅ FIX #10: answer_from_literature now returns 3 values (answer, found, image_list).
              All call sites updated.
  ✅ FIX #11: display_images_from_answer() added — surfaces related images to the
              user after every vision/asset answer via PIL, imgcat, or OS viewer.
  ✅ FIX #12: Indirect visual queries (e.g. "explain the methodology chart") now
              return and display the related image(s) alongside the llava:7b answer.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import re
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import Config
from engine import SM_Engine
from agent import OllamaAgent
from Query_classifier import QueryClassifier
from utils import QueryValidator, DateAnalyzer
from literature_manager import LiteratureManager
from literature_qa import (
    answer_from_literature,
    get_literature_context,
    parse_literature_command,
)
from intent_classifier import classify_query_intent

# ============================================================================
# LITERATURE HELP BLOCK
# ============================================================================

LITERATURE_HELP_BLOCK = """
  ─────────────────────────────────────────────────────────────────────────
  📚  LITERATURE Q&A  (text + vision + tables)
  ─────────────────────────────────────────────────────────────────────────
  load literature <path>    Load PDF / DOCX / TXT files
  list literature           Show loaded literature files
  clear literature          Remove all literature
  vision status             Show extracted figures + tables summary
  vision images [file]      List figures for a specific PDF
  list tables               List all extracted tables with links

  FIGURE / TABLE LINKS:
    "show me figure 3"               → opens figure 3 in image viewer
    "give me table 2 from anoop"     → opens table 2 image + shows data
    "where is image 5"               → opens file + prints path
    "open figure on page 4"          → opens all assets on page 4

  INDIRECT IMAGE QUERIES:
    "explain the methodology chart"  → llava:7b explains + shows related image
    "what does the scatter plot show" → llava:7b answers + shows related image
    "describe the trend graph"       → llava:7b answers + shows related image

  ROUTING:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Dataset statistics / maps       → DATASET                           │
  │ Scientific explanations/papers  → LITERATURE (text)                 │
  │ Figures / charts / diagrams     → LITERATURE (vision / llava:7b)      │
  │ Table content / data            → LITERATURE (table extraction)     │
  │ Explicit request for both       → BOTH                              │
  └──────────────────────────────────────────────────────────────────────┘

  DATASET EXAMPLES:
    What is mean soil moisture in India in 2020?
    Show mean moisture values of Kerala in 2007 and 2019
    Show map of mean moisture values of Kerala in 2020 and 2021
    Compare India in 2018, 2020 and 2022
    Compare Rajasthan and Gujarat in 2021

  LITERATURE TEXT EXAMPLES:
    Explain AMSR2 retrieval algorithm
    What RMSE was reported for AMSR2 validation?
    Summarise LPRM_Anoop.pdf in 3 points

  BOTH EXAMPLES:
    Compare 2020 drought data and explain literature findings
  ─────────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# HEADER
# ============================================================================

def display_header():
    print("\n" + "=" * 75)
    print("🌍 SOIL MOISTURE ANALYSIS ENGINE — ADVANCED NLP + VISION INTERFACE")
    print("=" * 75)
    print("""
📌 SUPPORTED QUERIES

  Mean / Average
    "What is average moisture in Rajasthan for June 2022?"
    "Show mean moisture values of Kerala in 2007 and 2019"     ← multi-year

  Trend
    "Show moisture trend in Punjab during monsoon 2022"

  Minimum / Maximum
    "Find minimum moisture in Kerala for July 2022"

  Comparison (any number of periods)
    "Compare Rajasthan and Gujarat in 2021"
    "Compare India between 2020 and 2023"
    "Compare India in 2018, 2020 and 2022"                     ← 3-way

📅 DATE FORMATS
   June 2022 | 2022-06-15 | monsoon 2022 | annual 2022
   2020 and 2021 | 2018, 2020 and 2022 | between 2020 and 2023

📊 OUTPUT OPTIONS
   scalar / map / both   (default: both — always shows map + stats)
""")
    print(LITERATURE_HELP_BLOCK)
    print("\nType 'exit' to quit, 'help' to display this again.")
    print("=" * 75 + "\n")

# ============================================================================
# SPLIT MULTI-QUESTION INPUT
# ============================================================================

def split_queries(raw: str) -> list:
    parts   = re.split(r"\s*\?\s*", raw)
    queries = [p.strip() for p in parts if len(p.strip()) >= 5]
    return queries if queries else [raw.strip()]

# ============================================================================
# SANITISE RAW INPUT
# ============================================================================

def sanitise_input(raw: str) -> str:
    cleaned = raw.strip().strip("`")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    return cleaned.strip()

# ============================================================================
# HANDLE UNCLEAR QUERIES
# ============================================================================

def handle_unclear_query(classification, classifier):
    if classification.get("query_clarity") == "clear":
        return classification

    print("\n⚠️  Query interpretation is uncertain.")
    print(classifier.describe(classification))

    proceed = input("\nProceed with this interpretation? (yes / no): ").strip().lower()
    if proceed not in ["yes", "y"]:
        print("\nPlease enter a clearer query.")
        print("Suggestions:")
        print("  • Mention region/state clearly")
        print("  • Mention operation (mean/trend/max/min/compare)")
        print("  • Mention dates clearly")
        print("  • For multi-year: '2020 and 2022' or '2018, 2020 and 2022'")
        print("  • For comparison: mention 'compare' explicitly")
        return None
    return classification

# ============================================================================
# FORMAT ANALYSIS OUTPUT
# ============================================================================

def format_analysis_output(result_message, visualization_created, output_type,
                            viz_filename="latest_analysis.png"):
    output  = "\n" + "=" * 75 + "\n"
    output += result_message

    if visualization_created and output_type in ["map", "both"]:
        output += f"\n✅ Visualization saved to '{viz_filename}'"
    elif output_type in ["map", "both"] and not visualization_created:
        output += "\n⚠️  Visualization generation failed."

    output += "\n" + "=" * 75 + "\n"
    return output

# ============================================================================
# PRINT INTERPRETATION
# ============================================================================

def print_interpretation(cls, classifier):
    print("\n✅ Interpretation")
    print(classifier.describe(cls))

# ============================================================================
# BUILD COMPARISON INFO
# ============================================================================

def build_comparison_info(cls: dict) -> dict:
    return {
        "comparison_type"   : cls.get("comparison_type",    "time"),
        "comparison_metric" : cls.get("comparison_metric",  "mean"),
        "comparison_periods": cls.get("comparison_periods", []),
        "comparison_period1": cls.get("comparison_period1"),
        "comparison_period2": cls.get("comparison_period2"),
        "comparison_region2": cls.get("comparison_region2"),
    }

# ============================================================================
# DATASET BOUNDS
# ============================================================================

def get_dataset_bounds(engine):
    try:
        import pandas as pd
        times = engine.ds["time"].values
        min_t = pd.Timestamp(times.min()).strftime("%Y-%m-%d")
        max_t = pd.Timestamp(times.max()).strftime("%Y-%m-%d")
        return min_t, max_t
    except Exception:
        return None, None

# ============================================================================
# DATE RANGE VALIDATION
# ============================================================================

def check_date_bounds(start_date, end_date, ds_start, ds_end):
    if ds_start is None:
        return True, start_date, end_date, ""
    
    if end_date < ds_start or start_date > ds_end:
        return False, start_date, end_date, (
            f"❌ Requested period ({start_date} to {end_date}) "
            f"is outside dataset range.\n"
            f"Available dataset: {ds_start} → {ds_end}"
        )
    
    warn = ""
    new_start = start_date
    new_end = end_date
    
    if start_date < ds_start:
        warn += f"⚠️ Data is only available from {ds_start}.\n"
        new_start = ds_start
    if end_date > ds_end:
        warn += f"⚠️ Data is only available till {ds_end}.\n"
        new_end = ds_end
        
    return True, new_start, new_end, warn


def _apply_date_correction(validator, start_date, end_date):
    dv = validator.validate_dates(start_date, end_date)
    if not dv['valid']:
        return False, start_date, end_date, dv['message']
    return True, dv['start_date'], dv['end_date'], dv['message']

# ============================================================================
# REGION RESOLUTION
# ============================================================================

def resolve_region(cls: dict, engine) -> bool:
    from difflib import get_close_matches

    all_valid = list(engine.available_regions) + ['india']

    if not cls.get('region_missing'):
        region = cls.get('region', '')
        if not region:
            cls['region_missing'] = True
        else:
            region_lower = region.lower()
            if region_lower not in [r.lower() for r in all_valid]:
                print(f"\n❌ Data unavailable for region '{region.title()}'.")
                print(f"   This region is not covered by the dataset.")
                _print_available_regions(engine)
                return False
            return True

    print("\n" + "─" * 60)
    print("⚠️  No region was detected in your query.")
    print("   Please specify an Indian state or 'India' for national-level analysis.")
    _print_available_regions(engine)
    print("─" * 60)

    user_input = input("\n📍 Enter region name: ").strip()

    if not user_input:
        print("❌ No region entered. Query cancelled.")
        return False

    user_lower = user_input.lower()

    if user_lower in [r.lower() for r in all_valid]:
        for r in all_valid:
            if r.lower() == user_lower:
                cls['region']         = r
                cls['region_missing'] = False
                print(f"   ✅ Region set to: {r.title()}")
                return True

    close = get_close_matches(user_lower, [r.lower() for r in all_valid],
                               n=1, cutoff=0.6)
    if close:
        matched_lower = close[0]
        canonical = next(r for r in all_valid if r.lower() == matched_lower)
        if matched_lower != user_lower:
            confirm = input(
                f"   Did you mean '{canonical.title()}'? (yes / no): "
            ).strip().lower()
            if confirm not in ('yes', 'y'):
                print("❌ Region not confirmed. Query cancelled.")
                return False
        cls['region']         = canonical
        cls['region_missing'] = False
        print(f"   ✅ Region set to: {canonical.title()}")
        return True

    print(f"\n❌ Data unavailable for region '{user_input.title()}'.")
    print(f"   '{user_input}' is not covered by this dataset.")
    _print_available_regions(engine)
    return False


def _print_available_regions(engine):
    sorted_regions = sorted(engine.available_regions)
    lines = []
    row   = []
    for i, r in enumerate(sorted_regions):
        row.append(r.title())
        if len(row) == 4:
            lines.append("   " + " | ".join(f"{s:<22}" for s in row))
            row = []
    if row:
        lines.append("   " + " | ".join(f"{s:<22}" for s in row))

    print("\n   📋 Regions available in this dataset:")
    for line in lines:
        print(line)
    print()

# ============================================================================
# IMAGE DISPLAY HELPER  (NEW in v2.8)
# ============================================================================

def display_images_from_answer(image_display_list: list):
    """
    Display images returned by the literature pipeline after any vision answer.

    For direct requests ("show me figure 3") AND indirect queries
    ("explain the methodology chart"), any related images are surfaced here.

    Strategy (priority order):
      1. PIL .show() → opens OS default image viewer
      2. imgcat      → inline terminal rendering (iTerm2 / VS Code terminal)
      3. OS default viewer (startfile / open / xdg-open)
      4. Print path for manual copy-paste
    """
    if not image_display_list:
        return

    print(f"\n🖼️  {len(image_display_list)} related image(s) — opening now:")

    for i, dd in enumerate(image_display_list, start=1):
        rtype   = dd.get("type", "image")
        label   = "Table" if rtype == "table" else "Figure"
        caption = dd.get("caption", "")
        path    = dd.get("path", "")
        uri     = dd.get("uri", "")
        source  = dd.get("source", "")
        page    = dd.get("page", "?")
        idx     = dd.get("index", "?")

        print(f"\n  [{i}] {label} {idx}  |  page {page}  |  {source}")
        if caption:
            print(f"       Caption : {caption}")
        if path:
            print(f"       Path    : {path}")
        if uri:
            print(f"       Link    : {uri}")

        if not path or not os.path.isfile(path):
            print("       ⚠️  Image file not found on disk.")
            continue

        opened = False

        # ── 1. PIL (Disabled to prevent popping out) ──────────────────
        # try:
        #     from PIL import Image as PILImage
        #     img = PILImage.open(path)
        #     img.show(title=f"{label} {idx} — {source}")
        #     print("       ✅ Opened in image viewer (PIL).")
        #     opened = True
        # except ImportError:
        #     pass
        # except Exception as e:
        #     print(f"       ⚠️  PIL open failed: {e}")

        # ── 2. imgcat (inline terminal) ───────────────────────────────
        if not opened:
            try:
                result = subprocess.run(
                    ["imgcat", path],
                    capture_output=False,
                    timeout=5,
                )
                if result.returncode == 0:
                    opened = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # ── 3. OS default viewer (Disabled to prevent popping out) ─────
        # if not opened:
        #     try:
        #         if sys.platform.startswith("win"):
        #             os.startfile(path)          # type: ignore[attr-defined]
        #             print("       ✅ Opened with Windows viewer.")
        #             opened = True
        #         elif sys.platform == "darwin":
        #             subprocess.Popen(["open", path])
        #             print("       ✅ Opened with macOS viewer.")
        #             opened = True
        #         else:
        #             subprocess.Popen(["xdg-open", path])
        #             print("       ✅ Opened with system viewer.")
        #             opened = True
        #     except Exception as e:
        #         print(f"       ⚠️  OS viewer failed: {e}")

        if not opened:
            print("       ℹ️  Copy the path or link above to open/view the image.")

# ============================================================================
# PROCESS SINGLE QUERY (dataset path)
# ============================================================================


def _run_state_ranking_query(cls, engine, validator, ds_start, ds_end):
    import xarray as xr
    from Query_classifier import VALID_REGIONS
    
    op = cls['operation']
    s = cls['start_date']
    e = cls['end_date']
    
    valid, s, e, msg = _apply_date_correction(validator, s, e)
    if not valid:
        print(f"❌ {msg}")
        return
        
    ok, s, e, msg = check_date_bounds(s, e, ds_start, ds_end)
    if not ok:
        print(f"❌ {msg}")
        return
    if msg: print(msg)

    print(f"⏳ Calculating means for all states/territories. This may take ~20 seconds...")
    state_means = []
    
    # We only check valid Indian states, excluding the country 'india'
    states = [r for r in VALID_REGIONS if r != 'india']
    
    for state in states:
        try:
            result = engine.analyze(
                region=state,
                start_date=s,
                end_date=e,
                operation='mean',
                output_type='text'
            )
            # engine.analyze returns a dict with 'analysis_result' which is a float or dict
            val = result.get('analysis_result')
            if isinstance(val, (int, float)):
                state_means.append((state.title(), val))
        except Exception as ex:
            pass # Skip states with geometry errors or empty bounds
            
    if not state_means:
        print("❌ Could not calculate means for states.")
        return
        
    # Sort
    state_means.sort(key=lambda x: x[1])
    
    print(f"\n╔{'═'*72}╗")
    print(f"║ {'🏆 STATE RANKING ANALYSIS':<70} ║")
    print(f"╠{'═'*72}╣")
    print(f"║ Period:   {s} to {e}")
    print(f"╠{'═'*72}╣")
    
    if op == 'driest_state':
        winner = state_means[0]
        print(f"║  🏜️ DRIEST STATE: {winner[0]} ({winner[1]:.6f} m³/m³)")
        print(f"║")
        print(f"║  Bottom 3 Driest:")
        for i, (st, val) in enumerate(state_means[:3]):
            print(f"║    {i+1}. {st}: {val:.6f}")
    else:
        winner = state_means[-1]
        print(f"║  💧 WETTEST STATE: {winner[0]} ({winner[1]:.6f} m³/m³)")
        print(f"║")
        print(f"║  Top 3 Wettest:")
        for i, (st, val) in enumerate(reversed(state_means[-3:])):
            print(f"║    {i+1}. {st}: {val:.6f}")
            
    print(f"╚{'═'*72}╝\n")

def process_single_query(
    query,
    engine,
    classifier,
    agent,
    validator,
    ds_start,
    ds_end,
    lit_manager=None,
    query_index=None,
    intent="dataset",
):
    if query_index is not None:
        print(f"\n{'─'*75}")
        print(f"🔹 QUERY {query_index}: {query}")
        print(f"{'─'*75}")

    print("\n⏳ Processing query...")

    cls = classifier.classify(query)

    if cls.get("query_clarity") in ["unclear", "ambiguous"]:
        cls = agent.process_query(cls)

    if cls.get("query_clarity") != "clear":
        cls = handle_unclear_query(cls, classifier)
        if cls is None:
            return

    print_interpretation(cls, classifier)

    if not resolve_region(cls, engine):
        return

    ov = validator.validate_operation(cls["operation"])
    if not ov["valid"]:
        print(f"❌ {ov['message']}")
        return

    if cls["operation"] == "comparison":
        _run_comparison_query(
            cls, engine, validator, ds_start, ds_end, lit_manager, intent
        )
        return

    all_ranges = cls.get("all_date_ranges", [])

    if len(all_ranges) <= 1:
        _run_single_date_range(
            cls, engine, validator, ds_start, ds_end, lit_manager, intent,
            start_date=cls["start_date"],
            end_date=cls["end_date"],
        )
        return

    print(f"\n📋 {len(all_ranges)} date ranges detected — running each separately.\n")
    for i, (s, e) in enumerate(all_ranges, 1):
        valid, s, e, msg = _apply_date_correction(validator, s, e)
        if not valid:
            print(f"\n[Range {i}] ❌ {msg}")
            continue

        ok, bounds_msg = check_date_bounds(s, e, ds_start, ds_end)
        if not ok:
            print(f"\n[Range {i}] {bounds_msg}")
            continue
        if bounds_msg:
            print(bounds_msg)

        print(f"\n{'─'*60}")
        print(f"📅 Range {i} of {len(all_ranges)}: {s} → {e}")
        print(f"{'─'*60}")

        viz_filename = None
        if cls["output_type"] in ["map", "both"]:
            from utils import get_unique_viz_filename
            viz_filename = get_unique_viz_filename(cls["operation"], index=i)

        _run_single_date_range(
            cls, engine, validator, ds_start, ds_end, lit_manager, intent,
            start_date=s,
            end_date=e,
            viz_filename=viz_filename,
        )


def _run_single_date_range(cls, engine, validator, ds_start, ds_end,
                            lit_manager, intent,
                            start_date=None, end_date=None,
                            viz_filename="latest_analysis.png"):
    s = start_date or cls["start_date"]
    e = end_date   or cls["end_date"]

    if not s or not e:
        print("❌ Could not determine dates from query.")
        return

    valid, s, e, msg = _apply_date_correction(validator, s, e)
    if not valid:
        print(f"❌ {msg}")
        return

    ok, bounds_msg = check_date_bounds(s, e, ds_start, ds_end)
    if not ok:
        print(bounds_msg)
        return
    if bounds_msg:
        print(bounds_msg)

    print("\n🔍 Executing analysis...")
    result_msg, viz_created = engine.execute_analysis(
        region      = cls["region"],
        start_date  = s,
        end_date    = e,
        operation   = cls["operation"],
        output_type = cls["output_type"],
    )

    actual_viz_filename = None
    if viz_created:
        import shutil
        from utils import get_unique_viz_filename
        actual_viz_filename = viz_filename or get_unique_viz_filename(cls["operation"])
        try:
            shutil.copy("latest_analysis.png", actual_viz_filename)
        except Exception as mv_err:
            print(f"⚠️  Could not copy output file: {mv_err}")
            actual_viz_filename = "latest_analysis.png"

    print(format_analysis_output(result_msg, viz_created, cls["output_type"],
                                  viz_filename=actual_viz_filename))

    if hasattr(engine, "ds") and "time" in engine.ds.dims:
        try:
            available_times = engine.ds["time"].values
            missing_info    = DateAnalyzer.find_missing_dates(available_times, s, e)
            if missing_info["has_gaps"]:
                print(DateAnalyzer.format_missing_report(
                    missing_info, cls["region"], s, e
                ))
        except Exception:
            pass


def _run_comparison_query(cls, engine, validator, ds_start, ds_end,
                           lit_manager, intent):
    comparison_info = build_comparison_info(cls)
    ctype = comparison_info["comparison_type"]

    output_type = cls.get("output_type", "both")
    if output_type not in ("scalar", "map", "both"):
        output_type = "both"
    if output_type == "scalar":
        print("ℹ️  Comparison: scalar-only mode (no map will be generated).")
    cls["output_type"] = output_type

    if ctype == "time":
        periods = comparison_info.get("comparison_periods", [])
        if len(periods) < 2:
            print("❌ Two or more time periods required.")
            return

        corrected_periods = []
        for i, (s, e) in enumerate(periods, 1):
            valid, s, e, msg = _apply_date_correction(validator, s, e)
            if not valid:
                print(f"❌ Period {i}: {msg}")
                return
            ok, bounds_msg = check_date_bounds(s, e, ds_start, ds_end)
            if not ok:
                print(f"❌ Period {i}: {bounds_msg}")
                return
            if bounds_msg:
                print(bounds_msg)
            corrected_periods.append((s, e))

        comparison_info["comparison_periods"] = corrected_periods
        comparison_info["comparison_period1"] = corrected_periods[0]
        comparison_info["comparison_period2"] = corrected_periods[1]
        cls["start_date"] = corrected_periods[0][0]
        cls["end_date"]   = corrected_periods[-1][1]

        print(f"\n📋 Time comparison: {len(corrected_periods)} periods detected.")
        for i, (s, e) in enumerate(corrected_periods, 1):
            print(f"   Period {i}: {s} → {e}")

    elif ctype == "region":
        if not comparison_info["comparison_region2"]:
            print("❌ Two regions required.")
            return
        s = cls["start_date"]
        e = cls["end_date"]
        if not s or not e:
            print("❌ Could not determine dates from query.")
            return

        valid, s, e, msg = _apply_date_correction(validator, s, e)
        if not valid:
            print(f"❌ {msg}")
            return
        cls["start_date"] = s
        cls["end_date"]   = e

        ok, bounds_msg = check_date_bounds(s, e, ds_start, ds_end)
        if not ok:
            print(bounds_msg)
            return
        if bounds_msg:
            print(bounds_msg)

        r1 = cls["region"].title()
        r2 = comparison_info["comparison_region2"].title()
        print(f"\n📋 Region comparison: {r1} vs {r2}  ({s} → {e})")

    print("\n🔍 Executing analysis...")
    result_msg, viz_created = engine.execute_analysis(
        region          = cls["region"],
        start_date      = cls["start_date"],
        end_date        = cls["end_date"],
        operation       = cls["operation"],
        output_type     = cls["output_type"],
        comparison_info = comparison_info,
    )

    actual_viz_filename = None
    if viz_created:
        import shutil
        from utils import get_unique_viz_filename
        actual_viz_filename = get_unique_viz_filename(cls["operation"])
        try:
            shutil.copy("latest_analysis.png", actual_viz_filename)
        except Exception as e:
            print(f"⚠️  Could not copy output file: {e}")
            actual_viz_filename = "latest_analysis.png"

    print(format_analysis_output(result_msg, viz_created, cls["output_type"],
                                  viz_filename=actual_viz_filename))

    if hasattr(engine, "ds") and "time" in engine.ds.dims:
        try:
            available_times = engine.ds["time"].values
            missing_info    = DateAnalyzer.find_missing_dates(
                available_times, cls["start_date"], cls["end_date"]
            )
            if missing_info["has_gaps"]:
                print(DateAnalyzer.format_missing_report(
                    missing_info, cls["region"],
                    cls["start_date"], cls["end_date"],
                ))
        except Exception:
            pass

# ============================================================================
# NOTEBOOK-STYLE LITERATURE ANSWER PRINTER
# ============================================================================

def print_literature_answer(query: str, answer: str, source_label: str = ""):
    import textwrap
    width = 75

    answer_body    = answer
    asset_section  = ""
    source_section = ""

    if "\n📎 Source asset(s):" in answer:
        parts       = answer.split("\n📎 Source asset(s):", 1)
        answer_body = parts[0].strip()
        rest        = "📎 Source asset(s):" + parts[1]
        if "\n─" in rest and "\n📚 Sources:" in rest:
            rest_parts     = rest.split("\n─", 1)
            asset_section  = rest_parts[0].strip()
            source_section = ("─" + rest_parts[1]).strip()
        else:
            asset_section = rest.strip()

    elif "\n─" * 1 in answer and "📚 Sources:" in answer:
        parts          = answer.split("\n─", 1)
        answer_body    = parts[0].strip()
        source_section = ("─" + parts[1]).strip()

    print("\n" + "╔" + "═" * (width - 2) + "╗")
    print("║  📖  LITERATURE ANSWER" + " " * (width - 25) + "║")
    print("╠" + "═" * (width - 2) + "╣")

    if source_label:
        label_line = f"║  Source : {source_label}"
        print(label_line + " " * max(0, width - len(label_line) - 1) + "║")
        print("╠" + "═" * (width - 2) + "╣")

    print("║" + " " * (width - 2) + "║")

    for line in answer_body.splitlines():
        if not line.strip():
            print("║" + " " * (width - 2) + "║")
            continue
        wrapped = textwrap.wrap(line, width=width - 6) or [""]
        for wl in wrapped:
            row = f"║  {wl}"
            print(row + " " * max(0, width - len(row) - 1) + "║")

    print("║" + " " * (width - 2) + "║")

    if asset_section:
        print("╠" + "═" * (width - 2) + "╣")
        print("║  📎  ASSETS / LINKS" + " " * (width - 22) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        print("║" + " " * (width - 2) + "║")
        for line in asset_section.splitlines():
            if not line.strip() or line.startswith("📎 Source asset(s):"):
                continue
            if len(line) <= width - 6:
                row = f"║  {line}"
                print(row + " " * max(0, width - len(row) - 1) + "║")
            else:
                row = f"║  {line[:width-9]}..."
                print(row + " " * max(0, width - len(row) - 1) + "║")
        print("║" + " " * (width - 2) + "║")

    if source_section:
        print("╠" + "═" * (width - 2) + "╣")
        print("║  📚  CITATIONS" + " " * (width - 17) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        print("║" + " " * (width - 2) + "║")
        for line in source_section.splitlines():
            if line.startswith("─") or line.startswith("📚 Sources:"):
                continue
            if not line.strip():
                continue
            row = f"║  {line}"
            print(row + " " * max(0, width - len(row) - 1) + "║")
        print("║" + " " * (width - 2) + "║")

    print("╚" + "═" * (width - 2) + "╝")
    if asset_section:
        _print_full_paths(asset_section)


def _print_full_paths(asset_text: str):
    lines    = asset_text.splitlines()
    has_long = any(
        ("Path" in l or "Link" in l or "file://" in l) and len(l) > 60
        for l in lines
    )
    if not has_long:
        return
    print("\n  📋 Full paths (for copy-paste):")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("Path", "Link", "file://")):
            print(f"     {stripped}")


def print_asset_answer(answer: str):
    width = 75
    print("\n" + "┌" + "─" * (width - 2) + "┐")
    print("│  📎  FIGURE / TABLE LOOKUP" + " " * (width - 29) + "│")
    print("├" + "─" * (width - 2) + "┤")

    path_lines = []
    for line in answer.splitlines():
        if not line.strip():
            print("│" + " " * (width - 2) + "│")
            continue
        if ("Path" in line or "Link" in line or "file://" in line) and len(line) > width - 6:
            path_lines.append(line.strip())
            short = line[:width - 9] + "..."
            row   = f"│  {short}"
            print(row + " " * max(0, width - len(row) - 1) + "│")
        else:
            row = f"│  {line}"
            print(row + " " * max(0, width - len(row) - 1) + "│")

    print("└" + "─" * (width - 2) + "┘")
    if path_lines:
        print("\n  📋 Full paths (copy-paste):")
        for p in path_lines:
            print(f"     {p}")


def _is_asset_lookup_answer(answer: str) -> bool:
    indicators = ["📎", "🖼️  Figure", "📊 Table",
                  "Path    :", "Link    :", "file://", "Caption :", "Preview :"]
    return any(ind in answer for ind in indicators)

# ============================================================================
# LITERATURE HELPERS
# ============================================================================

def _resolve_lit_path(config_path: str) -> str:
    if os.path.isabs(config_path):
        return config_path
    return os.path.join(PROJECT_ROOT, config_path)


def _setup_literature(lit_manager: LiteratureManager, lit_dir: str):
    print(f"\n📚 Literature folder: {lit_dir}")
    if not os.path.isdir(lit_dir):
        os.makedirs(lit_dir, exist_ok=True)
        print("📁 Literature folder created. Add files and restart.")
        return

    supported_exts = {".pdf", ".docx", ".txt", ".md"}
    all_files = [
        f for f in sorted(os.listdir(lit_dir))
        if os.path.splitext(f)[1].lower() in supported_exts
    ]

    if not all_files:
        print("⚠️  No literature files found.")
        return

    print(f"Found {len(all_files)} file(s). Loading literature...")
    lit_manager.load_directory(lit_dir)
    lit_manager.set_literature_dir(lit_dir)

    sources = lit_manager.list_sources()
    if sources:
        print("\n" + lit_manager.summary())
    else:
        print("⚠️  Literature loaded but no text extracted.")

# ============================================================================
# MAIN APPLICATION LOOP
# ============================================================================

def run_app():
    display_header()

    try:
        print("🔌 Initialising Soil Moisture Engine...")
        engine = SM_Engine()
        print("✅ Engine initialised!")
    except Exception as e:
        print(f"❌ Engine initialisation failed: {e}")
        print("Check Config paths.")
        sys.exit(1)

    classifier = QueryClassifier()
    agent      = OllamaAgent(model_name=Config.OLLAMA_MODEL)
    validator  = QueryValidator()

    lit_index_path = _resolve_lit_path(Config.LITERATURE_INDEX_PATH)

    try:
        from cloud.literature_manager import sync_literature
        lit_dir = sync_literature()
    except Exception as e:
        print(f"⚠️  Could not sync literature from Google Drive: {e}")
        lit_dir = _resolve_lit_path(Config.LITERATURE_DIR)

    lit_manager = LiteratureManager(
        index_path        = lit_index_path,
        vision_enabled    = Config.VISION_ENABLED,
        vision_cache_dir  = Config.VISION_IMAGE_CACHE_DIR,
        vision_max_images = Config.VISION_MAX_IMAGES_PER_PDF,
    )
    _setup_literature(lit_manager, lit_dir)

    ds_start, ds_end = get_dataset_bounds(engine)
    if ds_start and ds_end:
        print(f"\n📅 Dataset covers: {ds_start} → {ds_end}\n")
    else:
        print("\n⚠️  Could not determine dataset range.\n")

    print("🤖 System ready! Semantic + Vision + Table routing active.\n")

    while True:
        try:
            raw_input_text = input("\n📝 Enter query (or 'exit' / 'help'): ").strip()

            if raw_input_text.lower() in ["exit", "quit", "bye", "q"]:
                print("\n👋 Thank you for using the system!")
                break

            if raw_input_text.lower() in ["help", "?"]:
                display_header()
                continue

            if not raw_input_text or len(raw_input_text) < 5:
                print("⚠️  Please enter a more detailed query.")
                continue

            raw_input_text = sanitise_input(raw_input_text)

            lower_input = raw_input_text.lower()
            if lower_input in ("list tables", "show tables", "tables"):
                result = parse_literature_command(raw_input_text, lit_manager)
                if result:
                    print(result)
                continue

            lit_cmd_result = parse_literature_command(raw_input_text, lit_manager)
            if lit_cmd_result is not None:
                print(lit_cmd_result)
                continue

            sub_queries = split_queries(raw_input_text)
            if len(sub_queries) > 1:
                print(f"\n📋 Detected {len(sub_queries)} queries.")

            for idx, sq in enumerate(sub_queries, start=1):
                try:
                    print("\n🧭 Classifying query intent...")
                    intent = classify_query_intent(
                        query        = sq,
                        ollama_url   = Config.OLLAMA_URL,
                        ollama_model = Config.OLLAMA_MODEL,
                        timeout      = Config.OLLAMA_TIMEOUT,
                    )

                    has_literature = bool(lit_manager.list_sources())

                    if intent in ("literature", "both"):
                        if not has_literature:
                            print("\n📚 No literature loaded.")
                            if intent == "literature":
                                continue
                            print("↩  Running dataset analysis only.")
                        else:
                            print("\n⏳ Searching literature (text + vision + tables)...")

                            source_filter = lit_manager.resolve_source_filter(sq)
                            if source_filter:
                                src_label = lit_manager.get_file_display_name(
                                    source_filter)
                                print(f"  📂 Scoped to: {source_filter}")
                            else:
                                src_label = ", ".join(
                                    lit_manager.get_file_display_name(s)
                                    for s in lit_manager.list_sources()
                                )
                                print(f"  📂 Searching all files: {src_label}")

                            # ── UPDATED: unpack 3 values ───────────────
                            ds_q_for_both = sq
                            lit_query     = sq
                            if intent == "both":
                                from utils import split_both_query_with_llm
                                ds_q_for_both, lit_query = split_both_query_with_llm(
                                    sq, Config.OLLAMA_URL, Config.OLLAMA_MODEL,
                                    Config.OLLAMA_TIMEOUT
                                )

                            answer, found_in_lit, image_display_list = answer_from_literature(
                                query          = lit_query,
                                lit_manager    = lit_manager,
                                ollama_url     = Config.OLLAMA_URL,
                                ollama_model   = Config.OLLAMA_MODEL,
                                ollama_timeout = Config.OLLAMA_TIMEOUT,
                                top_k          = Config.LITERATURE_TOP_K,
                                vision_model   = Config.VISION_MODEL,
                                vision_timeout = Config.VISION_TIMEOUT,
                                vision_top_k   = Config.VISION_TOP_K,
                                vision_enabled = Config.VISION_ENABLED,
                            )

                            if answer:
                                if _is_asset_lookup_answer(answer):
                                    print_asset_answer(answer)
                                else:
                                    print_literature_answer(sq, answer, src_label)
                                # ── NEW: display related images ────────
                                display_images_from_answer(image_display_list)
                            else:
                                print(
                                    "\n❓ No relevant passages, figures, "
                                    "or tables found."
                                )

                            if intent == "literature":
                                continue

                            print("\n🔄 Running dataset analysis as well...")

                    if intent in ("dataset", "both"):
                        _ds_q = ds_q_for_both if intent == "both" and 'ds_q_for_both' in dir() else sq
                        process_single_query(
                            _ds_q,
                            engine,
                            classifier,
                            agent,
                            validator,
                            ds_start,
                            ds_end,
                            lit_manager = (lit_manager if intent == "both" else None),
                            query_index = (idx if len(sub_queries) > 1 else None),
                            intent      = intent,
                        )

                except Exception as e:
                    print(f"\n❌ Error processing query: {e}")
                    if "--debug" in sys.argv:
                        import traceback
                        traceback.print_exc()

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            if "--debug" in sys.argv:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    run_app()