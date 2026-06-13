"""
QueryClassifier - Natural Language Query Parser for Soil Moisture Analysis

MODIFICATIONS v2.5 - MULTI-YEAR + N-WAY COMPARISON FIX:
  ✅ FIX #1: "YYYY and YYYY" no longer merges into one range (e.g. 2007 and 2019)
  ✅ FIX #2: "from YYYY to YYYY" kept as a single span range (explicit range intent)
  ✅ FIX #3: N bare years extracted independently (2007 and 2019 and 2022 → 3 ranges)
  ✅ FIX #4: N-way comparison date extraction supports any number of periods
  ✅ FIX #5: Comparison now stores comparison_periods as a list of N tuples
  ✅ FIX #6: Non-comparison queries with multiple years store all ranges for
             multi-year iteration in main.py
  ✅ Original fixes retained: trend vs comparison priority, between YYYY and YYYY,
     state spell-checking, ambiguous comparison detection.

MODIFICATIONS v2.6 - OUTPUT TYPE DETECTION FIX:
  ✅ FIX #7: _detect_output_type is now far more robust.
             - 'show', 'display', 'give', 'generate', 'produce' → map
             - 'map', 'spatial', 'visuali', 'geographic', 'plot' → map
             - Scalar triggers only fire on EXPLICIT scalar language
             - Comparison operation defaults to 'both' unless scalar is explicit
             - Prevents maps being silently skipped for natural-language queries

MODIFICATIONS v2.7 - REGION DETECTION FIX:
  ✅ FIX #8: No longer defaults to 'india' when no region is found.
             - Returns region=None and sets region_missing=True flag
             - Only defaults to 'india' when explicitly mentioned in query
             - main.py will intercept the flag and prompt the user
"""

import re
from datetime import datetime, timedelta
from difflib import get_close_matches

from Config import VALID_REGIONS, SEASONS

# ============================================================================
# COMPREHENSIVE LOCAL SEASONS
# ============================================================================
LOCAL_SEASONS = {
    'pre-monsoon': {'months': [3, 4, 5]},
    'premonsoon':  {'months': [3, 4, 5]},
    'post-monsoon':{'months': [10, 11]},
    'postmonsoon': {'months': [10, 11]},
    'rabi':        {'months': [10, 11, 12, 1, 2, 3]},
    'kharif':      {'months': [6, 7, 8, 9, 10, 11]},
    'summer':      {'months': [3, 4, 5]},
    'winter':      {'months': [12, 1, 2]},
    'spring':      {'months': [3, 4, 5]},
    'monsoon':     {'months': [6, 7, 8, 9]},
    'rainy':       {'months': [6, 7, 8, 9]},
    'autumn':      {'months': [9, 10, 11]},
    'fall':        {'months': [9, 10, 11]}
}

# ============================================================================
# METEOROLOGICAL SEASON ACRONYMS  (WMO standard abbreviations)
# Keys are LOWERCASE. Months listed in calendar order.
# ============================================================================

_SEASON_ACRONYMS = {
    'jjas':  [6, 7, 8, 9],      # June-July-August-September (SW monsoon)
    'mam':   [3, 4, 5],         # March-April-May (pre-monsoon / spring)
    'on':    [10, 11],          # October-November (post-monsoon)
    'djf':   [12, 1, 2],        # December-January-February (winter, wraps year)
    'djfm':  [12, 1, 2, 3],     # Winter + March
    'jf':    [1, 2],            # January-February
    'jja':   [6, 7, 8],         # June-July-August
    'son':   [9, 10, 11],       # September-October-November
    'amj':   [4, 5, 6],         # April-May-June
}

# Queries that signal the user wants a global/overview answer without a specific
# region or date — these should NOT crash the parser.
_GLOBAL_QUERY_SIGNALS = [
    'how many years', 'years of data', 'temporal coverage', 'data coverage',
    'dataset span', 'available data', 'time range', 'full dataset', 'all years',
    'how long', 'date range', 'since when', 'start date', 'end date',
    'data available', 'dataset range', 'coverage period', 'historical data',
    'archive coverage',
]

# Queries asking for national aggregates across the whole period — should be
# handled with a fast textual summary + Dashboard redirect instead of a 22-year scan.
_HEAVY_AGGREGATE_SIGNALS = [
    'which year', 'what year', 'best year', 'worst year',
    'highest national', 'lowest national', 'wettest year nationally',
    'driest year nationally', 'highest average soil moisture',
    'lowest average soil moisture', 'most moisture nationally',
    'least moisture nationally',
]


# ============================================================================
# MONTH NAME → NUMBER
# ============================================================================

MONTH_MAP = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12,
}

MONTH_NAMES = {v: k.capitalize() for k, v in MONTH_MAP.items() if len(k) > 3}


# ============================================================================
# HELPER: get first/last day of month / year
# ============================================================================

def _month_range(month: int, year: int):
    first = datetime(year, month, 1)
    if month == 12:
        last = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = datetime(year, month + 1, 1) - timedelta(days=1)
    return first.strftime('%Y-%m-%d'), last.strftime('%Y-%m-%d')


def _year_range(year: int):
    return f"{year}-01-01", f"{year}-12-31"


# ============================================================================
# SPELL CHECKER FOR STATE NAMES
# ============================================================================

class StateSpellChecker:
    """Auto-correct misspelled state names."""

    def __init__(self, valid_regions):
        # Include 'india' in the spellcheck list to handle national query typos
        self.valid_regions  = list(valid_regions)
        if 'india' not in self.valid_regions:
            self.valid_regions.append('india')
        self.sorted_regions = sorted(self.valid_regions, key=len, reverse=True)

    def auto_correct_regions_in_text(self, text: str) -> str:
        # Expand common abbreviations safely using word boundaries
        t = re.sub(r'\bup\b', 'uttar pradesh', text.lower())
        t = re.sub(r'\bmp\b', 'madhya pradesh', t)
        
        words           = t.split()
        corrected_words = []
        for word in words:
            # Skip spellchecking for standard parts of expanded abbreviations
            if word in ('uttar', 'pradesh', 'madhya'):
                corrected_words.append(word)
                continue
            close = get_close_matches(word, self.sorted_regions, cutoff=0.72, n=1)
            corrected_words.append(close[0] if close else word)
        return ' '.join(corrected_words)


# ============================================================================
# QUERY CLASSIFIER
# ============================================================================

class QueryClassifier:
    """
    Classifies natural-language queries into structured analysis parameters.

    KEY BEHAVIOUR (v2.4):
      • "Kerala in 2007 and 2019"     → two independent year ranges
      • "Kerala from 2007 to 2019"    → one span range (2007-01-01 to 2019-12-31)
      • "Kerala between 2007 and 2019"→ one span range
      • Comparison with N years       → N comparison_periods stored in list
      • result['all_date_ranges']     → every extracted range (for multi-run in main)

    KEY BEHAVIOUR (v2.6):
      • Output type detection is robust: show/display/give/generate → map
      • Comparison queries default to 'both' (never silently scalar-only)
      • Explicit scalar-only keywords required for scalar output

    KEY BEHAVIOUR (v2.7):
      • No longer defaults to 'india' when no region is detected
      • Returns region=None and region_missing=True so main.py can prompt user
      • Only 'india' / 'national' / 'country' keywords trigger India default
    """

    def __init__(self):
        self.valid_regions   = VALID_REGIONS
        self._sorted_regions = sorted(self.valid_regions, key=len, reverse=True)
        self.spell_checker   = StateSpellChecker(self.valid_regions)

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def classify(self, query: str) -> dict:
        q = self.spell_checker.auto_correct_regions_in_text(query.strip().lower())
        
        # ── 1. Expand Acronyms ──
        acronym_map = {
            r'\bap\b': 'andhra pradesh',
            r'\bup\b': 'uttar pradesh',
            r'\bmp\b': 'madhya pradesh',
            r'\bjk\b': 'jammu and kashmir',
            r'\btn\b': 'tamil nadu',
            r'\bhp\b': 'himachal pradesh'
        }
        for pat, repl in acronym_map.items():
            q = re.sub(pat, repl, q)

        # ── 2. Expand Seasons ──
        # Find year to firmly bind the season string if available
        y_match = re.search(r'\b(\d{4})\b', q)
        if y_match:
            y = y_match.group(1)
            q = re.sub(r'\b(jjas|monsoon)\b', f'from june {y} to september {y}', q, flags=re.IGNORECASE)
            q = re.sub(r'\bsummer\b', f'from march {y} to may {y}', q, flags=re.IGNORECASE)
            q = re.sub(r'\bwinter\b', f'from december {y} to february {y}', q, flags=re.IGNORECASE)
        else:
            q = re.sub(r'\b(jjas|monsoon)\b', 'from june to september', q, flags=re.IGNORECASE)
            q = re.sub(r'\bsummer\b', 'from march to may', q, flags=re.IGNORECASE)
            q = re.sub(r'\bwinter\b', 'from december to february', q, flags=re.IGNORECASE)

        result = {
            'valid': True,
            'message': 'OK',
            'region': None,               # v2.7: None until a region is found
            'region_missing': False,      # v2.7: flag for main.py to handle
            'operation': 'mean',
            'output_type': 'both',
            'start_date': None,
            'end_date': None,
            'query_clarity': 'clear',
            'comparison_type': None,
            'comparison_region2': None,
            'comparison_periods': [],      # list of (start, end) for N-way
            'comparison_period1': None,    # kept for back-compat
            'comparison_period2': None,    # kept for back-compat
            'comparison_metric': 'mean',
            'all_date_ranges': [],         # all ranges found (for multi-run)
            'original_query': query,       # preserve original for literature routing
            # v2.8 additions:
            'is_global_query':    False,   # dataset overview / coverage question
            'is_heavy_aggregate': False,   # multi-year national aggregate scan
        }

        if len(q) < 5:
            result['valid']   = False
            result['message'] = "Query too short. Please provide more details."
            return result

        # ── v2.8: Detect global overview queries EARLY ────────────────────
        # These do not need a region or specific date — app.py will handle
        # them with a fast textual summary and/or Dashboard redirect.
        if any(sig in q for sig in _GLOBAL_QUERY_SIGNALS) and not re.search(r'\b20\d{2}\b', q):
            result['is_global_query'] = True
            result['query_clarity']   = 'clear'
            # Default to India + full dataset span — let app.py substitute real ds_start/ds_end
            result['region']          = 'india'
            result['region_missing']  = False
            # Leave start/end date None — app.py replaces with ds_start/ds_end
            return result

        # ── v2.8: Detect heavy national aggregate queries ─────────────────
        # e.g. "which year had the highest national average soil moisture"
        if any(sig in q for sig in _HEAVY_AGGREGATE_SIGNALS):
            result['is_heavy_aggregate'] = True
            result['query_clarity']      = 'clear'
            result['region']             = 'india'
            result['region_missing']     = False
            # Leave dates None — app.py returns fast text summary
            return result

        # Step 1: operation
        result['operation'] = self._detect_operation(q)
        if result['operation'] in ['driest_state', 'wettest_state']:
            result['region'] = 'india'
            result['region_missing'] = False

        # Step 2: output type
        result['output_type'] = self._detect_output_type(q, result['operation'])

        # Step 3: regions
        regions = self._detect_all_regions(q)
        if regions:
            result['region']         = regions[0]
            result['region_missing'] = False
        elif result['operation'] not in ['driest_state', 'wettest_state']:
            # v2.7: no region detected — flag it for main.py to prompt user
            result['region']         = None
            result['region_missing'] = True

        if len(regions) >= 2:
            result['comparison_region2'] = regions[1]

        # Step 4: comparison meta
        if result['operation'] == 'comparison':
            result['comparison_type']   = self._detect_comparison_type(q, regions)
            result['comparison_metric'] = self._detect_comparison_metric(q)

        # Step 5: dates
        if result['operation'] == 'comparison' and result.get('comparison_type') != 'region':
            dates = self._extract_comparison_dates(q)
        else:
            dates = self._extract_all_date_ranges(q)

        result['all_date_ranges'] = dates

        # ----------------------------------------------------------------
        # COMPARISON LOGIC
        # ----------------------------------------------------------------
        if result['operation'] == 'comparison':

            if len(dates) >= 2:
                # N-way time comparison
                result['comparison_periods'] = dates
                result['comparison_period1'] = dates[0]   # back-compat
                result['comparison_period2'] = dates[1]   # back-compat
                result['start_date'] = dates[0][0]
                result['end_date']   = dates[-1][1]
                result['query_clarity'] = 'clear'

            elif len(regions) >= 2:
                # Region comparison (single time window)
                result['comparison_periods'] = [dates[0]] if dates else []
                result['start_date'] = dates[0][0] if dates else None
                result['end_date']   = dates[0][1] if dates else None
                result['query_clarity'] = 'clear'

            else:
                if dates:
                    result['start_date'] = dates[0][0]
                    result['end_date']   = dates[0][1]
                result['query_clarity'] = 'ambiguous'
                result['message'] = (
                    "⚠️  Comparison query is AMBIGUOUS:\n\n"
                    "You said 'compare' but provided only ONE period and ONE region.\n"
                    "A comparison requires TWO (or more) things to compare.\n\n"
                    "✅ Valid Options:\n\n"
                    "   Option 1: Multiple TIME PERIODS\n"
                    "   'Compare India in 2018, 2020 and 2022'\n"
                    "   'Compare India between 2020 and 2023'\n\n"
                    "   Option 2: Two DIFFERENT REGIONS\n"
                    "   'Compare Rajasthan vs Maharashtra between 2020 and 2023'\n\n"
                    "Please clarify what you want to compare."
                )

        # ----------------------------------------------------------------
        # NON-COMPARISON LOGIC
        # ----------------------------------------------------------------
        else:
            if dates:
                # Primary dates = first range; all_date_ranges has the rest
                result['start_date'] = dates[0][0]
                result['end_date']   = dates[0][1]
            else:
                result['query_clarity'] = 'unclear'
                result['message'] = "No dates found. Please specify a date or period."

        # Final validation
        if result['operation'] != 'comparison':
            if result['start_date'] is None or result['end_date'] is None:
                if result['query_clarity'] == 'clear':
                    result['query_clarity'] = 'unclear'

        if result['start_date'] and result['end_date']:
            if result['start_date'] > result['end_date']:
                result['start_date'], result['end_date'] = (
                    result['end_date'], result['start_date']
                )

        return result

    # ------------------------------------------------------------------ #
    # OPERATION DETECTION
    # ------------------------------------------------------------------ #

    def _detect_operation(self, q: str) -> str:
        """
        Detect the primary dataset operation from a query.

        Priority order (highest → lowest):
          1. Comparison  — explicit compare/versus keywords + multiple time/region refs
          2. Maximum     — wettest / maximum / highest / peak
          3. Minimum     — driest / minimum / lowest
          4. Mean        — mean / average
          5. Slope       — trend / slope / rate-of-change (ONLY when not in a
                           literature-framing context such as "rate of change
                           according to paper.pdf")
          6. Default     — mean

        This priority ensures that "maximum soil moisture ... and rate of change
        in vwc according to paper.pdf" → 'maximum', not 'slope'.
        """
        comparison_kw = ['compar', 'versus', ' vs ', ' vs.', 'differ', 'contrast']
        has_comp_kw   = any(kw in q for kw in comparison_kw)
        years         = re.findall(r'\b\d{4}\b', q)
        regions       = [r for r in VALID_REGIONS if r in q]

        if has_comp_kw and (len(years) >= 2 or len(regions) >= 2
                            or ' vs ' in q or ' vs.' in q or 'versus' in q):
            return 'comparison'

        # ── Check for publication framing that should suppress slope detection ──
        # If the query references a paper/pdf/publication, words like 'rate',
        # 'change', 'trajectory' are about the paper content, not a dataset op.
        _pub_frame = [
            'according to', '.pdf', 'in the paper', 'in the study',
            'from the paper', 'in the literature', 'published', 'findings',
        ]
        is_pub_framed = any(pf in q for pf in _pub_frame)

        # ── Strong explicit operation keywords — checked BEFORE slope ──────────
        # These take priority even when slope-like words also appear.
        
        # Check for state-level ranking queries first
        if re.search(r'\b(driest|least moisture)\s+(state|region|part)\b', q):
            return 'driest_state'
        if re.search(r'\b(wettest|most moisture|highest|maximum)\s+(state|region|part)\b', q):
            return 'wettest_state'
        maximum_kw = [
            r'\bmaximum\b', r'\bmax\b', r'\bhighest\b',
            r'\bwettest\b', r'\bmost moisture\b', r'\bpeak\b',
            r'\bwettest season\b', r'\bwettest month\b', r'\bwettest period\b',
        ]
        for kw in maximum_kw:
            if re.search(kw, q):
                return 'maximum'

        minimum_kw = [
            r'\bminimum\b', r'\bmin\b', r'\blowest\b',
            r'\bdriest\b', r'\bleast moisture\b',
            r'\bdriest season\b', r'\bdriest month\b', r'\bdriest period\b',
        ]
        for kw in minimum_kw:
            if re.search(kw, q):
                return 'minimum'

        mean_kw = ['mean', 'average', 'avg', 'typical', 'normal']
        for kw in mean_kw:
            if kw in q:
                return 'mean'

        # ── Slope keywords — only if NOT in a publication-framing context ──────
        # "rate of change in vwc according to paper.pdf" → skip slope
        slope_kw_safe = [
            'trend', 'slope', 'increasing', 'decreasing',
            'drying', 'wetting', 'over time', 'getting',
            'change over', 'temporal', 'time series', 'trajectory',
        ]
        slope_kw_pub_risky = ['rate', 'change']  # These words often appear in literature context

        for kw in slope_kw_safe:
            if kw in q:
                return 'slope'

        # Only trigger slope on 'rate'/'change' if NOT pub-framed
        if not is_pub_framed:
            for kw in slope_kw_pub_risky:
                if kw in q:
                    return 'slope'

        if re.search(r'\bplot\b|\btime.series\b|\btime series\b', q):
            return 'slope'

        for kw in comparison_kw:
            if kw in q:
                return 'comparison'

        return 'mean'

    # ------------------------------------------------------------------ #
    # OUTPUT TYPE DETECTION  (v2.6 — robust rewrite)
    # ------------------------------------------------------------------ #

    def _detect_output_type(self, q: str, operation: str = 'mean') -> str:
        """
        Determine whether the user wants 'scalar', 'map', or 'both'.

        Decision priority (highest → lowest):
          1. Explicit 'scalar only' language → 'scalar'
          2. Explicit 'both' language        → 'both'
          3. Explicit map/visual language    → 'map'  (returns 'both' so
             scalar stats are shown alongside the map)
          4. Default                         → 'both'

        The function errs on the side of 'both' so that maps are NEVER
        silently skipped for typical natural-language queries.
        """
        # ── 1. Explicit scalar-only ───────────────────────────────────
        # Only trigger when the user is clearly asking for a number/value only
        # and has NOT mentioned anything visual.
        scalar_only_phrases = [
            'only the value', 'only value', 'only number',
            'just the value', 'just value', 'just number',
            'only scalar', 'no map', 'without map',
            'scalar only', 'number only', 'value only',
            'text only', 'no visualization', 'no visualisation',
            'just the number', 'single value', 'only the result',
        ]
        for phrase in scalar_only_phrases:
            if phrase in q:
                return 'scalar'

        # ── 2. Explicit both ──────────────────────────────────────────
        both_phrases = [
            'both map and value', 'both value and map',
            'map and value', 'value and map',
            'map and number', 'number and map',
            'map and statistic', 'statistic and map',
            'include map', 'with map', 'show with map',
            'map along with', 'along with map',
        ]
        for phrase in both_phrases:
            if phrase in q:
                return 'both'

        # ── 3. Any visual / map keyword → return 'both' ───────────────
        # We return 'both' (not 'map') so scalars always accompany the map.
        # Use 'map' only when user says 'only map' / 'map only'.
        map_only_phrases = [
            'only map', 'map only', 'just map',
            'just the map', 'only the map',
        ]
        for phrase in map_only_phrases:
            if phrase in q:
                return 'map'

        # Generalized map-only checks to avoid strict phrasing constraints
        if re.search(r'\b(?:only|just)\s+(?:the\s+)?(?:spatial\s+)?map\b', q):
            return 'map'
        if re.search(r'\bno\s+(?:numbers?|values?|stats?|statistics?)\b', q):
            return 'map'
        if re.search(r'\brender\s+spatial\s+distribution\s+map\b', q):
            return 'map'

        map_keywords = [
            'map', 'spatial', 'visuali', 'geographic',
            'show', 'display', 'generate map', 'produce map',
            'plot', 'draw', 'render', 'image', 'distribution',
        ]
        for kw in map_keywords:
            if kw in q:
                return 'both'

        # ── 4. Scalar trigger (weak — only if NO map keywords present) ─
        # These are question-style queries that don't mention any visual.
        scalar_weak = [
            'what is', 'what was', 'what are',
            'tell me', 'give me the', 'report',
            'numerical', 'statistic',
        ]
        for kw in scalar_weak:
            if kw in q:
                # Still return 'both' for comparisons — always show map
                if operation == 'comparison':
                    return 'both'
                return 'scalar'

        # ── 5. Default ────────────────────────────────────────────────
        return 'both'

    # ------------------------------------------------------------------ #
    # REGION DETECTION  (v2.7 — no silent India default)
    # ------------------------------------------------------------------ #

    def _detect_all_regions(self, q: str) -> list:
        """
        Detect all valid Indian state/region names in the query.

        v2.7 change:
          • No longer appends 'india' as a fallback when nothing is found.
          • 'india' is only added when the word 'india', 'national', or
            'country' is explicitly present in the query.
          • If nothing is found, returns an empty list so the caller can
            set region_missing=True and prompt the user.
        """
        found = []
        for region in self._sorted_regions:
            if region in q and region not in found:
                found.append(region)

        # Preserve the original order of appearance in the user query
        if found:
            found.sort(key=lambda r: q.find(r))

        if not found:
            # Only default to India for explicit national-level keywords
            if 'india' in q or 'national' in q or 'country' in q:
                found.append('india')
            # Otherwise return empty — main.py will prompt the user

        return found

    # ------------------------------------------------------------------ #
    # COMPARISON HELPERS
    # ------------------------------------------------------------------ #

    def _detect_comparison_type(self, q: str, regions: list) -> str:
        return 'region' if len(regions) >= 2 else 'time'

    def _detect_comparison_metric(self, q: str) -> str:
        if re.search(r'\b(min|minimum|driest|lowest)\b', q):
            return 'min'
        if re.search(r'\b(max|maximum|wettest|highest|peak)\b', q):
            return 'max'
        if re.search(r'\b(slope|trend|rate)\b', q):
            return 'slope'
        return 'mean'

    # ------------------------------------------------------------------ #
    # COMPARISON DATE EXTRACTOR  (supports N periods)
    # ------------------------------------------------------------------ #

    def _extract_comparison_dates(self, q: str) -> list:
        """
        Extract N date ranges for comparison queries.
        Supports:
          • between YYYY and YYYY          → 2 annual ranges
          • YYYY vs YYYY vs YYYY           → N annual ranges
          • Month YYYY vs Month YYYY       → 2 month ranges
          • from YYYY to YYYY              → 2 annual ranges
          • comma/and-separated years:     2018, 2020 and 2022 → 3 ranges
        """

        # ── Pattern A: between YYYY and YYYY ──────────────────────────
        between = re.search(r'\bbetween\s+(\d{4})\s+and\s+(\d{4})\b', q)
        if between:
            y1, y2 = int(between.group(1)), int(between.group(2))
            if 1990 <= y1 <= 2100 and 1990 <= y2 <= 2100 and y1 != y2:
                return [_year_range(min(y1, y2)), _year_range(max(y1, y2))]

        # ── Pattern B: YYYY vs YYYY (vs YYYY …) ───────────────────────
        vs_chain = re.findall(r'\b(\d{4})\s+(?:vs\.?|versus)\s+(\d{4})\b', q)
        if vs_chain:
            years_seen = []
            for pair in vs_chain:
                for y in pair:
                    yi = int(y)
                    if 1990 <= yi <= 2100 and yi not in years_seen:
                        years_seen.append(yi)
            if len(years_seen) >= 2:
                return [_year_range(y) for y in years_seen]

        # ── Pattern C: Month YYYY vs Month YYYY ───────────────────────
        month_pat = r'(?:' + '|'.join(MONTH_MAP.keys()) + r')'
        mv_mv = re.search(
            rf'\b({month_pat})\s+(\d{{4}})\s+(?:vs\.?|versus|and)\s+'
            rf'({month_pat})\s+(\d{{4}})\b',
            q
        )
        if mv_mv:
            m1, y1 = MONTH_MAP[mv_mv.group(1)], int(mv_mv.group(2))
            m2, y2 = MONTH_MAP[mv_mv.group(3)], int(mv_mv.group(4))
            return [_month_range(m1, y1), _month_range(m2, y2)]
            
        # ── Pattern C2/C3: N months with a shared year ───────────────
        # Handles ANY of:
        #   "June and July 2020"
        #   "June and July in 2020"
        #   "June vs July in 2020"
        #   "June, July and August in 2021"  (3 months)
        # Strategy: find all month names in q, find a shared 4-digit year,
        # and only fire if 2+ months are found — this handles 2 or 3+ months.
        _all_months_found = re.findall(rf'\b({month_pat})\b', q)
        _year_found = re.search(r'\b(\d{4})\b', q)
        if len(_all_months_found) >= 2 and _year_found:
            _shared_year = int(_year_found.group(1))
            if 1990 <= _shared_year <= 2100:
                # Deduplicate while preserving order
                _seen = []
                for _m in _all_months_found:
                    if _m not in _seen:
                        _seen.append(_m)
                return [_month_range(MONTH_MAP[_m], _shared_year) for _m in _seen]

        # ── Pattern D: from/over YYYY to YYYY ──────────────────────────
        from_to = re.search(r'\b(?:from\s+|over\s+)?(\d{4})\s+to\s+(\d{4})\b', q)
        if from_to:
            y1, y2 = int(from_to.group(1)), int(from_to.group(2))
            if 1990 <= y1 <= 2100 and 1990 <= y2 <= 2100 and y1 != y2:
                return [_year_range(y1), _year_range(y2)]

        # ── Pattern E: comma/and-separated years ──────────────────────
        multi_year = self._extract_listed_years(q)
        if len(multi_year) >= 2:
            # Check if a season is mentioned in the query
            active_season = None
            for season in LOCAL_SEASONS.keys():
                if re.search(rf'(?<![\w-])\b{season}\b', q):
                    active_season = season
                    break
            
            if active_season:
                cfg = LOCAL_SEASONS[active_season]
                months = cfg['months']
                season_ranges = []
                for y in multi_year:
                    if months[0] > months[-1]:
                        s = f"{y}-{months[0]:02d}-01"
                        _, e = _month_range(months[-1], y + 1)
                    else:
                        s, _ = _month_range(months[0], y)
                        _, e = _month_range(months[-1], y)
                    season_ranges.append((s, e))
                return season_ranges

            return [_year_range(y) for y in multi_year]

        # ── Pattern D2: Quarters in comparison: "Q1 2021 vs Q3 2021" ──────
        _QUARTERS_C = {1: [1,2,3], 2: [4,5,6], 3: [7,8,9], 4: [10,11,12]}
        _quarter_matches = list(re.finditer(r'\bq([1-4])\s+(\d{4})\b', q, re.IGNORECASE))
        if len(_quarter_matches) >= 2:
            _qranges = []
            for _qm in _quarter_matches:
                _qnum, _qyr = int(_qm.group(1)), int(_qm.group(2))
                if 1990 <= _qyr <= 2100:
                    _qmons = _QUARTERS_C[_qnum]
                    _sq, _ = _month_range(_qmons[0],  _qyr)
                    _,  _eq = _month_range(_qmons[-1], _qyr)
                    _qranges.append((_sq, _eq))
            if len(_qranges) >= 2:
                return _qranges

        # ── Fallback: general date ranges ───────────────────────────────
        return self._extract_all_date_ranges(q)

    def _extract_listed_years(self, q: str) -> list:
        """
        Extract a list of explicitly enumerated years, e.g.:
          "in 2007 and 2019"         → [2007, 2019]
          "in 2018, 2020 and 2022"   → [2018, 2020, 2022]
          "for 2015, 2018, 2021"     → [2015, 2018, 2021]

        Carefully avoids merging "from/between YYYY and YYYY" constructs.
        """
        cleaned = re.sub(r'\bfrom\s+\d{4}\s+to\s+\d{4}\b', '', q)
        cleaned = re.sub(r'\bbetween\s+\d{4}\s+and\s+\d{4}\b', '', cleaned)

        years = []
        all_years = re.findall(r'\b(\d{4})\b', cleaned)
        for y_str in all_years:
            y = int(y_str)
            if 1990 <= y <= 2100 and y not in years:
                years.append(y)

        return years

    def _parse_numeric_date(self, date_str: str) -> str:
        """
        Parse a numeric date string like DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY.
        Returns a YYYY-MM-DD string, or None if unparseable.
        Defaults to DD-MM-YYYY when both v1 and v2 are <= 12 (ambiguous).
        """
        import calendar
        m = re.match(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})$', date_str.strip())
        if not m:
            return None

        v1, v2, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1990 <= year <= 2100):
            return None
            
        def _clamp_and_format(y, month, day):
            if month < 1 or month > 12:
                raise ValueError
            import calendar
            from datetime import datetime
            _, max_d = calendar.monthrange(y, month)
            day = min(day, max_d)
            return datetime(y, month, day).strftime('%Y-%m-%d')
            
        if v1 > 12 and v2 <= 12:           # must be DD-MM-YYYY
            try:
                return _clamp_and_format(year, v2, v1)
            except ValueError:
                return None
        elif v2 > 12 and v1 <= 12:         # must be MM-DD-YYYY
            try:
                return _clamp_and_format(year, v1, v2)
            except ValueError:
                return None
        else:                               # ambiguous → default DD-MM-YYYY
            try:
                return _clamp_and_format(year, v2, v1)
            except ValueError:
                try:
                    return _clamp_and_format(year, v1, v2)
                except ValueError:
                    return None

    def _extract_all_date_ranges(self, q: str) -> list:
        """
        Extract date ranges from a query string.

        KEY CHANGE v2.4:
          • "YYYY and YYYY" (without from/between) → TWO separate year ranges
          • "from YYYY to YYYY" / "between YYYY and YYYY" → ONE span range
          • All other patterns unchanged.

        KEY CHANGE v2.8 (date-range bug fix):
          • "from DD/MM/YYYY to DD/MM/YYYY" now forms a proper range (early exit).
          • "from YYYY-MM-DD to YYYY-MM-DD" no longer re-parses individual ISO
            dates in section 3.1, eliminating spurious (date, date) singles.
          • Early returns added after sections 5, 6, 4/4c so that downstream
            month/year sections don't append extra ranges when a range is
            already found.
        """
        import calendar
        ranges = []
        
        # ── 0. Catch "latest", "present", "now" ──
        q = re.sub(r'\b(latest(\s+data)?|present|now)\b', '2099', q, flags=re.IGNORECASE)
        month_pat   = r'(?:' + '|'.join(MONTH_MAP.keys()) + r')'
        ordinal_pat = r'(?:\d{1,2}(?:st|nd|rd|th)?)'

        # ── 1. "between YYYY and YYYY"  → single SPAN range ───────────
        for m in re.finditer(r'\bbetween\s+(\d{4})\s+and\s+(\d{4})\b', q):
            y1, y2 = int(m.group(1)), int(m.group(2))
            if 1990 <= y1 <= 2100 and 1990 <= y2 <= 2100:
                s = f"{min(y1,y2)}-01-01"
                e = f"{max(y1,y2)}-12-31"
                ranges.append((s, e))
                return ranges          # span found → done

        # ── 2. "YYYY to YYYY"  → single SPAN range ──────────
        consumed_span_years = set()
        for m in re.finditer(r'\b(?:from\s+|over\s+)?(\d{4})\s+to\s+(\d{4})\b', q):
            y1, y2 = int(m.group(1)), int(m.group(2))
            if 1990 <= y1 <= 2100 and 1990 <= y2 <= 2100 and y1 != y2:
                s = f"{min(y1,y2)}-01-01"
                e = f"{max(y1,y2)}-12-31"
                ranges.append((s, e))
                consumed_span_years.update({y1, y2})
        if consumed_span_years:
            return ranges

        # ── 2b. "from/between EXACT_DATE to/and EXACT_DATE" ──────────
        # Handles ISO (YYYY-MM-DD) and numeric (DD/MM/YYYY, MM-DD-YYYY) forms.
        # Must run BEFORE section 3/3.1 so we can return early and avoid
        # individual dates being emitted as spurious (d, d) singles.

        # ISO: "from 2020-01-01 to 2020-03-31"
        m_iso2 = re.search(
            r'\b(?:from\s+|between\s+)?(\d{4}-\d{2}-\d{2})\s+(?:to|and)\s+(\d{4}-\d{2}-\d{2})\b', q
        )
        if m_iso2:
            d1, d2 = m_iso2.group(1), m_iso2.group(2)
            try:
                datetime.strptime(d1, '%Y-%m-%d')
                datetime.strptime(d2, '%Y-%m-%d')
                s, e = (d1, d2) if d1 <= d2 else (d2, d1)
                return [(s, e)]
            except ValueError:
                pass

        # Numeric: "from 01/01/2020 to 31/03/2020" or "from 01-01-2020 to 31-03-2020"
        _num_date_pat = r'\d{1,2}[-/]\d{1,2}[-/]\d{4}'
        m_num2 = re.search(
            rf'\b(?:from\s+|between\s+)?({_num_date_pat})\s+(?:to|and)\s+({_num_date_pat})\b', q
        )
        if m_num2:
            d1 = self._parse_numeric_date(m_num2.group(1))
            d2 = self._parse_numeric_date(m_num2.group(2))
            if d1 and d2:
                s, e = (d1, d2) if d1 <= d2 else (d2, d1)
                return [(s, e)]

        # ── 3. ISO dates (single or unpairable) ───────────────────────
        iso_dates = []
        for m in re.findall(r'\b(\d{4}-\d{2}-\d{2})\b', q):
            try:
                datetime.strptime(m, '%Y-%m-%d')
                iso_dates.append(m)
            except ValueError:
                pass
        if len(iso_dates) >= 2:
            for i in range(0, len(iso_dates) - 1, 2):
                ranges.append((iso_dates[i], iso_dates[i+1]))
            if len(iso_dates) % 2 == 1:
                ranges.append((iso_dates[-1], iso_dates[-1]))
            return ranges          # ISO range found → skip numeric re-parse
        elif len(iso_dates) == 1:
            ranges.append((iso_dates[0], iso_dates[0]))
            return ranges

        # ── 3.1. Standalone numeric dates DD/MM/YYYY or MM/DD/YYYY ───
        # Only reached when NO ISO dates are present (avoids double-counting).
        # Pairs them in order; does NOT re-parse YYYY-MM-DD (already handled).
        num_dates_parsed = []
        for m in re.finditer(r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b', q):
            d_str = self._parse_numeric_date(m.group(0))
            if d_str and d_str not in num_dates_parsed:
                num_dates_parsed.append(d_str)

        # YYYY/MM/DD with slash only (dash form is ISO and handled above)
        for m in re.finditer(r'\b(\d{4})/(\d{1,2})/(\d{1,2})\b', q):
            year, v1, v2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1990 <= year <= 2100:
                try:
                    d_str = datetime(year, v1, v2).strftime('%Y-%m-%d')
                except ValueError:
                    try:
                        d_str = datetime(year, v2, v1).strftime('%Y-%m-%d')
                    except ValueError:
                        continue
                if d_str not in num_dates_parsed:
                    num_dates_parsed.append(d_str)

        if len(num_dates_parsed) >= 2:
            for i in range(0, len(num_dates_parsed) - 1, 2):
                ranges.append((num_dates_parsed[i], num_dates_parsed[i+1]))
            if len(num_dates_parsed) % 2 == 1:
                ranges.append((num_dates_parsed[-1], num_dates_parsed[-1]))
            return ranges
        elif len(num_dates_parsed) == 1:
            ranges.append((num_dates_parsed[0], num_dates_parsed[0]))
            return ranges

        # ── 3b. Last X days/months/years ─────────────────────────────
        for m in re.finditer(r'\b(?:last\s+)?(\d+)\s+(day|month|year)s?\b', q):
            num = int(m.group(1))
            unit = m.group(2)
            end_date = datetime(2023, 12, 31)
            if unit == 'day':
                start_date = end_date - timedelta(days=num - 1)
            elif unit == 'month':
                start_date = end_date - timedelta(days=num * 30 - 1)
            else:  # year
                start_date = end_date - timedelta(days=num * 365 - 1)
            ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            return ranges

        # ── 4. Month Day Year  AND  4c. Day Month Year ────────────────
        md_y_pat  = rf'\b({month_pat})\s+({ordinal_pat}),?\s*(\d{{4}})\b'
        d_m_y_pat = rf'\b({ordinal_pat}),?\s*({month_pat}),?\s*(\d{{4}})\b'
        mdy_dates = []
        for m in re.finditer(md_y_pat, q):
            month_n = MONTH_MAP[m.group(1)]
            day     = int(re.sub(r'\D', '', m.group(2)))
            year    = int(m.group(3))
            try:
                d = datetime(year, month_n, day).strftime('%Y-%m-%d')
                mdy_dates.append(d)
            except ValueError:
                pass

        for m in re.finditer(d_m_y_pat, q):
            day     = int(re.sub(r'\D', '', m.group(1)))
            month_n = MONTH_MAP[m.group(2)]
            year    = int(m.group(3))
            try:
                d = datetime(year, month_n, day).strftime('%Y-%m-%d')
                mdy_dates.append(d)
            except ValueError:
                pass

        # Sort and deduplicate extracted exact dates before pairing
        mdy_dates = sorted(list(set(mdy_dates)))

        if len(mdy_dates) >= 2:
            for i in range(0, len(mdy_dates) - 1, 2):
                ranges.append((mdy_dates[i], mdy_dates[i+1]))
            if len(mdy_dates) % 2 == 1:
                ranges.append((mdy_dates[-1], mdy_dates[-1]))
            return ranges          # exact-date range found → done
        elif len(mdy_dates) == 1:
            ranges.append((mdy_dates[0], mdy_dates[0]))
            return ranges

        # ── 4b. Month Day to Day, Year ────────────────────────────────
        md_to_d_y_pat = rf'\b({month_pat})\s+({ordinal_pat})\s+(?:to|and|-)\s+({ordinal_pat}),?\s+(\d{{4}})\b'
        for m in re.finditer(md_to_d_y_pat, q):
            month_n = MONTH_MAP[m.group(1)]
            day1    = int(re.sub(r'\D', '', m.group(2)))
            day2    = int(re.sub(r'\D', '', m.group(3)))
            year    = int(m.group(4))
            try:
                d1 = datetime(year, month_n, day1).strftime('%Y-%m-%d')
                d2 = datetime(year, month_n, day2).strftime('%Y-%m-%d')
                ranges.append((d1, d2))
            except ValueError:
                pass
        if ranges:
            return ranges

        # ── 5. Month Year to Month Year ───────────────────────────────
        for m in re.finditer(
            rf'\b(?:from\s+|between\s+)?({month_pat})\s+(\d{{4}})\s+(?:to|and)\s+({month_pat})\s+(\d{{4}})\b', q
        ):
            m1, y1 = MONTH_MAP[m.group(1)], int(m.group(2))
            m2, y2 = MONTH_MAP[m.group(3)], int(m.group(4))
            s, _   = _month_range(m1, y1)
            _, e   = _month_range(m2, y2)
            ranges.append((s, e))
        if ranges:
            return ranges          # month-year span found → done

        # ── 6. Month to Month Year ────────────────────────────────────
        for m in re.finditer(
            rf'\b(?:from\s+|between\s+)?({month_pat})\s+(?:to|and)\s+({month_pat})\s+(\d{{4}})\b', q
        ):
            m1, m2 = MONTH_MAP[m.group(1)], MONTH_MAP[m.group(2)]
            year   = int(m.group(3))
            s, _   = _month_range(m1, year)
            _, e   = _month_range(m2, year)
            ranges.append((s, e))
        if ranges:
            return ranges          # month-to-month span found → done

        # Ensure month_year_matches is always defined for section 9
        month_year_matches = []

        # ── 6b. "Month and Month in YEAR" → two separate monthly ranges ──
        # e.g. "compare june and july in 2020" →
        #       [(2020-06-01,2020-06-30), (2020-07-01,2020-07-31)]
        # Handles 2 OR 3 comma/and-joined months sharing a single year.
        _6b_pat = (
            r'\b(' + '|'.join(MONTH_MAP.keys()) + r')'
            r'(?:\s*,\s*|\s+and\s+)'
            r'(' + '|'.join(MONTH_MAP.keys()) + r')'
            r'(?:(?:\s*,\s*|\s+and\s+)(' + '|'.join(MONTH_MAP.keys()) + r'))?'
            r'\s+(?:in\s+|of\s+|,?\s*)(\d{4})\b'
        )
        _6b_match = re.search(_6b_pat, q)
        if _6b_match:
            _yr_6b = int(_6b_match.group(4))
            if 1990 <= _yr_6b <= 2100:
                _months_6b = [
                    MONTH_MAP[_6b_match.group(1)],
                    MONTH_MAP[_6b_match.group(2)],
                ]
                if _6b_match.group(3):
                    _months_6b.append(MONTH_MAP[_6b_match.group(3)])
                for _mn in _months_6b:
                    _s6b, _e6b = _month_range(_mn, _yr_6b)
                    if (_s6b, _e6b) not in ranges:
                        ranges.append((_s6b, _e6b))
                if ranges:
                    return ranges

        # ── 6c. Year-first month: "2020 june" / "2021 march" ─────────────
        _6c_pat = r'\b(\d{4})[\s/-]+(' + '|'.join(MONTH_MAP.keys()) + r')\b'
        if not ranges:
            for _m6c in re.finditer(_6c_pat, q):
                _yr_6c = int(_m6c.group(1))
                if 1990 <= _yr_6c <= 2100:
                    _mn_6c = MONTH_MAP[_m6c.group(2)]
                    _s6c, _e6c = _month_range(_mn_6c, _yr_6c)
                    if (_s6c, _e6c) not in ranges:
                        ranges.append((_s6c, _e6c))
            if ranges:
                return ranges

        # ── 6d. ISO year-month without day: "2020-06" or "2020/06" ────────
        if not ranges:
            for _m6d in re.finditer(r'\b(\d{4})[-/](0[1-9]|1[0-2])\b', q):
                _yr_6d  = int(_m6d.group(1))
                _mn_6d  = int(_m6d.group(2))
                if 1990 <= _yr_6d <= 2100:
                    _s6d, _e6d = _month_range(_mn_6d, _yr_6d)
                    if (_s6d, _e6d) not in ranges:
                        ranges.append((_s6d, _e6d))
            if ranges:
                return ranges

        # ── 6e. Meteorological / fiscal quarters: "Q1 2021", "Q3 2022" ───
        _QUARTERS = {1: [1,2,3], 2: [4,5,6], 3: [7,8,9], 4: [10,11,12]}
        if not ranges:
            for _m6e in re.finditer(r'\bq([1-4])\s+(\d{4})\b', q, re.IGNORECASE):
                _qnum_e  = int(_m6e.group(1))
                _yr_6e   = int(_m6e.group(2))
                if 1990 <= _yr_6e <= 2100:
                    _qm_e        = _QUARTERS[_qnum_e]
                    _sq6e, _     = _month_range(_qm_e[0],  _yr_6e)
                    _,     _eq6e = _month_range(_qm_e[-1], _yr_6e)
                    if (_sq6e, _eq6e) not in ranges:
                        ranges.append((_sq6e, _eq6e))
            if ranges:
                return ranges


        # ── 7. Month Year ─────────────────────────────────────────────
        for m in re.finditer(rf'\b({month_pat})\s+(\d{{4}})\b', q):
            month_n = MONTH_MAP[m.group(1)]
            year    = int(m.group(2))
            s, e    = _month_range(month_n, year)
            ranges.append((s, e))
            month_year_matches.append(year)

        # ── 8. Seasons ────────────────────────────────────────────────
        for season, cfg in LOCAL_SEASONS.items():
            m = re.search(rf'(?<![\w-])\b{season}\s+(\d{{4}})\b', q)
            if m:
                year   = int(m.group(1))
                months = cfg['months']
                if months[0] > months[-1]:
                    s    = f"{year}-{months[0]:02d}-01"
                    _, e = _month_range(months[-1], year + 1)
                else:
                    s, _ = _month_range(months[0], year)
                    _, e = _month_range(months[-1], year)
                ranges.append((s, e))

        # ── 8b. Meteorological acronyms: JJAS / MAM / DJF / ON etc. ──────
        # Matches patterns like: "JJAS 2020", "MAM 2021", "DJF 2019"
        # Also handles bare acronym (no year) — treated as 'no date' so
        # app.py can ask or fall back.
        if not ranges:
            for acronym, months in _SEASON_ACRONYMS.items():
                acr_pat = rf'\b{re.escape(acronym)}\s+(\d{{4}})\b'
                for acr_m in re.finditer(acr_pat, q):
                    year = int(acr_m.group(1))
                    if 1990 <= year <= 2100:
                        # Handle year-wrapping seasons (e.g. DJF)
                        if months[0] > months[-1]:
                            s    = f"{year}-{months[0]:02d}-01"
                            _, e = _month_range(months[-1], year + 1)
                        else:
                            s, _ = _month_range(months[0], year)
                            _, e = _month_range(months[-1], year)
                        if (s, e) not in ranges:
                            ranges.append((s, e))

        # ── 9. Bare years  (each gets its own independent range) ───────
        active_season = None
        for season in LOCAL_SEASONS.keys():
            if re.search(rf'(?<![\w-])\b{season}\b', q):
                active_season = season
                break

        covered_years = set(month_year_matches)
        annual_pat = (
            r'\b(?:annual|year|yearly|full\s+year)?'
            r'\s*(\d{4})\b'
        )
        for m in re.finditer(annual_pat, q):
            year = int(m.group(1))
            if 1990 <= year <= 2100 and year not in covered_years:
                covered_years.add(year)
                already = any(r[0].startswith(str(year)) for r in ranges)
                if not already:
                    if active_season:
                        cfg = LOCAL_SEASONS[active_season]
                        months = cfg['months']
                        if months[0] > months[-1]:
                            s = f"{year}-{months[0]:02d}-01"
                            _, e = _month_range(months[-1], year + 1)
                        else:
                            s, _ = _month_range(months[0], year)
                            _, e = _month_range(months[-1], year)
                        ranges.append((s, e))
                    else:
                        s, e = _year_range(year)
                        ranges.append((s, e))

        # ── 10. beginning / end of YYYY ───────────────────────────────
        for m in re.finditer(r'\b(?:beginning|start)\s+of\s+(\d{4})\b', q):
            year = int(m.group(1))
            ranges.append(_month_range(1, year))
        for m in re.finditer(r'\bend\s+of\s+(\d{4})\b', q):
            year = int(m.group(1))
            ranges.append(_month_range(12, year))

        # ── Deduplicate (preserve order) ──────────────────────────────
        seen, unique = set(), []
        for r in ranges:
            if r not in seen:
                seen.add(r)
                unique.append(r)

        return unique

    # ------------------------------------------------------------------ #
    # DESCRIBE CLASSIFICATION
    # ------------------------------------------------------------------ #

    def describe(self, result: dict) -> str:
        region_display = result['region'].title() if result['region'] else '⚠️  NOT DETECTED'
        lines = [
            f"  • Region:      {region_display}",
            f"  • Operation:   {result['operation'].upper()}",
            f"  • Output:      {result['output_type'].upper()}",
        ]

        all_ranges = result.get('all_date_ranges', [])
        if len(all_ranges) > 1 and result['operation'] != 'comparison':
            lines.append(f"  • Periods:     {len(all_ranges)} date ranges detected")
            for i, (s, e) in enumerate(all_ranges, 1):
                lines.append(f"    [{i}] {s} to {e}")
        else:
            lines.append(
                f"  • Period:      {result['start_date']} to {result['end_date']}"
            )

        if result['operation'] == 'comparison':
            lines.append(f"  • Comp. type:  {result['comparison_type']}")
            lines.append(f"  • Comp. metric:{result['comparison_metric']}")
            if result['comparison_type'] == 'region':
                lines.append(
                    f"  • Region 2:    "
                    f"{result.get('comparison_region2', '?').title()}"
                )
            else:
                periods = result.get('comparison_periods', [])
                for i, p in enumerate(periods, 1):
                    lines.append(f"  • Period {i}:    {p[0]} to {p[1]}")

        lines.append(f"  • Clarity:     {result['query_clarity']}")
        if result.get('region_missing'):
            lines.append(f"  • ⚠️  Region:   NOT DETECTED — user will be prompted")
        return "\n".join(lines)