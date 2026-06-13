"""
intent_classifier.py
====================

Classifies a user query as one of:
    "dataset"    - needs soil moisture data analysis
    "literature" - needs scientific paper Q&A
    "both"       - needs BOTH dataset analysis AND literature Q&A
    "chat"       - general conversation / greetings

Strategy (fast → smart):
  1. Trivial pre-filter: obvious greetings → chat
  2. Unambiguous keyword shortcuts: e.g. only a region+year with no lit words → dataset
  3. Ollama LLM for everything else (single word output, ~8 tokens)

The old approach used large keyword lists that grew brittle. Now the LLM
handles all the nuanced cases — it's already running locally, adds ~1s latency
which is acceptable, and is far more robust.
"""

import re
import requests


# ============================================================================
# MINIMAL FAST PRE-FILTERS
# (Only handle cases where we are 100% confident without the LLM)
# ============================================================================

# Words that are ONLY meaningful in a literature/paper context
_STRONG_LIT = {
    'algorithm', 'retrieval', 'methodology', 'lprm', 'tau-omega',
    'backscatter', 'dielectric', 'emissivity', 'amsr', 'smos', 'smap',
    'sentinel', 'passive microwave', 'active microwave', 'optical depth',
    'rmse', 'bias', 'validation', 'calibration', 'ieee', 'proceedings',
    'doi', 'published', 'journal', 'article', '.pdf', 'according to',
    # Figure / table references always point to a paper/document
    'figure', 'fig.', 'fig ', 'table ', 'tab.', 'tab ',
    'in the paper', 'in the study', 'in the literature', 'the paper says',
    'the study says', 'literature says', 'literature on',
}

# Words that unambiguously indicate a dataset action
_STRONG_DATA = {
    'mean', 'average', 'minimum', 'maximum', 'trend', 'slope',
    'compare', 'comparison',
}

# Regions list (loaded lazily)
def _get_regions():
    try:
        from Config import VALID_REGIONS
        return set(VALID_REGIONS) | {'india', 'national', 'country'}
    except Exception:
        return {'india'}

_YEAR_RE = re.compile(r'\b20\d{2}\b')

_GREETINGS = {
    'hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'goodbye',
    'how are you', 'who are you', 'what can you do', 'help',
}


def _fast_classify(query: str):
    """
    Returns a classification string OR None (= needs LLM).

    Rules (all must be high-confidence):
      - Short greeting → 'chat'
      - Has strong lit word + strong data word together → 'both'
      - Has strong lit word + NO strong data word + NO region+year → 'literature'
      - Has region + year + strong data word + NO strong lit → 'dataset'
      - Everything else → None (LLM decides)
    """
    q    = query.lower().strip()
    words = set(q.split())

    # Greeting check
    if len(words) <= 5 and (q in _GREETINGS or any(q.startswith(g) for g in _GREETINGS)):
        return 'chat'

    regions = _get_regions()

    has_strong_lit  = any(kw in q for kw in _STRONG_LIT)
    has_strong_data = any(kw in q for kw in _STRONG_DATA)
    has_region      = any(r in q for r in regions)
    has_year        = bool(_YEAR_RE.search(q))

    # Both: strong lit signals AND strong data signals present together
    # (works regardless of which comes first in the query)
    if has_strong_lit and has_strong_data:
        return 'both'

    # Both: region+year (dataset signal) AND strong lit signal
    if has_strong_lit and (has_region and has_year):
        return 'both'

    # Clear literature-only: strong lit signals, no dataset geography/action
    if has_strong_lit and not has_strong_data and not (has_region and has_year):
        return 'literature'

    # Clear dataset-only: region + year + data action, no lit signals
    if has_region and has_year and has_strong_data and not has_strong_lit:
        return 'dataset'

    # Let the LLM handle everything else
    return None


# ============================================================================
# LLM CLASSIFIER  (handles ambiguous, mixed, and nuanced queries)
# ============================================================================

_LLM_PROMPT = (
    "You are a routing classifier for a soil moisture analysis system.\n"
    "\n"
    "Classify the user query into EXACTLY ONE of these four categories:\n"
    "\n"
    "  chat\n"
    "    - Greetings, small talk, 'who are you', 'what can you do', off-topic.\n"
    "\n"
    "  dataset\n"
    "    - Wants soil moisture statistics, maps, trends, comparisons, regional\n"
    "      analysis, numerical data for a specific region and/or time period.\n"
    "    - Examples:\n"
    "        'What is the mean soil moisture in Rajasthan in June 2020?'\n"
    "        'Compare Rajasthan and Gujarat in 2021'\n"
    "        'Show moisture trend in Punjab monsoon 2022'\n"
    "\n"
    "  literature\n"
    "    - Wants scientific explanation, methodology, research findings, sensor\n"
    "      details, or information from a paper, journal, PDF, or research study.\n"
    "    - ANY reference to a figure, table, chart, plot, or diagram in a document.\n"
    "    - Examples:\n"
    "        'Explain the AMSR2 retrieval algorithm'\n"
    "        'What RMSE was reported for SMAP validation?'\n"
    "        'According to the paper, what is the LPRM method?'\n"
    "        'Show me figure 1'\n"
    "        'Show me table 1'\n"
    "        'What does figure 3 show?'\n"
    "        'From the scatter plot in the IEEE paper, which station...'\n"
    "\n"
    "  both\n"
    "    - The query asks for BOTH a dataset result AND a literature explanation\n"
    "      in the SAME question, regardless of which part comes first.\n"
    "    - Examples (dataset part first):\n"
    "        'Show moisture data for Rajasthan 2020 and explain what the literature says about AMSR2'\n"
    "        'What was the maximum soil moisture in India in 2020, and how does the paper explain this?'\n"
    "        'Give mean moisture for India in 2021 and explain the retrieval methodology from the paper'\n"
    "    - Examples (literature part first):\n"
    "        'Explain what the paper says about LPRM and show me the moisture trend for Punjab 2022'\n"
    "        'According to the study, what causes high soil moisture, and show data for Kerala monsoon 2021?'\n"
    "        'What does the literature say about AMSR2 bias and show the mean moisture in India 2020?'\n"
    "    - Do NOT use 'both' for:\n"
    "        'compare both files'          -> dataset\n"
    "        'both Rajasthan and Gujarat'  -> dataset\n"
    "        'summarise both papers'       -> literature\n"
    "\n"
    "Respond with ONLY ONE WORD - no punctuation, no explanation:\n"
    "chat | dataset | literature | both\n"
    "\n"
    "User query: {query}\n"
    "\n"
    "Category:"
)




def _llm_classify(
    query: str,
    ollama_url: str,
    ollama_model: str,
    timeout: int,
) -> str:
    """Call Ollama to classify the query intent."""
    try:
        resp = requests.post(
            ollama_url,
            json={
                "model" : ollama_model,
                "prompt": _LLM_PROMPT.format(query=query),
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_p"      : 0.1,
                    "num_ctx"    : 2048,
                    "num_predict": 8,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip().lower()
        raw_clean = re.sub(r"[^a-z\s]", "", raw).strip()
        print("LLM CLASSIFIER OUTPUT:", raw_clean)

        # Accept exact match first
        if raw_clean in ("chat", "dataset", "literature", "both"):
            return raw_clean
        # LLM sometimes outputs two words joined or with extra text — 
        # pick the FIRST valid intent word found
        for token in ("both", "literature", "dataset", "chat"):
            if token in raw_clean:
                print(f"  (parsed token: {token})")
                return token

    except Exception as e:
        print("LLM CLASSIFIER ERROR:", e)

    return "chat"   # safe fallback


# ============================================================================
# PUBLIC API
# ============================================================================

def classify_query_intent(
    query: str,
    ollama_url: str   = "http://localhost:11434/api/generate",
    ollama_model: str = "qwen2.5:3b",
    timeout: int      = 30,
) -> str:
    """
    Classify a user query intent.

    Returns: "dataset" | "literature" | "both" | "chat"
    """
    fast = _fast_classify(query)
    if fast is not None:
        print("FINAL INTENT (FAST):", fast)
        return fast

    result = _llm_classify(
        query        = query,
        ollama_url   = ollama_url,
        ollama_model = ollama_model,
        timeout      = timeout,
    )
    print("FINAL INTENT (LLM):", result)
    return result


# ============================================================================
# DEBUG HELPER
# ============================================================================

def explain_classification(query: str, intent: str) -> str:
    q = query.lower()
    if intent == "both":
        return "🔀 BOTH — Query requests dataset analysis AND literature explanation."
    if intent == "literature":
        hits = [kw for kw in _STRONG_LIT if kw in q]
        return f"📚 LITERATURE — Scientific/paper query. Signals: {hits[:4]}"
    if intent == "dataset":
        hits = [kw for kw in _STRONG_DATA if kw in q]
        return f"📊 DATASET — Soil moisture data/statistics query. Signals: {hits[:4]}"
    return "💬 CHAT — General conversation."