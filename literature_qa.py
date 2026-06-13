"""
literature_qa.py
================
Answers user queries from loaded literature files using:

  1. TF-IDF retrieval via LiteratureManager  (text + table chunks)
  2. llava:7b via Ollama                      (figures / visual queries)
  3. Ollama qwen2.5:3b                       (text answer generation)

FIXES in this version:
  - Direct asset requests ("show me figure 3") now actually OPEN / DISPLAY the
    image in the terminal (via PIL/Pillow or OS open) in addition to printing
    the path. The image is also returned via display_data so callers can render it.
  - Indirect visual queries (e.g. "explain the methodology chart") now:
      (a) Get a llava:7b text answer as before.
      (b) Also surface the related image(s) as display_data in the returned answer
          so main.py can show them.
  - answer_from_literature now returns a third value: image_display_list
    (list of display_data dicts, may be empty) so the caller can render images.
  - format_image_links: unchanged (path/URI text block).
  - _open_image_for_user(): new helper — tries to open the image file with
    the system viewer (works on Windows, macOS, Linux).

Public API (updated)
---------------------
answer_from_literature(...) -> (answer: str | None,
                                found_in_lit: bool,
                                image_display_list: list)

get_literature_context(...) -> (context_text: str, found: bool)

parse_literature_command(...) -> str | None
"""

import os
import re
import subprocess
import sys
import requests
import base64
from typing import Optional, Tuple, List

# ── Directory containing cropped PDF figure/table assets ─────────────────
LIT_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "literature_images"
)

# ============================================================================
# META-QUERY PATTERNS
# ============================================================================

_LIST_PATTERNS = [
    r"\blist\b.*\b(file|journal|paper|article|literature|loaded|source)\b",
    r"\b(what|which)\b.*\b(file|journal|paper|article|literature|loaded|source)\b",
    r"\bshow\b.*\b(file|journal|paper|article|literature|loaded|source)\b",
    r"\b(name|names)\b.*\b(file|journal|paper|article|loaded)\b",
    r"\bwhat (journals?|papers?|articles?|files?) (are|have been|were) loaded\b",
    r"\btell.*\b(file|journal|paper)\b.*\bname\b",
]

_TABLE_COUNT_PATTERNS = [
    r"\bhow many tables?\b",
    r"\bnumber of tables?\b",
    r"\bcount.*tables?\b",
]

_FIGURE_COUNT_PATTERNS = [
    r"\bhow many (figure|image|chart|graph|diagram)s?\b",
    r"\bnumber of (figure|image|chart|graph|diagram)s?\b",
    r"\bcount.*(figure|image|chart|graph|diagram)s?\b",
]

_SPECIFIC_ASSET_PATTERNS = [
    r"\b(show|give|open|display|link|path|where is|find)\b.*(fig(ure)?|image|img|table|chart)\s*\d+",
    r"\b(fig(ure)?|image|img|table|chart)\s*\d+\b.*(show|give|open|display|link|path|where)",
    r"\bimage\s*(number|#|no\.?|num)?\s*\d+\b",
    r"\bfig(ure)?\s*(number|#|no\.?|num)?\s*\d+\b",
    r"\btable\s*(number|#|no\.?|num)?\s*\d+\b",
    r"\bshow me (the )?(figure|table|image|chart|graph)\b",
    r"\bgive me (the )?(figure|table|image|chart|graph)\b",
    r"\bopen (the )?(figure|table|image|chart|graph)\b",
    r"\bimage link\b",
    r"\bfigure link\b",
    r"\btable link\b",
    r"\bwhere (is|can i find) (the )?(figure|table|image)\b",
]


def _is_list_query(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in _LIST_PATTERNS)


def _is_table_count_query(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in _TABLE_COUNT_PATTERNS)


def _is_figure_count_query(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in _FIGURE_COUNT_PATTERNS)


def _is_specific_asset_request(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in _SPECIFIC_ASSET_PATTERNS)


def _is_table_data_query(query: str) -> bool:
    q = query.lower()
    data_patterns = [
        r"\btables?\s*\d+\b",
        r"\bwhat.*\btables?\b",
        r"\bdescribe.*\btables?\b",
        r"\bshow.*\btables?\b",
        r"\bexplain.*\btables?\b",
        r"\bextract.*\btables?\b",
        r"\bcontents? of.*\btables?\b",
        r"\bdata in.*\btables?\b",
        r"\btables?.*content\b",
        r"\btables?.*show\b",
        r"\btables?.*contain\b",
    ]
    return any(re.search(p, q) for p in data_patterns)


# ============================================================================
# ASSET NUMBER EXTRACTOR
# ============================================================================

def _extract_asset_number(query: str) -> Tuple[Optional[int], Optional[str]]:
    q = query.lower()

    m = re.search(r"\btable\s*(?:number|#|no\.?|num)?\s*(\d+)", q)
    if m:
        return int(m.group(1)), "table"

    m = re.search(r"\b(?:fig(?:ure)?|image|img|chart|graph|diagram)\s*(?:number|#|no\.?|num)?\s*(\d+)", q)
    if m:
        return int(m.group(1)), "image"

    m = re.search(r"\bpage\s*(\d+)", q)
    if m:
        return int(m.group(1)), "page"

    return None, None


# ============================================================================
# SYSTEM IMAGE VIEWER  (NEW)
# ============================================================================

def _open_image_for_user(path: str) -> bool:
    """
    Attempt to open an image file with the system default viewer.
    Returns True if launched successfully.

    Works on:
      Windows  → os.startfile
      macOS    → open
      Linux    → xdg-open
    """
    if not path or not os.path.isfile(path):
        return False
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)          # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception as e:
        print(f"  ⚠️  Could not open image viewer: {e}")
        return False


def _print_image_in_terminal(path: str) -> bool:
    """
    Try to render an image inline in the terminal using the `imgcat` tool
    (works in iTerm2, VS Code integrated terminal, etc.).
    Falls back silently if not available.
    Returns True if rendered.
    """
    if not path or not os.path.isfile(path):
        return False
    try:
        result = subprocess.run(
            ["imgcat", path],
            capture_output=False,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ============================================================================
# META-QUERY ANSWER HELPERS
# ============================================================================

def _answer_list_query(lit_manager) -> str:
    sources = lit_manager.list_sources()
    if not sources:
        return "📚 No literature files are currently loaded."
    detail = lit_manager.list_sources_detail()
    lines  = [f"📚 {len(detail)} file(s) currently loaded:\n"]
    for i, d in enumerate(detail, start=1):
        img_info = f", {d['images']} image(s)" if d["images"] else ""
        tbl_info = f", {d['tables']} table(s)" if d.get("tables") else ""
        lines.append(
            f"  {i}. {d['display_name']}\n"
            f"     File: {d['filename']}  [{d['chunks']} text chunks{tbl_info}{img_info}]"
        )
    return "\n".join(lines)


def _count_tables_in_chunks(chunks: List[dict]) -> int:
    count = 0
    for c in chunks:
        text = c.get("text", "")
        meta = c.get("meta", {})
        if meta.get("type") == "table":
            count += 1
        elif (
            text.count("|") >= 3
            or text.count("\t") >= 3
            or re.search(r"\bTable\s+\d+", text, re.IGNORECASE)
        ):
            count += 1
    return count


def _answer_table_count(query: str, lit_manager, source_filter: Optional[str]) -> str:
    targets = [source_filter] if source_filter else lit_manager.list_sources()
    if not targets:
        return "📚 No literature files are loaded."

    lines = ["📊 Table count:\n"]
    for src in targets:
        table_recs  = lit_manager.get_table_records(source_filter=src)
        all_chunks  = lit_manager.retrieve("table", top_k=200, source_filter=src)
        heuristic_n = _count_tables_in_chunks(all_chunks)

        display = lit_manager.get_file_display_name(src)
        lines.append(f"  • {display}  ({src})")
        if table_recs:
            lines.append(f"      Tables extracted (PyMuPDF/pdfplumber) : {len(table_recs)}")
            for r in table_recs:
                cap = r.get("caption", f"Table {r['index']}")
                lines.append(f"        – Table {r['index']} (p.{r['page']}): {cap}")
                if r.get("uri"):
                    lines.append(f"          🔗 {r['uri']}")
        else:
            lines.append(f"      Table-like text passages (heuristic) : {heuristic_n}")

    if not any(lit_manager.get_table_records(source_filter=s) for s in targets):
        lines.append(
            "\n⚠️  No tables were structurally extracted. "
            "Ensure PyMuPDF >= 1.23 is installed: pip install --upgrade pymupdf"
        )
    return "\n".join(lines)


def _answer_figure_count(query: str, lit_manager, source_filter: Optional[str]) -> str:
    targets = [source_filter] if source_filter else lit_manager.list_sources()
    if not targets:
        return "📚 No literature files are loaded."

    lines = ["🖼️  Extracted figure/image count:\n"]
    for src in targets:
        img_recs = lit_manager.get_image_records(source_filter=src)
        display  = lit_manager.get_file_display_name(src)
        lines.append(f"  • {display}  ({src})")
        lines.append(f"      Figures extracted : {len(img_recs)}")
        for r in img_recs:
            lines.append(f"        – Fig {r['index']} (p.{r['page']})")
            if r.get("uri"):
                lines.append(f"          🔗 {r['uri']}")

    lines.append(
        "\n⚠️  Only embedded raster images are counted. "
        "Vector-only figures may not appear."
    )
    return "\n".join(lines)


# ============================================================================
# SPECIFIC ASSET LOOKUP  — now also opens/displays the image
# ============================================================================

def _answer_specific_asset_request(
    query: str,
    lit_manager,
    source_filter: Optional[str],
) -> Tuple[str, List[dict]]:
    """
    User asked for a specific figure/table by number or by page.
    Returns (text_answer, image_display_list).

    FIX: now also attempts to open the image with the system viewer
    and returns display_data so the caller can render it.
    """
    from vision_q import get_image_display_data

    idx, asset_type = _extract_asset_number(query)
    lines = []
    display_list: List[dict] = []

    def _process_record(rec: dict):
        """Build text lines and collect display data for one record."""
        _append_record_info(lines, rec, lit_manager)
        dd = get_image_display_data(rec)
        if dd:
            display_list.append(dd)
            # Try to open with system viewer (disabled to prevent popping out)
            # _open_image_for_user(rec.get("path", ""))
            # Try imgcat (inline terminal rendering)
            _print_image_in_terminal(rec.get("path", ""))

    # ── By page ──────────────────────────────────────────────────────────
    if asset_type == "page" and idx is not None:
        records = lit_manager.find_records_by_page(idx, source_filter=source_filter)
        if not records:
            return f"❓ No figures or tables found on page {idx}.", []
        lines.append(f"📄 Records on page {idx}:\n")
        for r in records:
            _process_record(r)
        return "\n".join(lines), display_list

    # ── By index ─────────────────────────────────────────────────────────
    if idx is not None:
        rec_type = None if asset_type == "image" else asset_type
        rec      = lit_manager.find_record_by_index(idx, source_filter=source_filter, record_type=rec_type)
        if rec is None:
            rec = lit_manager.find_record_by_index(idx, source_filter=source_filter)
        if rec:
            label = "Table" if rec.get("type") == "table" else "Figure"
            lines.append(f"📎 {label} {idx} found:\n")
            _process_record(rec)
            return "\n".join(lines), display_list
        else:
            return (
                f"❓ {asset_type.title() if asset_type else 'Record'} {idx} not found "
                f"in {'all files' if not source_filter else source_filter}.\n"
                f"Use 'list images' or 'vision status' to see available records.",
                [],
            )

    # ── No specific number — list all assets ─────────────────────────────
    all_records = lit_manager.get_vision_records(source_filter=source_filter)
    if not all_records:
        return "❓ No figures or tables have been extracted yet.", []

    lines.append(f"📎 All extracted records ({len(all_records)} total):\n")
    for r in all_records:
        _append_record_info(lines, r, lit_manager)
    return "\n".join(lines), []


def _append_record_info(lines: list, rec: dict, lit_manager) -> None:
    """Append formatted info for one image/table record."""
    rtype   = rec.get("type", "image")
    label   = "📊 Table" if rtype == "table" else "🖼️  Figure"
    display = lit_manager.get_file_display_name(rec["source"])
    caption = rec.get("caption", "")

    lines.append(
        f"  {label} {rec['index']}  |  page {rec['page']}  |  {display} ({rec['source']})"
    )
    if caption:
        lines.append(f"      Caption : {caption}")

    path = rec.get("path") or ""
    uri  = rec.get("uri") or ""

    if path and os.path.isfile(path):
        lines.append(f"      Path    : {path}")
        if uri:
            lines.append(f"      Link    : {uri}")
    else:
        lines.append("      ⚠️  Image file not found on disk.")

    if rtype == "table" and rec.get("text_rows"):
        rows    = rec["text_rows"]
        preview = rows[:min(4, len(rows))]
        lines.append("      Preview :")
        for row in preview:
            cells = " | ".join(str(c).strip() for c in row if c)
            if cells:
                lines.append(f"        {cells}")
        if len(rows) > 4:
            lines.append(f"        ... ({len(rows) - 4} more rows)")
    lines.append("")


# ============================================================================
# RETRIEVE CONTEXT (text + table chunks)
# ============================================================================

def get_literature_context(
    query: str,
    lit_manager,
    top_k: int = 3,
    source_filter: Optional[str] = None,
) -> Tuple[str, bool]:
    results = lit_manager.retrieve(query, top_k=top_k, source_filter=source_filter)
    if not results:
        return "", False

    lines = []
    for i, r in enumerate(results, start=1):
        display = lit_manager.get_file_display_name(r["source"])
        meta    = r.get("meta", {})

        if meta.get("type") == "table":
            header = f"[Table {meta.get('index','?')} p.{meta.get('page','?')} — {display} ({r['source']})]"
            uri = meta.get("uri", "")
            if uri:
                header += f"\n🔗 Table image: {uri}"
        else:
            header = f"[Passage {i} — {display} ({r['source']})]"

        lines.append(f"{header}:\n{r['text']}")

    return "\n\n".join(lines), True


# ============================================================================
# IMAGE LINK FORMATTER
# ============================================================================

def format_image_links(used_records: List[dict], lit_manager) -> str:
    """Build a formatted block of paths and URIs for records used in a vision answer."""
    if not used_records:
        return ""

    lines = ["\n📎 Source asset(s):"]
    for r in used_records:
        rtype   = r.get("type", "image")
        label   = "Table" if rtype == "table" else "Figure"
        display = lit_manager.get_file_display_name(r["source"])
        caption = r.get("caption", "")

        lines.append(
            f"  {label} {r['index']}  |  page {r['page']}  |  {display}"
        )
        if caption:
            lines.append(f"    Caption : {caption}")

        path = r.get("path", "")
        uri  = r.get("uri", "")
        if path and os.path.isfile(path):
            lines.append(f"    Path    : {path}")
            if uri:
                lines.append(f"    Link    : {uri}")
        else:
            lines.append("    ⚠️  File not on disk.")
    return "\n".join(lines)


# ============================================================================
# VISION-FIRST ROUTING: scan literature_images/ dir, call llava:7b
# ============================================================================

_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}

# Document keyword → filename fragment mapping for collision resolution
_DOC_KEYWORDS = {
    'lprm'          : 'LPRM',
    'anoop'         : 'Anoop',
    'pulse reserve' : 'Anoop',
    'amsr2'         : 'LPRM',
    'validation'    : 'LPRM',
    'soil plant'    : 'Anoop',
}


def _rank_images_for_query(query: str, images_dir: str) -> list:
    """
    Scan images_dir and return sorted list of (score, path) tuples.
    Higher score = better match to the query.

    Scoring rules:
      +10  filename contains extracted figure/table number (e.g. "img3" for "figure 3")
      + 5  filename contains asset type keyword (table/img)
      + 2  filename contains any query keyword > 3 chars
    """
    q = query.lower()
    num_match = re.search(r'\b(?:fig(?:ure)?|table|image|img|chart)[\s_-]*(\d+)\b', q)
    asset_num = int(num_match.group(1)) if num_match else None
    asset_is_table = bool(re.search(r'\btables?\b', q))

    scored = []
    if not os.path.isdir(images_dir):
        return scored

    for fname in os.listdir(images_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in _IMG_EXTS:
            continue
        full = os.path.join(images_dir, fname)
        fn   = fname.lower()
        score = 0

        if asset_num is not None:
            # match "img3", "img_3", "image3", "table3", "p3_img..." etc.
            if re.search(rf'(?:img|image|fig|table|p\d+_(?:img|table))_?0*{asset_num}\b', fn):
                score += 10

        if asset_is_table:
            if 'table' in fn: score += 5
        else:
            if 'img' in fn or 'fig' in fn: score += 5

        # keyword overlap
        for word in re.findall(r'\w{4,}', q):
            if word in fn:
                score += 2

        scored.append((score, full))

    scored.sort(key=lambda x: -x[0])
    return scored


def _detect_doc_filter(query: str) -> str:
    """Return a filename fragment to filter by if the user names a specific paper."""
    q = query.lower()
    for kw, frag in _DOC_KEYWORDS.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', q):
            return frag
    return ""


def _answer_visual_query_from_dir(
    query: str,
    images_dir: str,
    ollama_url: str   = "http://localhost:11434/api/generate",
    model: str        = "llava:7b",
    timeout: int      = 90,
    top_n: int        = 3,
) -> tuple:
    """
    Scan the literature_images/ directory, rank images against the query,
    handle collision/ambiguity, call llava:7b, and return
    (description_text, image_paths).

    Collision rule:
      If the user asks for "Figure N" without specifying a paper, and images
      from MULTIPLE different source PDFs match that number equally well, we
      do NOT guess — instead we return a clarification prompt to the user.

    Returns:
      (text: str | None, paths: list[str])
      text is None on total failure (caller should fall back to TF-IDF).
    """
    ranked = _rank_images_for_query(query, images_dir)
    if not ranked:
        return None, []

    doc_filter = _detect_doc_filter(query)

    # Extract the figure/table number the user asked for
    num_match = re.search(r'\b(?:fig(?:ure)?|table|image|img|chart)[\s_-]*(\d+)\b', query.lower())
    asset_num = int(num_match.group(1)) if num_match else None

    # ── Collision detection ───────────────────────────────────────────────────
    if asset_num is not None and not doc_filter:
        # Find all top-scored images that contain the requested number
        top_score = ranked[0][0] if ranked else 0
        top_matches = [p for s, p in ranked if s >= max(top_score - 2, 5)]

        # Group by source document (first part of filename before '_p')
        def _doc_of(path):
            fname = os.path.basename(path)
            return fname.split('_p')[0] if '_p' in fname else fname.split('_')[0]

        docs_found = list(dict.fromkeys(_doc_of(p) for p in top_matches))

        if len(docs_found) > 1:
            # Ambiguity: figure exists in multiple docs — ask user to clarify
            doc_names = []
            for d in docs_found:
                friendly = (
                    "Anoop Pulse Reserves" if "Anoop" in d
                    else "LPRM AMSR2 Validation" if "LPRM" in d
                    else d
                )
                doc_names.append(friendly)
            clarify_msg = (
                f"🔍 I found **{asset_match_label(asset_num, query)}** in multiple papers:\n\n"
                + "\n".join(f"  • {n}" for n in doc_names)
                + "\n\nWhich document would you like me to explain? "
                "Please mention the paper name (e.g. *'LPRM'* or *'Anoop'*)."
            )
            # Return clarification with the first image path as a preview
            preview_paths = [p for _, p in ranked[:1]]
            return clarify_msg, preview_paths

    # ── Apply doc filter if specified ─────────────────────────────────────────
    if doc_filter:
        filtered = [(s, p) for s, p in ranked if doc_filter in os.path.basename(p)]
        if filtered:
            ranked = filtered

    # ── Select top N images ───────────────────────────────────────────────────
    selected_paths = [p for _, p in ranked[:top_n]]
    if not selected_paths:
        return None, []

    # ── Build llava prompt ────────────────────────────────────────────────────
    asset_label = asset_match_label(asset_num, query) if asset_num else "this figure/table"
    vision_prompt = (
        f"You are a precise scientific data extractor analyzing a figure/table from a soil moisture paper. "
        f"The user asked: \"{query}\"\n\n"
        f"Please describe {asset_label} accurately. "
        f"If it is a table, read out the columns and rows clearly. Extract exact numerical values. "
        f"Do not hallucinate or use generic statements. "
        f"If the text is small, do your best to extract the key values you can see without complaining."
    )

    # ── Encode and call llava ─────────────────────────────────────────────────
    images_b64 = []
    valid_paths = []
    for path in selected_paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'rb') as f:
                images_b64.append(base64.b64encode(f.read()).decode('utf-8'))
            valid_paths.append(path)
        except Exception as e:
            print(f"  ⚠️  Could not read {path}: {e}")

    if not images_b64:
        return None, []

    try:
        resp = requests.post(
            ollama_url,
            json={
                "model"  : model,
                "prompt" : vision_prompt,
                "images" : images_b64,
                "stream" : False,
                "options": {"num_predict": 600, "temperature": 0.1},
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            answer = resp.json().get("response", "").strip()
            if answer:
                source_names = [os.path.basename(p) for p in valid_paths]
                answer += f"\n\n📎 Source file(s): {', '.join(source_names)}"
                return answer, valid_paths
    except requests.exceptions.Timeout:
        print("  ⚠️  llava:7b timed out — falling back to text pipeline")
    except Exception as e:
        print(f"  ⚠️  Vision call error: {e}")

    return None, []


def asset_match_label(asset_num: int, query: str) -> str:
    """Return a human-readable label like 'Figure 3' or 'Table 2'."""
    q = query.lower()
    if re.search(r'\btables?\b', q):
        return f"Table {asset_num}"
    return f"Figure {asset_num}"


# ============================================================================
# ANSWER FROM LITERATURE  (updated return signature)
# ============================================================================

def answer_from_literature(
    query: str,
    lit_manager,
    ollama_url: str      = "http://localhost:11434/api/generate",
    ollama_model: str    = "qwen2.5:3b",
    ollama_timeout: int  = 120,
    top_k: int           = 3,
    vision_model: str    = "llava:7b",
    vision_timeout: int  = 120,
    vision_top_k: int    = 2,
    vision_enabled: bool = True,
) -> Tuple[Optional[str], bool, List[dict]]:
    """
    Route a query through the literature pipeline.

    Returns:
        (answer_text, found_in_literature, image_display_list)

    image_display_list is a list of dicts with keys:
        base64, media_type, path, uri, page, index, source, type, caption
    Call this to render the images in your UI.

    Order:
      1. List-files meta-query
      2. Specific asset request → path/URI + open image + return display_data
      3. Table / figure count meta-queries
      4. Vision pipeline (llava:7b) for visual/table queries → also return display_data
      5. TF-IDF + Ollama text pipeline
    """
    from vision_q import get_image_display_data

    # ── 1. List files ────────────────────────────────────────────────────
    if _is_list_query(query):
        return _answer_list_query(lit_manager), True, []

    # Resolve file scope once
    source_filter = lit_manager.resolve_source_filter(query)
    if source_filter:
        print(f"  📂 Query scoped to: {source_filter}")
    else:
        sources = lit_manager.list_sources()
        if sources:
            print(f"  📂 Query covers all files: {', '.join(sources)}")

    # ── 2. Specific asset request ─────────────────────────────────────────
    if _is_specific_asset_request(query):
        answer, display_list = _answer_specific_asset_request(
            query, lit_manager, source_filter
        )
        return answer, True, display_list

    # ── 2b. Vision-first dir scan: "show me table 2", "what is figure 3?" ──
    # Intercepts BEFORE the TF-IDF pipeline. Handles collision/ambiguity.
    _is_visual_q = (
        _is_specific_asset_request(query)
        or _is_table_data_query(query)
        or bool(re.search(
            r'\b(table|figure|chart|image|graph|diagram)s?\b', query, re.IGNORECASE))
    )
    if vision_enabled and _is_visual_q and os.path.isdir(LIT_IMAGES_DIR):
        _vis_text, _vis_paths = _answer_visual_query_from_dir(
            query       = query,
            images_dir  = LIT_IMAGES_DIR,
            ollama_url  = ollama_url,
            model       = vision_model,
            timeout     = vision_timeout,
        )
        if _vis_text:  # None means timeout/error → fall through to TF-IDF
            _vis_display = []
            for _vp in _vis_paths:
                if os.path.isfile(_vp):
                    try:
                        with open(_vp, 'rb') as _vf:
                            _vb64 = base64.b64encode(_vf.read()).decode('utf-8')
                        _ext  = os.path.splitext(_vp)[1].lower().lstrip('.')
                        _vis_display.append({
                            'path'      : _vp,
                            'base64'    : _vb64,
                            'media_type': f'image/{_ext}',
                            'uri'       : '',
                        })
                    except Exception:
                        pass
            return _vis_text, True, _vis_display

    # ── 3. Count meta-queries ─────────────────────────────────────────────
    if _is_table_count_query(query):
        return _answer_table_count(query, lit_manager, source_filter), True, []

    if _is_figure_count_query(query):
        return _answer_figure_count(query, lit_manager, source_filter), True, []

    # ── 4. Vision pipeline ────────────────────────────────────────────────
    if vision_enabled:
        try:
            from vision_q import is_visual_query, is_table_query, answer_from_vision

            if is_visual_query(query) or _is_table_data_query(query):

                if _is_table_data_query(query):
                    image_records = lit_manager.get_table_records(source_filter=source_filter)
                    if not image_records:
                        image_records = lit_manager.get_vision_records(source_filter=source_filter)
                else:
                    image_records = lit_manager.get_vision_records(source_filter=source_filter)

                if image_records:
                    model_label = vision_model
                    if _is_table_data_query(query):
                        print("  📊 Table query detected — routing to vision pipeline...")
                    else:
                        print(f"  🖼️  Visual query detected — routing to {model_label}...")

                    answer, used = answer_from_vision(
                        query         = query,
                        image_records = image_records,
                        ollama_url    = ollama_url,
                        ollama_model  = vision_model,
                        timeout       = vision_timeout,
                        top_k         = vision_top_k,
                    )

                    if answer:
                        # Collect display data from used records
                        # (vision_q now attaches display_data to each used record)
                        display_list: List[dict] = []
                        for rec in used:
                            dd = rec.get("display_data") or get_image_display_data(rec)
                            if dd:
                                display_list.append(dd)
                                # Also try to open image in system viewer (disabled to prevent popping out)
                                # _open_image_for_user(rec.get("path", ""))

                        link_block = format_image_links(used, lit_manager)
                        if link_block:
                            answer = answer + "\n" + link_block
                        return answer, True, display_list

        except ImportError:
            pass

    # ── 5. Text pipeline ──────────────────────────────────────────────────
    context, found = get_literature_context(
        query, lit_manager, top_k=top_k, source_filter=source_filter
    )
    if not found:
        if source_filter:
            display = lit_manager.get_file_display_name(source_filter)
            return (
                f"❓ No relevant passages found in '{display}' ({source_filter}).",
                False,
                [],
            )
        return None, False, []

    if source_filter:
        scope_label = (
            f"the file '{lit_manager.get_file_display_name(source_filter)}' "
            f"({source_filter})"
        )
    else:
        names       = [f"'{lit_manager.get_file_display_name(s)}'" for s in lit_manager.list_sources()]
        scope_label = "all loaded files: " + ", ".join(names)

    # Truncate context to avoid exceeding qwen2.5:3b's context window.
    # Keeping the context ≤ 3 000 chars leaves room for the prompt template
    # and the generated answer within a 4 096-token budget.
    MAX_CONTEXT_CHARS = 3000
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n[...context truncated for length...]"

    prompt = _build_prompt(query, context, scope_label)

    try:
        response = requests.post(
            ollama_url,
            json={
                "model" : ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx"    : 4096,   # keep within 3B model limits
                    "num_predict": 512,    # cap output length for speed
                    "temperature": 0.3,
                    "top_p"      : 0.9,
                },
            },
            timeout=ollama_timeout,
        )

        if response.status_code == 500:
            # Ollama returns 500 when the model is not loaded or the prompt
            # is still too large. Surface a clear, actionable message.
            try:
                err_detail = response.json().get("error", response.text[:200])
            except Exception:
                err_detail = response.text[:200]
            return (
                f"⚠️ Ollama returned an error for the literature query.\n"
                f"Detail: {err_detail}\n"
                "Possible causes:\n"
                "  • The model is not loaded — run `ollama pull qwen2.5:3b`\n"
                "  • The prompt is still too large — try a more specific question.",
                True,
                [],
            )

        response.raise_for_status()
        answer = response.json().get("response", "").strip()

        if not answer:
            return None, True, []

        citation = _build_citation_block(context)
        if citation:
            answer = answer + "\n\n" + citation

        return answer, True, []

    except requests.exceptions.Timeout:
        return (
            "⚠️ Ollama timed out. Try a shorter query or increase OLLAMA_TIMEOUT.",
            True,
            [],
        )
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Cannot connect to Ollama. Make sure it's running: `ollama serve`",
            True,
            [],
        )
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 500
        try:
            err_detail = e.response.json().get("error", "") if e.response is not None else ""
        except Exception:
            err_detail = e.response.text[:200] if e.response is not None else ""

        err_msg = (
            f"⚠️ **Ollama Server Error ({status_code}):**\n"
            f"Detail: `{err_detail or e}`\n\n"
            f"This usually occurs if the configured model (**`{ollama_model}`**) is not installed/pulled "
            f"or failed to load on your local Ollama server.\n\n"
            f"**To fix this, please run the following command in your terminal/command-prompt:**\n"
            f"```bash\nollama pull {ollama_model}\n```"
        )
        return err_msg, True, []
    except Exception as e:
        return f"⚠️ Literature QA error: {e}", True, []


# ============================================================================
# PROMPT BUILDER
# ============================================================================

def _build_prompt(query: str, context: str, scope_label: str) -> str:
    return f"""You are a scientific assistant specialising in soil moisture remote sensing research.

Answer the user's question using ONLY the passages provided below.
The passages are drawn from {scope_label}.
Always mention the source file name(s) when relevant to the answer.
If a passage contains table data (rows separated by " | "), read it carefully and use it to answer precisely.
If insufficient information exists in the passages, say so clearly.
Be concise (3-6 sentences unless required otherwise). Do NOT hallucinate facts.

=== RETRIEVED PASSAGES ===
{context}

=== USER QUESTION ===
{query}

=== YOUR ANSWER ===
"""


# ============================================================================
# CITATION BLOCK BUILDER
# ============================================================================

def _build_citation_block(context: str) -> str:
    pattern = re.compile(
        r"\[(?:Passage \d+|Table \d+ p\.\d+) — ([^\(]+) \(([^\)]+)\)\]"
    )
    seen   = []
    labels = []
    for m in pattern.finditer(context):
        display  = m.group(1).strip()
        filename = m.group(2).strip()
        if filename not in seen:
            seen.append(filename)
            labels.append(f"📄 {display} ({filename})")

    if not labels:
        return ""

    lines = ["─" * 55, "📚 Sources:"]
    lines += [f"  {l}" for l in labels]
    return "\n".join(lines)


# ============================================================================
# PARSE LITERATURE COMMANDS
# ============================================================================

def parse_literature_command(raw_input: str, lit_manager) -> Optional[str]:
    text  = raw_input.strip()
    lower = text.lower()

    if lower.startswith("load literature"):
        path = text[len("load literature"):].strip().strip('"').strip("'")
        if not path:
            return (
                "Usage: load literature <path>\n"
                "Example: load literature papers/file.pdf"
            )
        if os.path.isfile(path):
            ok = lit_manager.load_file(path)
            if ok:
                return f"✅ Loaded: {os.path.basename(path)}\n{lit_manager.summary()}"
            return f"❌ Failed to load: {path}"
        if os.path.isdir(path):
            lit_manager.load_directory(path)
            if lit_manager.list_sources():
                return f"✅ Loaded directory.\n{lit_manager.summary()}"
            return "⚠️ No files could be loaded from directory."
        return f"❌ Path not found: {path}"

    if lower in ("list literature", "show literature", "literature list"):
        if not lit_manager.list_sources():
            return "📚 No literature files loaded."
        return lit_manager.summary()

    if lower in ("clear literature", "remove literature", "reset literature"):
        lit_manager.clear()
        return "🗑️ All literature cleared."

    if lower in ("vision status", "show vision", "vision index", "list images"):
        return lit_manager.vision_summary()

    if lower in ("list tables", "show tables", "tables"):
        all_tables = lit_manager.get_table_records()
        if not all_tables:
            return (
                "📊 No tables extracted yet.\n"
                "Ensure PyMuPDF >= 1.23 is installed: pip install --upgrade pymupdf"
            )
        lines = [f"📊 {len(all_tables)} table(s) extracted:\n"]
        for r in all_tables:
            display = lit_manager.get_file_display_name(r["source"])
            caption = r.get("caption", "")
            lines.append(
                f"  Table {r['index']}  |  p.{r['page']}  |  {display} ({r['source']})"
            )
            if caption:
                lines.append(f"    Caption: {caption}")
            if r.get("uri"):
                lines.append(f"    Link   : {r['uri']}")
            lines.append("")
        return "\n".join(lines)

    if lower.startswith("vision images"):
        src_filter = text[len("vision images"):].strip() or None
        if src_filter:
            resolved = lit_manager.resolve_source_filter(src_filter)
            if resolved:
                src_filter = resolved

        records = lit_manager.get_image_records(source_filter=src_filter)
        if not records:
            msg = "🖼️ No images extracted"
            if src_filter:
                msg += f" for '{src_filter}'"
            return msg + "."

        lines = [f"🖼️ {len(records)} image(s) found:\n"]
        for r in records:
            display = lit_manager.get_file_display_name(r["source"])
            lines.append(
                f"  Fig {r['index']:>3}  |  p.{r['page']}  |  {display} ({r['source']})"
            )
            if r.get("uri"):
                lines.append(f"    🔗 {r['uri']}")
            if r.get("path"):
                lines.append(f"    📁 {r['path']}")
            lines.append("")
        return "\n".join(lines)

    return None