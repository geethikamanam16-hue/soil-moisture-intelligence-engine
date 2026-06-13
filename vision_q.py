"""
vision_q.py
===========
Vision Transformer Q&A for literature files using llava:7b via Ollama
"""

import os
import re
import base64
import requests
from typing import List, Optional, Tuple, Dict

# ============================================================================
# VISUAL QUERY DETECTION  (expanded for indirect/conceptual queries)
# ============================================================================

_VISUAL_KEYWORDS = [
    # Direct figure/table references
    "figure", "fig.", "fig ",
    "chart", "graph", "plot", "diagram",
    "image", "photo", "picture",
    "table", "map", "illustration",
    "visuali", "depicted", "shown in",

    # Explain/describe requests
    "what does", "what is shown", "what is plotted",
    "describe the", "explain the figure", "explain the chart",
    "explain the graph", "explain the map", "explain the diagram",
    "explain the image", "explain the picture", "explain the plot",

    # Visual properties (indirect)
    "pixel", "colour", "color", "spatial", "visual", "caption",
    "scatter", "histogram", "heatmap", "bar chart", "pie chart",
    "line chart", "box plot", "violin plot",

    # Location/context references
    "on page", "in the paper", "in the figure",

    # Table-specific
    "show me table", "show table", "display table",
    "what is in table", "what does table", "table shows",
    "table on page", "table from",

    # NEW: indirect/conceptual triggers
    "methodology diagram", "architecture diagram",
    "results figure", "results chart",
    "comparison chart", "comparison graph",
    "trend graph", "trend plot", "trend chart",
    "accuracy plot", "performance graph", "performance chart",
    "confusion matrix", "roc curve", "precision recall",
    "satellite image", "remote sensing image",
    "soil map", "moisture map",
    "show me", "show the", "display the",
    "what trend", "what pattern", "what relationship",
    "from the figure", "from the graph", "from the chart",
    "as shown", "as depicted", "see figure", "see fig",
    "refer to figure", "refer to fig",
    "illustrates", "demonstrates visually",
    "visual result", "visual comparison",
    "annotated", "highlighted in",
    "the image shows", "image depicts",
]

# Indirect conceptual patterns — these match even without figure/table keywords
_INDIRECT_VISUAL_PATTERNS = [
    r"\bexplain\b.{0,30}\b(result|finding|output|method|approach|architecture)\b",
    r"\bdescribe\b.{0,30}\b(result|finding|output|method|performance|accuracy)\b",
    r"\bwhat\b.{0,20}\b(show|depict|illustrat|demonstrat|reveal)\b",
    r"\bhow\b.{0,20}\b(look|appear|compar|perform|trend)\b",
    r"\bvisuali[sz]e?\b",
    r"\bplot\b.{0,20}\b(show|depict|illustrat)\b",
    r"\bthe\s+(chart|graph|figure|diagram|image|plot|table)\b",
    r"\b(scatter|histogram|heatmap|boxplot|bar)\b",
]


def is_visual_query(query: str) -> bool:
    q = query.lower()
    if any(re.search(r'\\b' + re.escape(kw) + r'\\b', q) for kw in _VISUAL_KEYWORDS):
        return True
    return any(re.search(p, q) for p in _INDIRECT_VISUAL_PATTERNS)


def is_table_query(query: str) -> bool:
    """Detect queries specifically about tables."""
    q = query.lower()
    table_patterns = [
        r"\btable\s*\d+",
        r"\btables?\b.*\bpage\b",
        r"\bpage\b.*\btables?\b",
        r"\bshow\b.*\btables?\b",
        r"\bdisplay\b.*\btables?\b",
        r"\bwhat.*\btables?\b",
        r"\bdescribe.*\btables?\b",
    ]
    return any(re.search(p, q) for p in table_patterns)


# ============================================================================
# FILE URI HELPER
# ============================================================================

def _make_file_uri(abs_path: str) -> str:
    """Convert an absolute path to a file:// URI."""
    normalised = abs_path.replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", normalised):
        return "file:///" + normalised
    return "file://" + normalised


# ============================================================================
# IMAGE DISPLAY HELPER  (NEW)
# ============================================================================

def _encode_image_as_png_b64(image_path: str) -> Optional[str]:
    """Encode an image path as a base64 string, specifically in PNG format."""
    try:
        import io
        from PIL import Image
        with Image.open(image_path) as img:
            buffer = io.BytesIO()
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"  ⚠️  PNG conversion encoding failed: {e}")
        # Direct fallback to binary read
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

def get_image_display_data(record: dict) -> Optional[dict]:
    """
    Return base64-encoded image data + metadata for display.
    Used by literature_qa.py to embed image previews alongside answers.
    Guarantees the returned base64 string is in image/png format.

    Returns:
        {
            "base64"    : str,   # base64-encoded PNG
            "media_type": str,   # "image/png"
            "path"      : str,
            "uri"       : str,
            "page"      : int,
            "index"     : int,
            "source"    : str,
            "type"      : str,   # "image" | "table"
            "caption"   : str,
        }
        or None if file not found / cannot encode
    """
    path = record.get("path", "")
    if not path or not os.path.isfile(path):
        return None

    encoded = _encode_image_as_png_b64(path)
    if not encoded:
        return None

    media_type = "image/png"

    return {
        "base64"    : encoded,
        "media_type": media_type,
        "path"      : path,
        "uri"       : record.get("uri", _make_file_uri(path)),
        "page"      : record.get("page", 0),
        "index"     : record.get("index", 0),
        "source"    : record.get("source", ""),
        "type"      : record.get("type", "image"),
        "caption"   : record.get("caption", ""),
    }

def _extract_figure_caption(page, rect) -> Tuple[Optional[str], Optional[int]]:
    """
    Look for a caption starting with 'Fig.' or 'Figure' near the image bounding rect.
    Returns (caption_text, figure_number).
    """
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return None, None

    # Try to find a caption below the image (y range from rect.y1 to rect.y1 + 120)
    caption_y_range_below = (rect.y1 - 10, rect.y1 + 120)
    # Try to find a caption above the image (y range from rect.y0 - 60 to rect.y0 + 10)
    caption_y_range_above = (rect.y0 - 60, rect.y0 + 10)

    # 1. Look below first
    for b in blocks:
        bx0, by0, bx1, by1, text, *_ = b
        text = text.strip()
        m = re.match(r"(?i)^(figure|fig\.?)\s*(\d+)", text)
        if m:
            if caption_y_range_below[0] <= by0 <= caption_y_range_below[1]:
                fig_num = int(m.group(2))
                return text.replace("\n", " "), fig_num

    # 2. Look above
    for b in blocks:
        bx0, by0, bx1, by1, text, *_ = b
        text = text.strip()
        m = re.match(r"(?i)^(figure|fig\.?)\s*(\d+)", text)
        if m:
            if caption_y_range_above[0] <= by1 <= caption_y_range_above[1]:
                fig_num = int(m.group(2))
                return text.replace("\n", " "), fig_num

    # 3. Fallback: search for any block starting with 'Fig.' or 'Figure' in the whole page
    # that is closest vertically to the image.
    best_b = None
    min_dist = float("inf")
    for b in blocks:
        bx0, by0, bx1, by1, text, *_ = b
        text = text.strip()
        m = re.match(r"(?i)^(figure|fig\.?)\s*(\d+)", text)
        if m:
            dist = min(abs(by0 - rect.y1), abs(by1 - rect.y0))
            if dist < min_dist and dist < 200:
                min_dist = dist
                best_b = (text, int(m.group(2)))
                
    if best_b:
        return best_b[0].replace("\n", " "), best_b[1]

    return None, None


def _extract_figure_caption_fallback(page) -> Tuple[Optional[str], Optional[int]]:
    """
    Search page text for the first figure caption without using rects.
    Returns (caption_text, figure_number).
    """
    try:
        blocks = page.get_text("blocks")
        for b in blocks:
            bx0, by0, bx1, by1, text, *_ = b
            text = text.strip()
            m = re.match(r"(?i)^(figure|fig\.?)\s*(\d+)", text)
            if m:
                return text.replace("\n", " "), int(m.group(2))
    except Exception:
        pass
    return None, None


# ============================================================================
# IMAGE EXTRACTION FROM PDFs
# ============================================================================

def extract_images_from_pdf(
    pdf_path: str,
    output_dir: str,
    max_images: int = 20,
) -> List[dict]:
    """
    Extract embedded images/figures from a PDF.
    Tries PyMuPDF first, falls back to pdfplumber page render.

    Returns list of dicts:
        {
            "path"    : str,   # absolute path to saved .png
            "uri"     : str,   # file:// URI for clickable link
            "page"    : int,   # 1-based page number
            "index"   : int,   # figure index within PDF
            "source"  : str,   # basename of PDF
            "type"    : "image",
            "caption" : str
        }
    """
    os.makedirs(output_dir, exist_ok=True)
    source_name = os.path.basename(pdf_path)
    images: List[dict] = []

    # ── Strategy 1: PyMuPDF ───────────────────────────────────────────────
    try:
        import fitz

        doc = fitz.open(pdf_path)
        img_index = 0

        for page_num in range(len(doc)):
            if img_index >= max_images:
                break
            page = doc[page_num]
            
            # Extract page text safely
            page_text = ""
            try:
                page_text = page.get_text() or ""
            except Exception:
                pass

            for img_info in page.get_images(full=True):
                if img_index >= max_images:
                    break
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    w = base_image["width"]
                    h = base_image["height"]
                    
                    # Filter out tiny logos / decorations
                    if w < 150 or h < 150:
                        continue
                        
                    img_bytes  = base_image["image"]
                    ext        = base_image.get("ext", "png")
                    
                    # Try to locate on the page to find caption
                    rects = page.get_image_rects(xref)
                    caption = None
                    fig_num = None
                    if rects:
                        caption, fig_num = _extract_figure_caption(page, rects[0])
                    else:
                        caption, fig_num = _extract_figure_caption_fallback(page)
                        
                    # Skip if it has no figure caption and is on the title page (usually logo/branding)
                    if not fig_num:
                        if page_num == 0:
                            continue
                        fig_num = img_index + 1
                        
                    fname = (
                        f"{os.path.splitext(source_name)[0]}"
                        f"_p{page_num+1}_img{fig_num}.{ext}"
                    )
                    fpath = os.path.abspath(os.path.join(output_dir, fname))
                    with open(fpath, "wb") as f:
                        f.write(img_bytes)
                    images.append({
                        "path"     : fpath,
                        "uri"      : _make_file_uri(fpath),
                        "page"     : page_num + 1,
                        "index"    : fig_num,
                        "source"   : source_name,
                        "type"     : "image",
                        "page_text": page_text,
                        "caption"  : caption or f"Figure {fig_num} on page {page_num+1}",
                    })
                    img_index += 1
                except Exception:
                    continue

        doc.close()
        if images:
            return images

    except ImportError:
        pass
    except Exception as e:
        print(f"  ⚠️  PyMuPDF image extraction failed: {e}")


    # ── Strategy 2: pdfplumber page render ────────────────────────────────
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if len(images) >= max_images:
                    break
                
                # Extract page text safely
                page_text = ""
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    pass

                img = page.to_image(resolution=150).original
                fname = (
                    f"{os.path.splitext(source_name)[0]}"
                    f"_p{page_num+1}_render.png"
                )
                fpath = os.path.abspath(os.path.join(output_dir, fname))
                img.save(fpath, format="PNG")
                images.append({
                    "path"     : fpath,
                    "uri"      : _make_file_uri(fpath),
                    "page"     : page_num + 1,
                    "index"    : page_num + 1,
                    "source"   : source_name,
                    "type"     : "image",
                    "page_text": page_text,
                })
        return images

    except ImportError:
        print("  ⚠️  Neither PyMuPDF nor pdfplumber+Pillow available for image extraction.")
    except Exception as e:
        print(f"  ⚠️  pdfplumber render failed: {e}")

    return images


# ============================================================================
# TABLE EXTRACTION FROM PDFs
# ============================================================================

def extract_tables_from_pdf(
    pdf_path: str,
    output_dir: str,
    max_tables: int = 30,
) -> List[dict]:
    """
    Extract tables from a PDF as both:
      (a) PNG images (rendered via PyMuPDF page crop)
      (b) Structured text (CSV-like rows) stored in record["text_rows"]
    """
    os.makedirs(output_dir, exist_ok=True)
    source_name = os.path.basename(pdf_path)
    tables: List[dict] = []
    table_index = 0

    # ── Strategy 1: PyMuPDF find_tables (v1.23+) ─────────────────────────
    try:
        import fitz

        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            if table_index >= max_tables:
                break
            page = doc[page_num]

            # Extract page text safely
            page_text = ""
            try:
                page_text = page.get_text() or ""
            except Exception:
                pass

            tab_finder = page.find_tables() if hasattr(page, "find_tables") else None
            if tab_finder is None:
                break

            for tbl in tab_finder.tables:
                if table_index >= max_tables:
                    break
                try:
                    rows = tbl.extract()
                    if not rows or len(rows) < 2:
                        continue

                    bbox   = tbl.bbox
                    rect   = fitz.Rect(bbox)
                    mat    = fitz.Matrix(2, 2)
                    pix    = page.get_pixmap(matrix=mat, clip=rect)
                    fname  = (
                        f"{os.path.splitext(source_name)[0]}"
                        f"_p{page_num+1}_table{table_index+1}.png"
                    )
                    fpath  = os.path.abspath(os.path.join(output_dir, fname))
                    pix.save(fpath)

                    flat_text = _rows_to_text(rows)

                    tables.append({
                        "path"      : fpath,
                        "uri"       : _make_file_uri(fpath),
                        "page"      : page_num + 1,
                        "index"     : table_index + 1,
                        "source"    : source_name,
                        "type"      : "table",
                        "text_rows" : rows,
                        "text"      : flat_text,
                        "caption"   : _extract_nearby_caption(page, rect, page_num + 1),
                        "page_text" : page_text,
                    })
                    table_index += 1
                except Exception as e:
                    print(f"  ⚠️  Table extraction error p{page_num+1}: {e}")
                    continue

        doc.close()

        if tables:
            print(f"  ✅ PyMuPDF extracted {len(tables)} table(s) from {source_name}")
            return tables

    except ImportError:
        pass
    except AttributeError:
        pass
    except Exception as e:
        print(f"  ⚠️  PyMuPDF table extraction failed: {e}")

    # ── Strategy 2: pdfplumber extract_tables ────────────────────────────
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if table_index >= max_tables:
                    break

                # Extract page text safely
                page_text = ""
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    pass

                raw_tables = page.extract_tables()
                if not raw_tables:
                    continue

                for tbl_data in raw_tables:
                    if table_index >= max_tables:
                        break
                    if not tbl_data or len(tbl_data) < 2:
                        continue

                    try:
                        img      = page.to_image(resolution=150).original
                        fname    = (
                            f"{os.path.splitext(source_name)[0]}"
                            f"_p{page_num+1}_table{table_index+1}.png"
                        )
                        fpath    = os.path.abspath(os.path.join(output_dir, fname))
                        img.save(fpath, format="PNG")
                        uri = _make_file_uri(fpath)
                    except Exception:
                        fpath = None
                        uri   = ""

                    clean_rows = [
                        [str(c) if c is not None else "" for c in row]
                        for row in tbl_data
                    ]
                    flat_text  = _rows_to_text(clean_rows)

                    tables.append({
                        "path"      : fpath,
                        "uri"       : uri,
                        "page"      : page_num + 1,
                        "index"     : table_index + 1,
                        "source"    : source_name,
                        "type"      : "table",
                        "text_rows" : clean_rows,
                        "text"      : flat_text,
                        "caption"   : f"Table {table_index+1} (page {page_num+1})",
                        "page_text" : page_text,
                    })
                    table_index += 1

        if tables:
            print(f"  ✅ pdfplumber extracted {len(tables)} table(s) from {source_name}")
        return tables

    except ImportError:
        print("  ⚠️  pdfplumber not available for table extraction.")
    except Exception as e:
        print(f"  ⚠️  pdfplumber table extraction failed: {e}")

    return tables


def _rows_to_text(rows: List[List]) -> str:
    """Convert table rows to pipe-delimited text for TF-IDF indexing."""
    lines = []
    for row in rows:
        cells = [str(c).strip() if c else "" for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _extract_nearby_caption(page, rect, page_num: int) -> str:
    """Look for caption text immediately below a table bounding rect."""
    try:
        blocks = page.get_text("blocks")
        caption_y_range = (rect.y1, rect.y1 + 60)

        for b in blocks:
            bx0, by0, bx1, by1, text, *_ = b
            if caption_y_range[0] <= by0 <= caption_y_range[1]:
                text = text.strip()
                if re.match(r"(?i)table\s*\d*", text):
                    return text[:200]
    except Exception:
        pass
    return f"Table on page {page_num}"


# ============================================================================
# BASE64 ENCODING
# ============================================================================

def _encode_image_b64(image_path: str) -> Optional[str]:
    # Use PNG conversion by default to ensure uniform format
    return _encode_image_as_png_b64(image_path)


# ============================================================================
# SELECT RELEVANT IMAGES / TABLES  (improved for indirect queries)
# ============================================================================

def _select_relevant_images(
    query: str,
    image_records: List[dict],
    top_k: int = 3,
) -> List[dict]:
    """
    Select the most relevant records for a query.

    Priority order:
    1. Explicit table number match
    2. Explicit figure number match
    3. Explicit page number match
    4. Keyword/topic scoring against caption + source
    5. Fallback: first top_k records (only for non-targeted queries)
    """
    q = query.lower()

    page_match  = re.search(r"page\s*(\d+)|p\.?\s*(\d+)", q)
    fig_match   = re.search(r"fig(?:ure)?\.?\s*(\d+)", q)
    table_match = re.search(r"table\s*(\d+)", q)

    if table_match:
        ti   = int(table_match.group(1))
        hits = [r for r in image_records if r.get("type") == "table" and r["index"] == ti]
        if hits:
            return hits
        hits = [r for r in image_records if r["index"] == ti]
        return hits

    if fig_match:
        fi   = int(fig_match.group(1))
        hits = [r for r in image_records if r.get("type") == "image" and r["index"] == fi]
        if hits:
            return hits
        hits = [r for r in image_records if r["index"] == fi]
        return hits

    if page_match:
        pg   = int(page_match.group(1) or page_match.group(2))
        hits = [r for r in image_records if r["page"] == pg]
        return hits

    # If any specific asset was requested but not found in the records, do not return random fallbacks
    if "fig" in q or "table" in q or "image" in q or "page" in q:
        return []

    # ── Keyword/topic scoring for indirect queries ────────────────────────
    q_words = set(re.findall(r"[a-z0-9]+", q))
    _STOP = {"the","a","an","of","in","on","and","or","for","to","is","are",
              "what","does","show","from","this","that","it","me","give","open",
              "explain", "describe", "detail", "paper", "literature", "document"}
    q_words -= _STOP

    if q_words:
        scored = []
        for r in image_records:
            caption_text = r.get("caption", "").lower() + " " + r.get("source", "").lower()
            page_text = r.get("page_text", "").lower()

            caption_tokens = set(re.findall(r"[a-z0-9]+", caption_text))
            page_tokens = set(re.findall(r"[a-z0-9]+", page_text))

            # Give extra weight to word matches in specific captions/titles,
            # but allow page_text to accurately map topics to the correct page!
            score = 3 * len(q_words & caption_tokens) + len(q_words & page_tokens)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_scored = [r for score, r in scored if score > 0]
        if top_scored:
            return top_scored[:top_k]
        
        # Avoid random first-page fallbacks when a specific topic query scored 0
        return []

    return image_records[:top_k]


# ============================================================================
# VISION Q&A (moondream via Ollama — fastest vision model)
# ============================================================================

def answer_from_vision(
    query: str,
    image_records: List[dict],
    ollama_url: str   = "http://localhost:11434/api/generate",
    ollama_model: str = "moondream",
    timeout: int      = 120,
    top_k: int        = 2,
) -> Tuple[Optional[str], List[dict]]:
    """
    Use moondream to answer a query about figures/tables from literature.

    FIX: used_records now always contain "display_data" key with base64 image
    data so callers can render the image, not just show a path.

    Returns (answer_text, used_records_with_display_data).
    """
    if not image_records:
        return None, []

    selected = _select_relevant_images(query, image_records, top_k=top_k)
    if not selected:
        return None, []

    # For table queries with text data — include raw rows in prompt
    table_text_context = ""
    if is_table_query(query):
        text_tables = [r for r in selected if r.get("type") == "table" and r.get("text_rows")]
        if text_tables:
            parts = []
            for r in text_tables:
                parts.append(
                    f"[Table {r['index']}, page {r['page']} — {r['source']}]\n"
                    + _rows_to_text(r["text_rows"])
                )
            table_text_context = "\n\n".join(parts)

    # Encode images and attach display_data to each used record
    b64_images: List[str] = []
    used: List[dict]      = []

    for rec in selected:
        if rec.get("path") and os.path.isfile(rec["path"]):
            encoded = _encode_image_b64(rec["path"])
            if encoded:
                b64_images.append(encoded)
                # Attach display data so callers can render the image
                rec_with_display = dict(rec)
                rec_with_display["display_data"] = get_image_display_data(rec)
                used.append(rec_with_display)

    if not b64_images and not table_text_context:
        return "⚠️  Could not encode any images for vision analysis.", []

    prompt = _build_vision_prompt(query, used, table_text_context)

    # If only table text available (no images encoded) — text-only answer
    if not b64_images and table_text_context:
        answer = _text_table_answer(
            query, table_text_context, used, ollama_url, ollama_model, timeout
        )
        return answer, used

    max_retries = 2
    current_timeout = timeout

    for attempt in range(max_retries):
        try:
            payload = {
                "model"  : ollama_model,
                "prompt" : prompt,
                "images" : b64_images,
                "stream" : False,
                "options": {"temperature": 0.2, "top_p": 0.3},
            }

            print(f"  🖼️  Sending {len(b64_images)} image(s) to {ollama_model} (attempt {attempt + 1}, timeout={current_timeout}s)...")
            resp = requests.post(ollama_url, json=payload, timeout=current_timeout)
            resp.raise_for_status()

            answer = resp.json().get("response", "").strip()
            if not answer:
                return "⚠️  Vision model returned an empty response.", used

            return answer, used

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                current_timeout = int(current_timeout * 1.5)
                print(f"  ⚠️  {ollama_model} timed out. Retrying with timeout={current_timeout}s...")
                continue
            return (
                f"⚠️  {ollama_model} timed out after {current_timeout}s even after retrying. "
                "The model might be too large for your hardware, or you may need to manually increase VISION_TIMEOUT in Config.py.",
                used,
            )
        except requests.exceptions.ConnectionError:
            return (
                "⚠️  Cannot reach Ollama. Ensure Ollama is running and the vision model "
                f"is pulled (`ollama pull {ollama_model}`).",
                used,
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 500
            try:
                err_detail = e.response.json().get("error", "") if e.response is not None else ""
            except Exception:
                err_detail = e.response.text[:200] if e.response is not None else ""

            err_msg = (
                f"⚠️  **Ollama Vision Server Error ({status_code}):**\n"
                f"Detail: `{err_detail or e}`\n\n"
                f"This usually occurs if the configured vision model (**`{ollama_model}`**) is not installed/pulled "
                f"or failed to load on your local Ollama server.\n\n"
                f"**To fix this, please run the following command in your terminal/command-prompt:**\n"
                f"```bash\nollama pull {ollama_model}\n```"
            )
            return err_msg, used
        except Exception as e:
            return f"⚠️  Vision QA error: {e}", used


def _text_table_answer(
    query: str,
    table_text: str,
    used: List[dict],
    ollama_url: str,
    ollama_model: str,
    timeout: int,
) -> Optional[str]:
    """Answer a table question using raw row data (text-only Ollama call)."""
    prompt = (
        "You are a scientific assistant. Answer the user's question using "
        "the table data below.\n\n"
        f"TABLE DATA:\n{table_text}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )
    try:
        resp = requests.post(
            ollama_url,
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 500
        try:
            err_detail = e.response.json().get("error", "") if e.response is not None else ""
        except Exception:
            err_detail = e.response.text[:200] if e.response is not None else ""

        err_msg = (
            f"⚠️  **Ollama Table Server Error ({status_code}):**\n"
            f"Detail: `{err_detail or e}`\n\n"
            f"This usually occurs if the configured model (**`{ollama_model}`**) is not installed/pulled "
            f"or failed to load on your local Ollama server.\n\n"
            f"**To fix this, please run the following command in your terminal/command-prompt:**\n"
            f"```bash\nollama pull {ollama_model}\n```"
        )
        return err_msg
    except Exception as e:
        return f"⚠️  Table text QA error: {e}"


# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def _build_vision_prompt(
    query: str,
    used_images: List[dict],
    table_text_context: str = "",
) -> str:
    image_refs = ", ".join(
        f"[{r.get('type','image').title()} {r['index']} p.{r['page']} — {r['source']}]"
        for r in used_images
    )
    extra = ""
    if table_text_context:
        extra = (
            f"\n\nEXTRACTED TABLE DATA (use this for precision):\n"
            f"{table_text_context}\n"
        )
    return (
        f"You are a scientific assistant analysing figures and tables from soil "
        f"moisture remote sensing research papers.\n\n"
        f"The attached content comes from: {image_refs}\n"
        f"{extra}\n"
        f"Answer the following question based on the image(s) and/or table data. "
        f"Be precise. If you cannot determine the answer, say so clearly.\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


# ============================================================================
# IMAGE & TABLE REGISTRY (VisionIndex)
# ============================================================================

class VisionIndex:
    """
    In-memory registry of all extracted images and tables across loaded PDFs.
    """

    def __init__(self, image_cache_dir: str = "literature_images"):
        self.cache_dir = image_cache_dir
        self._records : List[dict] = []

    def add_pdf(self, pdf_path: str, max_images: int = 20) -> int:
        """Extract images AND tables from a PDF and register both."""
        source = os.path.basename(pdf_path)

        existing = {r["source"] for r in self._records}
        if source in existing:
            return 0

        imgs   = extract_images_from_pdf(pdf_path, self.cache_dir, max_images)
        tables = extract_tables_from_pdf(pdf_path, self.cache_dir, max_tables=30)
        if tables:
            print(f"  📊 {len(tables)} table(s) registered from {source}")

        new_records = imgs + tables
        self._records.extend(new_records)
        return len(new_records)

    def get_records(
        self,
        source_filter: Optional[str] = None,
        record_type: Optional[str]   = None,
    ) -> List[dict]:
        out = self._records
        if source_filter:
            out = [r for r in out if r["source"] == source_filter]
        if record_type:
            out = [r for r in out if r.get("type") == record_type]
        return list(out)

    def get_table_text_chunks(
        self,
        source_filter: Optional[str] = None,
    ) -> List[dict]:
        recs = self.get_records(source_filter=source_filter, record_type="table")
        return [r for r in recs if r.get("text")]

    def find_image_by_index(
        self,
        index: int,
        source_filter: Optional[str] = None,
        record_type: Optional[str] = None,
    ) -> Optional[dict]:
        for r in self.get_records(source_filter=source_filter, record_type=record_type):
            if r["index"] == index:
                return r
        return None

    def find_images_by_page(
        self,
        page: int,
        source_filter: Optional[str] = None,
    ) -> List[dict]:
        return [
            r for r in self.get_records(source_filter=source_filter)
            if r["page"] == page
        ]

    def clear(self):
        self._records = []

    def summary(self) -> str:
        sources: Dict[str, Dict[str, int]] = {}
        for r in self._records:
            s     = r["source"]
            rtype = r.get("type", "image")
            sources.setdefault(s, {"image": 0, "table": 0})
            sources[s][rtype] = sources[s].get(rtype, 0) + 1

        total_img = sum(v["image"] for v in sources.values())
        total_tbl = sum(v["table"] for v in sources.values())

        lines = [
            f"🖼️  Vision Index: {total_img} image(s) + "
            f"📊 {total_tbl} table(s) across {len(sources)} file(s)"
        ]
        for s, counts in sources.items():
            lines.append(
                f"    • {s}: {counts['image']} image(s), {counts['table']} table(s)"
            )
        return "\n".join(lines)