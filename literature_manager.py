"""
literature_manager.py
=====================
Loads PDF / DOCX / TXT literature files, chunks their text,
and builds a TF-IDF index for semantic retrieval.

UPDATES in this version:
  - On load_file() for PDFs, table records from VisionIndex are also injected
    into the TF-IDF index as extra chunks — so table content is searchable.
  - resolve_source_filter: no recursion (original fix retained).
  - Vision re-indexing on startup retained.
  - list_sources_detail() now shows table counts separately.
  - New method: get_table_records() for direct table access.

Supports:
  - .pdf  (via pdfplumber, fallback to pypdf)
  - .docx (via python-docx)
  - .txt / .md (plain read)
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
import json
import re
import math
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

# ── file loaders ─────────────────────────────────────────────────────────────

def _load_pdf(path: str) -> str:
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        if text.strip():
            return text
    except Exception:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception:
        pass

    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(path)
    except Exception as e:
        raise RuntimeError(f"Cannot extract PDF text: {e}")


def _load_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        raise RuntimeError(f"Cannot extract DOCX text: {e}")


def _load_txt(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode text file: {path}")


# ── chunker ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


# ── TF-IDF engine ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _TFIDF:
    def __init__(self):
        self.chunks    : List[str] = []
        self.sources   : List[str] = []
        self.chunk_meta: List[dict] = []   # NEW: extra metadata per chunk
        self.idf       : dict      = {}
        self._tf_cache : List[dict] = []

    def add_documents(
        self,
        chunks: List[str],
        source: str,
        meta: Optional[List[dict]] = None,
    ):
        for i, c in enumerate(chunks):
            self.chunks.append(c)
            self.sources.append(source)
            self.chunk_meta.append(meta[i] if meta and i < len(meta) else {})

    def build_index(self):
        N = len(self.chunks)
        df = defaultdict(int)
        self._tf_cache = []
        for chunk in self.chunks:
            tokens = _tokenize(chunk)
            freq   = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            total = max(len(tokens), 1)
            tf = {t: c / total for t, c in freq.items()}
            self._tf_cache.append(tf)
            for t in freq:
                df[t] += 1
        self.idf = {
            t: math.log((N + 1) / (cnt + 1)) + 1
            for t, cnt in df.items()
        }

    def query(
        self,
        text: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        q_tokens = _tokenize(text)
        q_freq   = defaultdict(int)
        for t in q_tokens:
            q_freq[t] += 1
        q_total = max(len(q_tokens), 1)

        scores = []
        for idx, tf in enumerate(self._tf_cache):
            if source_filter and self.sources[idx] != source_filter:
                continue
            score = 0.0
            for t, qf in q_freq.items():
                if t in tf:
                    q_tfidf = (qf / q_total) * self.idf.get(t, 0)
                    d_tfidf = tf[t]            * self.idf.get(t, 0)
                    score  += q_tfidf * d_tfidf
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ── source-filter resolver ────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    name = os.path.splitext(name)[0]
    name = name.lower()
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def resolve_source_filter(
    query: str,
    loaded_sources: List[str],
) -> Optional[str]:
    """
    Detect whether the user's query mentions a specific loaded file.
    Returns matching filename or None (use ALL files).
    No recursion — path basename extracted inline.
    """
    q_lower = query.lower()

    # Inline basename extraction from paths
    path_pattern = re.compile(
        r"""
        (?:
            [a-zA-Z]:\\[^\s`'"]+
          | /[^\s`'"]+
          | \.{1,2}/[^\s`'"]+
        )
        """,
        re.VERBOSE,
    )
    for m in path_pattern.finditer(query):
        raw_path = m.group(0).strip("\"'`")
        basename = os.path.basename(raw_path)
        if basename:
            q_lower = q_lower + " " + basename.lower()

    norm_sources = {s: _normalize(s) for s in loaded_sources}

    # Step 1: exact basename
    for src in loaded_sources:
        if src.lower() in q_lower:
            return src

    # Step 2: normalised stem exact
    for src, norm in norm_sources.items():
        if norm in q_lower:
            return src

    # Step 3: significant-word partial match
    _STOP = {
        "the","a","an","of","in","on","and","or","for",
        "to","is","are","was","were","be","by","pdf",
        "docx","txt","file","paper","journal","article",
    }
    source_words: Dict[str, List[str]] = {}
    for src, norm in norm_sources.items():
        words = [w for w in norm.split() if w not in _STOP and len(w) >= 4]
        source_words[src] = words

    word_freq: Dict[str, int] = defaultdict(int)
    for words in source_words.values():
        for w in set(words):
            word_freq[w] += 1

    n_sources = len(loaded_sources)
    matches: Dict[str, int] = defaultdict(int)
    for src, words in source_words.items():
        for w in words:
            if word_freq[w] < n_sources and w in q_lower:
                matches[src] += 1

    if matches:
        best_src     = max(matches, key=lambda s: matches[s])
        best_count   = matches[best_src]
        second_best  = sorted(matches.values(), reverse=True)
        if len(second_best) < 2 or second_best[1] < best_count:
            return best_src

    return None


# ── LiteratureManager ─────────────────────────────────────────────────────────

class LiteratureManager:
    """
    Load, index, and retrieve passages from literature files.
    Tables extracted by VisionIndex are injected into TF-IDF for searchability.
    """

    SUPPORTED = {".pdf", ".docx", ".txt", ".md"}

    def __init__(
        self,
        index_path        : str  = "literature_index.json",
        vision_enabled    : bool = True,
        vision_cache_dir  : str  = "literature_images",
        vision_max_images : int  = 20,
    ):
        self.index_path    = index_path
        self._tfidf        = _TFIDF()
        self._loaded_sources: List[str] = []
        self._index_built  = False
        self._known_dirs   : List[str] = []

        # ── Vision Index ──────────────────────────────────────────────────
        self._vision_enabled = vision_enabled
        if vision_enabled:
            try:
                from vision_q import VisionIndex
                self._vision_index    = VisionIndex(image_cache_dir=vision_cache_dir)
                self._vision_max_images = vision_max_images
                self._vision_available  = True
            except ImportError:
                print("  ⚠️  vision_q.py not found — vision pipeline disabled.")
                self._vision_available = False
                self._vision_index     = None
        else:
            self._vision_available = False
            self._vision_index     = None

        self._load_index()

    # ── directory registration ────────────────────────────────────────────

    def set_literature_dir(self, directory: str):
        directory = os.path.abspath(directory)
        if directory not in self._known_dirs:
            self._known_dirs.append(directory)
        if self._vision_available and self._loaded_sources:
            self._reindex_vision_from_dir(directory)

    def _reindex_vision_from_dir(self, directory: str):
        if not self._vision_available:
            return
        already_indexed = {r["source"] for r in self._vision_index.get_records()}
        for fname in self._loaded_sources:
            if not fname.lower().endswith(".pdf"):
                continue
            if fname in already_indexed:
                continue
            candidate = os.path.join(directory, fname)
            if not os.path.isfile(candidate):
                continue
            print(f"  🖼️  Re-indexing figures/tables from {fname} ...", end=" ", flush=True)
            n = self._vision_index.add_pdf(candidate, max_images=self._vision_max_images)
            print(f"{n} record(s) found." if n else "none found.")
            # Inject table chunks into TF-IDF
            self._inject_table_chunks(fname)

    # ── public API ────────────────────────────────────────────────────────

    def load_file(self, path: str) -> bool:
        ext      = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED:
            print(f"  ⚠️  Unsupported file type: {ext}  ({path})")
            return False

        filename = os.path.basename(path)
        if filename in self._loaded_sources:
            print(f"  ℹ️  Already loaded: {filename}")
            return True

        print(f"  📄 Loading: {filename} ...", end=" ", flush=True)
        try:
            if ext == ".pdf":
                text = _load_pdf(path)
            elif ext == ".docx":
                text = _load_docx(path)
            else:
                text = _load_txt(path)
        except Exception as e:
            print(f"FAILED ({e})")
            return False

        if not text or not text.strip():
            print("EMPTY — skipped")
            return False

        chunks = _chunk_text(text)
        if not chunks:
            print("NO CHUNKS — skipped")
            return False

        self._tfidf.add_documents(chunks, filename)
        self._loaded_sources.append(filename)
        self._index_built = False
        print(f"OK  ({len(chunks)} chunks)")
        self._rebuild_and_save()

        # Vision: extract images AND tables from PDFs
        if self._vision_available and ext == ".pdf":
            print(f"  🖼️  Extracting figures & tables from {filename} ...", end=" ", flush=True)
            n = self._vision_index.add_pdf(path, max_images=self._vision_max_images)
            print(f"{n} record(s) extracted." if n else "none found.")
            # Inject extracted table text into TF-IDF
            self._inject_table_chunks(filename)

        return True

    def _inject_table_chunks(self, filename: str):
        """
        Take table records from VisionIndex for *filename* and add their
        text content to TF-IDF so table data is searchable.
        """
        if not self._vision_available or self._vision_index is None:
            return
        table_records = self._vision_index.get_table_text_chunks(source_filter=filename)
        if not table_records:
            return

        new_chunks = []
        new_meta   = []
        for rec in table_records:
            text = rec.get("text", "")
            caption = rec.get("caption", "")
            full = f"TABLE {rec['index']} (page {rec['page']}): {caption}\n{text}"
            new_chunks.append(full)
            new_meta.append({
                "type"    : "table",
                "page"    : rec["page"],
                "index"   : rec["index"],
                "path"    : rec.get("path"),
                "uri"     : rec.get("uri", ""),
                "caption" : caption,
            })

        if new_chunks:
            self._tfidf.add_documents(new_chunks, filename, meta=new_meta)
            self._index_built = False
            self._tfidf.build_index()
            self._index_built = True
            print(f"  📊 Injected {len(new_chunks)} table chunk(s) into TF-IDF for {filename}")

    def load_directory(self, directory: str):
        if not os.path.isdir(directory):
            print(f"⚠️  Directory not found: {directory}")
            return

        files = sorted(
            f for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in self.SUPPORTED
        )
        if not files:
            print("⚠️  No supported files found in directory.")
            return

        for fname in files:
            self.load_file(os.path.join(directory, fname))

        directory = os.path.abspath(directory)
        if directory not in self._known_dirs:
            self._known_dirs.append(directory)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        source_filter: Optional[str] = None,
    ) -> List[dict]:
        if not self._loaded_sources:
            return []
        if not self._index_built:
            self._tfidf.build_index()
            self._index_built = True

        hits    = self._tfidf.query(query, top_k=top_k, source_filter=source_filter)
        results = []
        for idx, score in hits:
            if score > 0:
                entry = {
                    "source": self._tfidf.sources[idx],
                    "text"  : self._tfidf.chunks[idx],
                    "score" : round(score, 6),
                    "meta"  : self._tfidf.chunk_meta[idx],
                }
                results.append(entry)
        return results

    def resolve_source_filter(self, query: str) -> Optional[str]:
        return resolve_source_filter(query, self._loaded_sources)

    def get_file_display_name(self, filename: str) -> str:
        return os.path.splitext(filename)[0]

    def get_source_title(self, filename: str) -> str:
        """
        Return a human-readable title for a source file.
        For PDFs: tries to read the /Title metadata field first.
        Falls back to a cleaned-up version of the filename stem.
        """
        if filename.lower().endswith(".pdf"):
            for d in self._known_dirs:
                fpath = os.path.join(d, filename)
                if os.path.isfile(fpath):
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(fpath)
                        meta = reader.metadata
                        if meta and getattr(meta, "title", None):
                            t = str(meta.title).strip()
                            if t:
                                return t
                    except Exception:
                        pass
                    try:
                        import pdfplumber
                        with pdfplumber.open(fpath) as pdf:
                            info = pdf.metadata or {}
                            t = info.get("Title", "").strip()
                            if t:
                                return t
                    except Exception:
                        pass
                    break
        # Fallback: clean filename stem
        stem = os.path.splitext(filename)[0]
        stem = re.sub(r"[_\-]+", " ", stem)
        stem = re.sub(r"\s+", " ", stem).strip()
        return stem.title()

    def list_sources_with_paths(self) -> List[dict]:
        """
        Return list of dicts with keys:
          filename   : str  — raw filename (basename)
          title      : str  — human-readable title from PDF metadata or cleaned filename
          full_path  : str  — absolute path to file (empty string if not found)
        """
        result = []
        for fname in self._loaded_sources:
            full_path = ""
            for d in self._known_dirs:
                candidate = os.path.join(d, fname)
                if os.path.isfile(candidate):
                    full_path = os.path.abspath(candidate)
                    break
            result.append({
                "filename" : fname,
                "title"    : self.get_source_title(fname),
                "full_path": full_path,
            })
        return result

    def list_sources_detail(self) -> List[dict]:
        chunk_counts : Dict[str, int] = defaultdict(int)
        table_counts : Dict[str, int] = defaultdict(int)

        for idx, src in enumerate(self._tfidf.sources):
            meta = self._tfidf.chunk_meta[idx] if idx < len(self._tfidf.chunk_meta) else {}
            if meta.get("type") == "table":
                table_counts[src] += 1
            else:
                chunk_counts[src] += 1

        image_counts: Dict[str, int] = defaultdict(int)
        if self._vision_available and self._vision_index:
            for rec in self._vision_index.get_records(record_type="image"):
                image_counts[rec["source"]] += 1

        detail = []
        for fname in self._loaded_sources:
            detail.append({
                "filename"    : fname,
                "display_name": self.get_file_display_name(fname),
                "chunks"      : chunk_counts.get(fname, 0),
                "tables"      : table_counts.get(fname, 0),
                "images"      : image_counts.get(fname, 0),
            })
        return detail

    def get_vision_records(
        self,
        source_filter: Optional[str] = None,
        record_type: Optional[str]   = None,
    ) -> List[dict]:
        if not self._vision_available or self._vision_index is None:
            return []
        return self._vision_index.get_records(
            source_filter=source_filter,
            record_type=record_type,
        )

    def get_table_records(self, source_filter: Optional[str] = None) -> List[dict]:
        """Convenience method: return only table records."""
        return self.get_vision_records(source_filter=source_filter, record_type="table")

    def get_image_records(self, source_filter: Optional[str] = None) -> List[dict]:
        """Convenience method: return only image records."""
        return self.get_vision_records(source_filter=source_filter, record_type="image")

    def find_record_by_index(
        self,
        index: int,
        source_filter: Optional[str] = None,
        record_type: Optional[str] = None,
    ) -> Optional[dict]:
        """Find a specific figure or table by its index number."""
        if not self._vision_available or self._vision_index is None:
            return None
        return self._vision_index.find_image_by_index(index, source_filter, record_type)

    def find_records_by_page(
        self,
        page: int,
        source_filter: Optional[str] = None,
    ) -> List[dict]:
        if not self._vision_available or self._vision_index is None:
            return []
        return self._vision_index.find_images_by_page(page, source_filter)

    def vision_summary(self) -> str:
        if not self._vision_available or self._vision_index is None:
            return "🖼️  Vision pipeline not available."
        return self._vision_index.summary()

    def list_sources(self) -> List[str]:
        return list(self._loaded_sources)

    def summary(self) -> str:
        n      = len(self._tfidf.chunks)
        detail = self.list_sources_detail()
        lines  = ["📚 Literature Index Summary"]
        lines.append(f"  Files  : {len(detail)}")
        lines.append(f"  Chunks : {n}")
        for d in detail:
            img_info = f", {d['images']} image(s)" if d["images"] else ""
            tbl_info = f", {d['tables']} table(s)" if d["tables"] else ""
            lines.append(
                f"    • {d['filename']}  "
                f"[{d['chunks']} text chunks{tbl_info}{img_info}]"
            )
        if self._vision_available:
            lines.append("")
            lines.append(self._vision_index.summary())
        return "\n".join(lines)

    def clear(self):
        self._tfidf          = _TFIDF()
        self._loaded_sources = []
        self._index_built    = False
        if self._vision_available:
            self._vision_index.clear()
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        print("🗑️  Literature cleared.")

    # ── persistence ───────────────────────────────────────────────────────

    def _rebuild_and_save(self):
        self._tfidf.build_index()
        self._index_built = True
        try:
            data = {
                "sources"      : self._loaded_sources,
                "chunks"       : self._tfidf.chunks,
                "chunk_sources": self._tfidf.sources,
                "chunk_meta"   : self._tfidf.chunk_meta,
            }
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"  ⚠️  Could not save index: {e}")

    def _load_index(self):
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunks        = data.get("chunks", [])
            chunk_sources = data.get("chunk_sources", [])
            sources       = data.get("sources", [])
            chunk_meta    = data.get("chunk_meta", [{} for _ in chunks])

            if not chunks:
                return

            self._tfidf.chunks      = chunks
            self._tfidf.sources     = chunk_sources
            self._tfidf.chunk_meta  = chunk_meta
            self._loaded_sources    = sources
            self._tfidf.build_index()
            self._index_built       = True
            print(
                f"📚 Restored literature index: "
                f"{len(sources)} file(s), {len(chunks)} chunks."
            )
        except Exception as e:
            print(f"⚠️  Could not restore literature index: {e}")