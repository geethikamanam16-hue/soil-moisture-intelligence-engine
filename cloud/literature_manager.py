import os

# ==========================================
# PORTABLE PATH RESOLUTION
# ==========================================

# This file lives at: <project_root>/cloud/literature_manager.py
# Project root is one level up from the cloud/ directory.
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# ==========================================
# CACHE FOLDER
# ==========================================

CACHE_FOLDER = os.path.join(_PROJECT_ROOT, "cache", "literature")

os.makedirs(CACHE_FOLDER, exist_ok=True)

# ==========================================
# SYNC LITERATURE (LOCAL ONLY)
# ==========================================

def sync_literature():
    """
    Previously synced PDFs from Google Drive.
    Now simply returns the local literature cache folder — PDFs are already there.
    """
    print("\nUsing local literature cache...")

    pdf_files = [f for f in os.listdir(CACHE_FOLDER) if f.lower().endswith(".pdf")]

    if pdf_files:
        print(f"Found {len(pdf_files)} PDF(s) in local cache: {', '.join(pdf_files)}")
    else:
        print(f"⚠️  No PDFs found in local literature cache: {CACHE_FOLDER}")

    return CACHE_FOLDER

if __name__ == "__main__":
    cache_path = sync_literature()
    print(f"All literature files cached successfully in {cache_path}")