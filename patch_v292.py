import os, re

BASE = r'C:\Users\geeth\OneDrive\Documents\zip_smcodes\ml_SoilMoisture'

# 1. Update Query_classifier.py
cls_path = os.path.join(BASE, 'Query_classifier.py')
with open(cls_path, 'r', encoding='utf-8') as f:
    cls_src = f.read()

# Add logic for "latest" to `_extract_all_date_ranges`
# Find where years are extracted or around the end of the method
# Let's insert a check at the very beginning of `_extract_all_date_ranges`
old_extract_start = """    def _extract_all_date_ranges(self, text: str) -> list:
        \"\"\"
        Returns a list of (start_date, end_date) tuples.
        \"\"\"
        ranges = []
"""
new_extract_start = """    def _extract_all_date_ranges(self, text: str) -> list:
        \"\"\"
        Returns a list of (start_date, end_date) tuples.
        \"\"\"
        ranges = []
        
        # ── 0. Catch "latest", "present", "now" ──
        # We will map these to 2099-12-31 so the date boundary clipper handles it.
        # This replaces "latest" with "2099" in the text so the year parser catches it.
        text = re.sub(r'\\b(latest|present|now)\\b', '2099', text, flags=re.IGNORECASE)
"""
if old_extract_start in cls_src:
    cls_src = cls_src.replace(old_extract_start, new_extract_start)
else:
    print("WARNING: Could not patch _extract_all_date_ranges in Query_classifier.py")

with open(cls_path, 'w', encoding='utf-8') as f:
    f.write(cls_src)
print("Query_classifier.py updated.")

# 2. Update literature_qa.py
lit_path = os.path.join(BASE, 'literature_qa.py')
with open(lit_path, 'r', encoding='utf-8') as f:
    lit_src = f.read()

lit_src = lit_src.replace(
    r"\b(table|figure|chart|image|graph|diagram)\b",
    r"\b(table|figure|chart|image|graph|diagram)s?\b"
)
lit_src = lit_src.replace(
    r"\btable\b",
    r"\btables?\b"
)

# And in _rank_images_for_query, asset_is_table = bool(re.search(r'\btables?\b', q))
with open(lit_path, 'w', encoding='utf-8') as f:
    f.write(lit_src)
print("literature_qa.py updated.")

# 3. Update vision_q.py
vis_path = os.path.join(BASE, 'vision_q.py')
with open(vis_path, 'r', encoding='utf-8') as f:
    vis_src = f.read()

vis_src = vis_src.replace(
    r"\b(table|figure|chart|image|graph|diagram)\b",
    r"\b(table|figure|chart|image|graph|diagram)s?\b"
)
vis_src = vis_src.replace(
    r"\btable\b",
    r"\btables?\b"
)
with open(vis_path, 'w', encoding='utf-8') as f:
    f.write(vis_src)
print("vision_q.py updated.")

# 4. Update app.py Guardrails
app_path = os.path.join(BASE, 'app.py')
with open(app_path, 'r', encoding='utf-8') as f:
    app_src = f.read()

old_sys = """        "If asked what you can do, explain strictly that you can only do the following: "\\n        "1. Analyze the Zarr soil moisture dataset (2002-07-01 to 2023-12-30). "\\n        "2. Query the Cloud SMAP Google Earth Engine datasets. "\\n        "3. Answer questions specifically from the 'Anoop_pulse_reserves.pdf' and 'LPRM_Anoop.pdf' literature files. "\\n        "Do not mention any other capabilities. "\\n"""
new_sys = """        "You are the Soil Moisture Intelligence Engine. Your core expertise is:\\n"\\n        "1. Analyzing the Zarr soil moisture dataset (2002-07-01 to 2023-12-30).\\n"\\n        "2. Querying Cloud SMAP Google Earth Engine datasets.\\n"\\n        "3. Answering questions from 'Anoop_pulse_reserves.pdf' and 'LPRM_Anoop.pdf'.\\n"\\n        "You may converse naturally, but if asked about your capabilities, only mention these core functions. Do not claim to have general internet access, broad hydrology models, or external databases beyond what is provided.\\n"\\n"""
# Let's do a more robust string replacement for app_src in case formatting is slightly off.
# Or just use regex to replace the whole block.
sys_match = re.search(r'"If asked what you can do.*?mention any other capabilities\.\s*"', app_src, re.DOTALL)
if sys_match:
    app_src = app_src.replace(sys_match.group(0), new_sys.strip())
else:
    print("WARNING: Could not find system prompt in app.py")

with open(app_path, 'w', encoding='utf-8') as f:
    f.write(app_src)
print("app.py updated.")
