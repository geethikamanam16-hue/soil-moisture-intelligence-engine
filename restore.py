import sys, os

path = r'C:\Users\geeth\OneDrive\Documents\zip_smcodes\ml_SoilMoisture\Query_classifier.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find _extract_all_date_ranges definition
idx = -1
for i, line in enumerate(lines):
    if "def _extract_all_date_ranges" in line:
        idx = i
        break

if idx != -1:
    # Look at what is above idx. We need to restore the clamp code.
    restore_code = """        v1, v2, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
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
"""
    # Let's verify if the code is actually missing.
    # We look backwards from idx.
    if "return None" not in lines[idx-1]:
        # It's missing. We insert it at idx.
        lines.insert(idx, restore_code)
        
with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('File restored.')
