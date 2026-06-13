import sys, os, re

BASE = r'C:\Users\geeth\OneDrive\Documents\zip_smcodes\ml_SoilMoisture'

# 1. Update Query_classifier.py to set region='india' for driest/wettest state
qc_path = os.path.join(BASE, 'Query_classifier.py')
with open(qc_path, 'r', encoding='utf-8') as f:
    qc_src = f.read()

# Instead of modifying classify locally, we add driest_state logic:
target = """        # Step 1: operation
        result['operation'] = self._detect_operation(q)"""
insert = """        # Step 1: operation
        result['operation'] = self._detect_operation(q)
        if result['operation'] in ['driest_state', 'wettest_state']:
            result['region'] = 'india'
            result['region_missing'] = False"""
if insert not in qc_src:
    qc_src = qc_src.replace(target, insert)

with open(qc_path, 'w', encoding='utf-8') as f:
    f.write(qc_src)

print("Updated Query_classifier.py")


# 2. Update main.py to intercept driest_state/wettest_state
main_path = os.path.join(BASE, 'main.py')
with open(main_path, 'r', encoding='utf-8') as f:
    main_src = f.read()

# Add a function _run_state_ranking_query
ranking_fn = """
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
    
    print(f"\\n╔{'═'*72}╗")
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
            
    print(f"╚{'═'*72}╝\\n")
"""

# Insert it before `process_single_query`
if '_run_state_ranking_query' not in main_src:
    main_src = main_src.replace('def process_single_query(', ranking_fn + '\ndef process_single_query(')

# Intercept inside process_single_query
inter_target = """    if cls["operation"] == "comparison":
        _run_comparison_query(
            cls, engine, validator, ds_start, ds_end, lit_manager, intent
        )
        return"""
inter_insert = """    if cls["operation"] in ["driest_state", "wettest_state"]:
        _run_state_ranking_query(cls, engine, validator, ds_start, ds_end)
        return

    if cls["operation"] == "comparison":
        _run_comparison_query(
            cls, engine, validator, ds_start, ds_end, lit_manager, intent
        )
        return"""
if 'driest_state' not in main_src:
    main_src = main_src.replace(inter_target, inter_insert)

with open(main_path, 'w', encoding='utf-8') as f:
    f.write(main_src)
print("Updated main.py")


# 3. App.py intercept for Chat RAG. 
app_path = os.path.join(BASE, 'app.py')
with open(app_path, 'r', encoding='utf-8') as f:
    app_src = f.read()

app_target = """        if cls.get("operation") == "comparison":
            data_context = build_comparison_info(cls)
        else:
            all_ranges = cls.get("all_date_ranges", [])"""
app_insert = """        if cls.get("operation") in ["driest_state", "wettest_state"]:
            # Let the LLM know about the state ranking script
            data_context = {
                "operation": cls["operation"],
                "notice": "A background process is calculating the state rankings and will print them to the user. Acknowledge this."
            }
        elif cls.get("operation") == "comparison":
            data_context = build_comparison_info(cls)
        else:
            all_ranges = cls.get("all_date_ranges", [])"""

if '"driest_state", "wettest_state"' not in app_src:
    app_src = app_src.replace(app_target, app_insert)
    
with open(app_path, 'w', encoding='utf-8') as f:
    f.write(app_src)
print("Updated app.py")
