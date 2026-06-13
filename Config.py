"""
Config.py
=========
Complete Configuration for Soil Moisture Analysis Engine
Centralised configuration for all operations, paths, and display settings
"""

# ============================================================================
# FILE PATHS - CUSTOMIZE FOR YOUR SYSTEM
# ============================================================================

# ZARR and SHAPEFILE paths are now handled dynamically via cloud sync
# Data is cached in the 'cache/' directory.
OUTPUT_PATH    = "latest_analysis.png"

# ============================================================================
# OLLAMA CONFIGURATION  (text LLM)
# ============================================================================

OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "qwen2.5:3b"    # text-only model for NLP tasks
OLLAMA_TIMEOUT = 180             # increased to 180s to prevent timeouts on average machines

# ============================================================================
# VISION MODEL CONFIGURATION  (llava:7b / multimodal)
# ============================================================================

# Using llava:7b — the state-of-the-art vision model for literature Q&A.
# Pull command:
#   ollama pull llava:7b
#
# llava:7b is extremely lightweight and provides superb diagram and table analysis.
# Increase VISION_TIMEOUT if you consistently see timeout errors on slow hardware.

VISION_MODEL              = "llava:7b"
VISION_TIMEOUT            = 300          # seconds
VISION_MAX_IMAGES_PER_PDF = 20            # max figures extracted per PDF
VISION_IMAGE_CACHE_DIR    = "literature_images"   # where extracted figures are saved
VISION_TOP_K              = 2             # reduced from 3 → fewer images = faster
VISION_ENABLED            = True          # set False to disable vision pipeline

# ============================================================================
# VALID REGIONS
# ============================================================================

VALID_REGIONS = [
    'rajasthan', 'maharashtra', 'karnataka', 'tamil nadu',
    'andhra pradesh', 'telangana', 'uttar pradesh', 'bihar',
    'madhya pradesh', 'gujarat', 'west bengal', 'punjab',
    'himachal pradesh', 'uttarakhand', 'goa', 'kerala',
    'haryana', 'assam', 'odisha', 'jharkhand', 'chhattisgarh',
    'arunachal pradesh', 'manipur', 'meghalaya', 'mizoram',
    'nagaland', 'sikkim', 'tripura', 'jammu and kashmir', 'ladakh'
]

# ============================================================================
# OPERATION DEFINITIONS & DISPLAY SETTINGS
# ============================================================================

OPERATIONS = {
    'mean': {
        'description': 'Average soil moisture',
        'primary_stat': 'mean',
        'colormaps': {'map': 'YlGnBu', 'scalar': None},
        'shows_graph': False,
        'shows_map': True,
        'shows_scalar': True,
        'statistical_measures': ['mean', 'std', 'count', 'missing_pct']
    },
    'slope': {
        'description': 'Temporal trend (drying/wetting)',
        'primary_stat': 'slope',
        'colormaps': {'map': 'RdBu', 'temporal': None},
        'shows_graph': True,
        'shows_map': True,
        'shows_scalar': True,
        'statistical_measures': ['slope', 'p_value', 'r_squared', 'total_change']
    },
    'minimum': {
        'description': 'Minimum soil moisture',
        'primary_stat': 'min',
        'colormaps': {'map': 'RdYlGn', 'scalar': None},
        'shows_graph': False,
        'shows_map': True,
        'shows_scalar': True,
        'statistical_measures': ['min', 'count', 'missing_pct']
    },
    'maximum': {
        'description': 'Maximum soil moisture',
        'primary_stat': 'max',
        'colormaps': {'map': 'Blues', 'scalar': None},
        'shows_graph': False,
        'shows_map': True,
        'shows_scalar': True,
        'statistical_measures': ['max', 'count', 'missing_pct']
    },
    'comparison': {
        'description': 'Compare two time periods or two regions',
        'primary_stat': 'diff_mean',
        'colormaps': {'individual': 'YlGnBu', 'difference': 'RdBu'},
        'shows_graph': False,
        'shows_map': True,
        'shows_scalar': True,
        'statistical_measures': ['diff_mean', 'diff_min', 'diff_max']
    }
}

# ============================================================================
# OUTPUT TYPE SETTINGS
# ============================================================================

OUTPUT_TYPES = {
    'scalar': {
        'description': 'Only numerical values and statistics',
        'includes_map': False,
        'includes_graph': False,
        'includes_statistics': True
    },
    'map': {
        'description': 'Only spatial visualization(s)',
        'includes_map': True,
        'includes_graph': False,
        'includes_statistics': False
    },
    'both': {
        'description': 'Both numerical values and spatial visualization',
        'includes_map': True,
        'includes_graph': False,
        'includes_statistics': True
    }
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

FIGURE_SETTINGS = {
    'mean':       {'figsize': (12, 9),  'dpi': 100, 'layout': 'single'},
    'slope':      {'figsize': (18, 7),  'dpi': 100, 'layout': 'dual'},
    'minimum':    {'figsize': (12, 9),  'dpi': 100, 'layout': 'single'},
    'maximum':    {'figsize': (12, 9),  'dpi': 100, 'layout': 'single'},
    'comparison': {'figsize': (20, 7),  'dpi': 100, 'layout': 'triple'}
}

# ============================================================================
# BORDER VISUALIZATION SETTINGS
# ============================================================================

BORDER_SETTINGS = {
    'region_border': {
        'edgecolor': 'black', 'linewidth': 1.5,
        'facecolor': 'none',  'zorder': 5
    },
    'all_states_border': {
        'edgecolor': 'black', 'linewidth': 0.8,
        'facecolor': 'none',  'zorder': 5
    }
}

# ============================================================================
# SEASONS MAPPING
# ============================================================================

SEASONS = {
    'winter':  {'months': [12, 1, 2]},
    'spring':  {'months': [3, 4, 5]},
    'summer':  {'months': [4, 5, 6]},
    'monsoon': {'months': [6, 7, 8, 9]},
    'rainy':   {'months': [6, 7, 8, 9]},
    'autumn':  {'months': [9, 10, 11]},
    'fall':    {'months': [9, 10, 11]}
}

# ============================================================================
# COMPARISON METRIC MAPPING
# ============================================================================

COMPARISON_METRICS = {
    'mean': ['mean', 'average', 'avg'],
    'min':  ['min', 'minimum', 'driest', 'lowest'],
    'max':  ['max', 'maximum', 'wettest', 'highest']
}

# ============================================================================
# DISPLAY TEMPLATES
# ============================================================================

DISPLAY_TEMPLATES = {
    'mean_scalar': """
╔════════════════════════════════════════════════════════════════════════╗
║                    📊 MEAN SOIL MOISTURE ANALYSIS                      
╠════════════════════════════════════════════════════════════════════════╣
║ Region:   {region}
║ Period:   {start_date} to {end_date}
║ Duration: {duration} days
╠════════════════════════════════════════════════════════════════════════╣
║                           📈 RESULT                                    
║
║  Mean Soil Moisture: {mean:.6f} m³/m³
║
║  Data Points:        {count} measurements
║  Missing Data:       {missing_pct:.1f}%
╚════════════════════════════════════════════════════════════════════════╝
""",

    'slope_scalar': """
╔════════════════════════════════════════════════════════════════════════╗
║                   📈 SOIL MOISTURE TREND ANALYSIS                      
╠════════════════════════════════════════════════════════════════════════╣
║ Region:   {region}
║ Period:   {start_date} to {end_date}
║ Duration: {duration} days
╠════════════════════════════════════════════════════════════════════════╣
║                        🔍 TREND STATISTICS                             
║
║  Trend Direction:   {direction}
║  Slope:             {slope:.8f} m³/m³/day
║  R-squared:         {r_squared:.6f}
║  P-value:           {p_value:.6f}
║  Significance:      {significance}
║
║  Total Change:      {total_change:.6f} m³/m³ over the period
╠════════════════════════════════════════════════════════════════════════╣
║ 📍 INTERPRETATION:
║    • Positive slope → Soil getting WETTER (Moisture increasing)
║    • Negative slope → Soil getting DRIER  (Moisture decreasing)
║    • P-value < 0.05 → Trend is STATISTICALLY SIGNIFICANT
║    • P-value ≥ 0.05 → Trend may be DUE TO CHANCE
╚════════════════════════════════════════════════════════════════════════╝
""",

    'minimum_scalar': """
╔════════════════════════════════════════════════════════════════════════╗
║                  📉 MINIMUM SOIL MOISTURE ANALYSIS                     
╠════════════════════════════════════════════════════════════════════════╣
║ Region:   {region}
║ Period:   {start_date} to {end_date}
║ Duration: {duration} days
╠════════════════════════════════════════════════════════════════════════╣
║                           📈 RESULT                                    
║
║  Minimum Soil Moisture: {min:.6f} m³/m³  ⚠️  DRIEST VALUE
║
║  Data Points:           {count} measurements
║  Missing Data:          {missing_pct:.1f}%
╚════════════════════════════════════════════════════════════════════════╝
""",

    'maximum_scalar': """
╔════════════════════════════════════════════════════════════════════════╗
║                  📈 MAXIMUM SOIL MOISTURE ANALYSIS                     
╠════════════════════════════════════════════════════════════════════════╣
║ Region:   {region}
║ Period:   {start_date} to {end_date}
║ Duration: {duration} days
╠════════════════════════════════════════════════════════════════════════╣
║                           📈 RESULT                                    
║
║  Maximum Soil Moisture: {max:.6f} m³/m³  ⭐ WETTEST VALUE
║
║  Data Points:           {count} measurements
║  Missing Data:          {missing_pct:.1f}%
╚════════════════════════════════════════════════════════════════════════╝
""",

    'comparison_time_scalar': """
╔════════════════════════════════════════════════════════════════════════╗
║              🔄 SOIL MOISTURE COMPARISON ANALYSIS                      
╠════════════════════════════════════════════════════════════════════════╣
║ Region:    {region}
║ Metric:    {metric_label}
╠════════════════════════════════════════════════════════════════════════╣
║  Period 1: {period1_label}
║  {metric_label}: {value1:.6f} m³/m³
║
║  Period 2: {period2_label}
║  {metric_label}: {value2:.6f} m³/m³
╠════════════════════════════════════════════════════════════════════════╣
║  Difference (Period 2 − Period 1): {diff:.6f} m³/m³  {diff_direction}
╚════════════════════════════════════════════════════════════════════════╝
""",

    'comparison_region_scalar': """
╔════════════════════════════════════════════════════════════════════════╗
║              🔄 SOIL MOISTURE REGIONAL COMPARISON                      
╠════════════════════════════════════════════════════════════════════════╣
║ Period:    {start_date} to {end_date}
║ Metric:    {metric_label}
╠════════════════════════════════════════════════════════════════════════╣
║  Region 1: {region1}
║  {metric_label}: {value1:.6f} m³/m³
║
║  Region 2: {region2}
║  {metric_label}: {value2:.6f} m³/m³
╠════════════════════════════════════════════════════════════════════════╣
║  Difference ({region2} − {region1}): {diff:.6f} m³/m³  {diff_direction}
╚════════════════════════════════════════════════════════════════════════╝
"""
}

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'region_not_found':          "Region '{region}' not found. Available: {available}",
    'invalid_date':              "Invalid date format. Use YYYY-MM-DD, DD-MM-YYYY (e.g. 01-06-2020 for June 1, 2020), or Month Year.",
    'invalid_operation':         "Unknown operation. Use: mean, slope, minimum, maximum, comparison",
    'no_data':                   "No data for {region} from {start_date} to {end_date}",
    'date_range_invalid':        "Start date cannot be after end date",
    'unclear_query':             "Query is unclear. Please specify region (e.g. India or a state), dates (e.g. 01-06-2020), and operation.",
    'too_short_query':           "Query too short. Please provide more details",
    'ollama_error':              "Ollama error. Using fallback extraction",
    'ollama_timeout':            "Ollama request timed out. Using fallback extraction",
    'comparison_missing_second': "Comparison requires two time periods or two regions. Please specify both.",
    'vision_not_enabled':        "Vision pipeline is disabled. Set VISION_ENABLED=True in Config.py.",
    'vision_no_images':          "No figures/images have been extracted from the loaded literature.",
    'vision_model_missing':      "Vision model '{model}' not found. Run: ollama pull {model}",
}

# ============================================================================
# SUCCESS/INFO MESSAGES
# ============================================================================

SUCCESS_MESSAGES = {
    'analysis_complete':   "✅ Analysis complete!",
    'check_visualization': "📊 visualization ready!",
    'engine_ready':        "✅ Engine loaded successfully!",
    'ollama_ready':        "🤖 Ollama Agent ready!",
    'vision_ready':        "🖼️  Vision pipeline ready (llava:7b)!",
}

# ============================================================================
# ANALYSIS FLAGS
# ============================================================================

ANALYSIS_FLAGS = {
    'show_missing_date_report': True,
    'show_data_quality':        True,
    'show_spatial_statistics':  True,
    'save_visualization':       True,
    'save_raw_data':            False,
    'verbose_mode':             False,
    'draw_borders':             True
}

# ============================================================================
# LITERATURE / RAG CONFIGURATION
# ============================================================================

LITERATURE_DIR        = "literature"
LITERATURE_INDEX_PATH = "literature_index.json"
LITERATURE_TOP_K      = 2             # reduced from 3 → fewer chunks retrieved = faster LLM prompts

# ============================================================================
# MODEL GENERATION SETTINGS
# ============================================================================

# Deterministic routing/classification tasks
INTENT_TEMP            = 0
QUERY_CLASSIFIER_TEMP  = 0
PATCH_TEMP             = 0

# Structured analytical tasks
AGENT_TEMP             = 0.1
ENGINE_TEMP            = 0.2
LITERATURE_MANAGER_TEMP = 0.1

# Natural language explanation tasks
LITERATURE_QA_TEMP     = 0.3
SUMMARY_TEMP           = 0.4

# Sampling controls
STRICT_TOP_P           = 0.1
NORMAL_TOP_P           = 0.3

# ============================================================================
# GOOGLE EARTH ENGINE CONFIGURATION
# ============================================================================
# The default Google Cloud Project ID to use for Earth Engine initialization.
# New users must change this to their own GEE-enabled GCP Project ID.
GEE_PROJECT_ID         = "soil-moisture-agent"