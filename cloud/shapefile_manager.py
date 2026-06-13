import geopandas as gpd
import os

# ==========================================
# PORTABLE PATH RESOLUTION
# ==========================================

# This file lives at: <project_root>/cloud/shapefile_manager.py
# Project root is one level up from the cloud/ directory.
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# ==========================================
# CACHE FOLDER
# ==========================================

CACHE_FOLDER = os.path.join(_PROJECT_ROOT, "cache", "shapefiles")

os.makedirs(CACHE_FOLDER, exist_ok=True)

# ==========================================
# GET SHAPEFILE PATH FROM LOCAL CACHE
# ==========================================

def get_shapefile_path():
    """
    Previously synced from Google Drive.
    Now scans the local shapefiles cache for any .shp file and returns its path.
    """
    print("\nLoading shapefile from local cache...")

    shp_filename = None

    for entry in os.listdir(CACHE_FOLDER):
        if entry.endswith(".shp"):
            shp_filename = entry
            break   # use the first .shp found

    if shp_filename is None:
        raise FileNotFoundError(
            f"[shapefile_manager] No .shp file found in local cache.\n"
            f"Please place the shapefile components (.shp, .shx, .dbf, .prj, .cpg) in:\n"
            f"  {CACHE_FOLDER}"
        )

    local_shp_path = os.path.join(CACHE_FOLDER, shp_filename)
    print(f"Shapefile found: {shp_filename}")
    return local_shp_path

if __name__ == "__main__":
    local_shp_path = get_shapefile_path()
    print("\nLoading shapefile...\n")
    gdf = gpd.read_file(local_shp_path)
    print(gdf.head())
    print("\nCRS:")
    print(gdf.crs)
    print("\nShapefile loaded successfully.")