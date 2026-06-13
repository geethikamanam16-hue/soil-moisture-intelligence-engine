import xarray as xr
import os
import zipfile

# ==========================================
# PORTABLE PATH RESOLUTION
# ==========================================

# This file lives at: <project_root>/cloud/dataset_manager.py
# Project root is one level up from the cloud/ directory.
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# ==========================================
# CACHE FOLDER
# ==========================================

CACHE_FOLDER = os.path.join(_PROJECT_ROOT, "cache", "zarr")

os.makedirs(CACHE_FOLDER, exist_ok=True)

# ==========================================
# LOAD YEAR FROM LOCAL CACHE
# ==========================================

def download_year(year):
    """
    Previously fetched from Google Drive.
    Now reads directly from the local cache.

    Priority:
      1. If the extracted <year>_v2.zarr folder already exists → use it.
      2. If only the <year>_v2.zarr.zip exists → extract it first, then use it.
      3. Neither found → raise FileNotFoundError with a helpful message.
    """
    zarr_folder = os.path.join(CACHE_FOLDER, f"{year}_v2.zarr")

    # Already extracted — nothing to do
    if os.path.exists(zarr_folder):
        print(f"{year} already cached.")
        return zarr_folder

    # Only the zip is present — extract it locally
    zip_path = os.path.join(CACHE_FOLDER, f"{year}_v2.zarr.zip")
    if os.path.exists(zip_path):
        print(f"Extracting {year}_v2.zarr.zip from local cache...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(CACHE_FOLDER)
        print("Extraction complete.")
        return zarr_folder

    raise FileNotFoundError(
        f"[dataset_manager] Neither '{zarr_folder}' nor '{zip_path}' found locally.\n"
        f"Please place '{year}_v2.zarr' or '{year}_v2.zarr.zip' in:\n  {CACHE_FOLDER}"
    )

# ==========================================
# LOAD YEAR DATASET
# ==========================================

def load_year_dataset(year):

    zarr_path = download_year(year)

    print(f"\nLoading {year} dataset...")

    ds = xr.open_zarr(
        zarr_path,
        consolidated=True
    )

    print("Dataset loaded successfully.")

    return ds

# ==========================================
# LOAD MULTIPLE YEARS
# ==========================================

def load_multiple_years(start_year, end_year):

    datasets = []

    for year in range(start_year, end_year + 1):

        print(f"\nProcessing year {year}...")

        ds = load_year_dataset(str(year))

        datasets.append(ds)

    print("\nCombining datasets...")

    combined = xr.concat(
        datasets,
        dim="time"
    )

    print("Combined dataset ready.")

    return combined

def get_full_dataset():
    """
    Previously synced from Google Drive.
    Now scans the local cache folder directly to discover available years.
    """
    print("\nScanning local Zarr cache...")

    entries = os.listdir(CACHE_FOLDER)

    years = set()

    for entry in entries:
        # Match extracted zarr directories: YYYY_v2.zarr
        if entry.endswith("_v2.zarr") and os.path.isdir(os.path.join(CACHE_FOLDER, entry)):
            year_str = entry.split("_")[0]
            if year_str.isdigit():
                years.add(int(year_str))
        # Also match zips in case extraction hasn't happened yet: YYYY_v2.zarr.zip
        elif entry.endswith("_v2.zarr.zip"):
            year_str = entry.split("_")[0]
            if year_str.isdigit():
                years.add(int(year_str))

    if not years:
        raise FileNotFoundError(
            f"[dataset_manager] No Zarr datasets found in local cache.\n"
            f"Expected files like '2020_v2.zarr' or '2020_v2.zarr.zip' in:\n  {CACHE_FOLDER}"
        )

    years_sorted = sorted(years)
    start_year   = years_sorted[0]
    end_year     = years_sorted[-1]

    print(f"Found years: {years_sorted[0]} – {years_sorted[-1]}")

    return load_multiple_years(start_year, end_year)

# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    ds = load_multiple_years(2010, 2012)

    print("\n")
    print(ds)