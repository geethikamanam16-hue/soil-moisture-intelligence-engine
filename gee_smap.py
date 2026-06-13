"""
gee_smap.py
===========
Google Earth Engine handler for large-scale SMAP 9km soil moisture processing.
Processes NASA SMAP 9km data directly in the GEE cloud — no local HDF5
downloads required.

Dataset: NASA/SMAP/SPL3SMP_E/005 (9 km EASE-Grid)
Bands:
  soil_moisture_am — Surface Soil Moisture AM Pass, m³/m³
  soil_moisture_pm — Surface Soil Moisture PM Pass, m³/m³

Install: pip install earthengine-api
Auth:    earthengine authenticate
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd

# ── Indian state bounding boxes  [W, S, E, N] ────────────────────────────────
INDIA_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "India":               (68.0,  6.0,  98.0, 37.5),
    "Andhra Pradesh":      (76.8, 12.6,  84.8, 19.9),
    "Arunachal Pradesh":   (91.5, 26.5,  97.4, 29.5),
    "Assam":               (89.7, 24.1,  96.0, 27.9),
    "Bihar":               (83.3, 24.3,  88.3, 27.5),
    "Chhattisgarh":        (80.3, 17.8,  84.4, 24.1),
    "Goa":                 (73.9, 14.9,  74.4, 15.8),
    "Gujarat":             (68.2, 20.1,  74.5, 24.7),
    "Haryana":             (74.5, 27.7,  77.6, 30.9),
    "Himachal Pradesh":    (75.6, 30.4,  79.0, 33.2),
    "Jharkhand":           (83.3, 21.9,  87.9, 25.3),
    "Karnataka":           (74.0, 11.5,  78.6, 18.5),
    "Kerala":              (74.9,  8.2,  77.4, 12.8),
    "Madhya Pradesh":      (74.0, 21.1,  82.8, 26.9),
    "Maharashtra":         (72.6, 15.6,  80.9, 22.0),
    "Manipur":             (93.0, 23.8,  94.8, 25.7),
    "Meghalaya":           (89.8, 25.0,  92.8, 26.1),
    "Mizoram":             (92.3, 21.9,  93.5, 24.5),
    "Nagaland":            (93.3, 25.2,  95.2, 27.0),
    "Odisha":              (81.4, 17.8,  87.5, 22.6),
    "Punjab":              (73.9, 29.5,  76.9, 32.5),
    "Rajasthan":           (69.5, 23.0,  78.3, 30.2),
    "Sikkim":              (88.0, 27.1,  88.9, 28.1),
    "Tamil Nadu":          (76.2,  8.0,  80.4, 13.6),
    "Telangana":           (77.2, 15.8,  81.4, 19.9),
    "Tripura":             (91.1, 22.9,  92.3, 24.5),
    "Uttar Pradesh":       (77.1, 23.9,  84.6, 30.4),
    "Uttarakhand":         (77.6, 28.7,  81.1, 31.5),
    "West Bengal":         (85.8, 21.5,  89.9, 27.2),
    "Jammu & Kashmir":     (73.9, 32.3,  80.4, 37.1),
    "Ladakh":              (75.9, 32.0,  79.5, 35.5),
    "Delhi":               (76.8, 28.4,  77.4, 28.9),
    "Chandigarh":          (76.7, 30.6,  76.9, 30.8),
}

BAND_LABELS: dict[str, str] = {
    "soil_moisture_am": "Soil Moisture AM Pass — m³/m³",
    "soil_moisture_pm": "Soil Moisture PM Pass — m³/m³",
}

GEE_COLLECTION = "NASA/SMAP/SPL3SMP_E/005"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def initialize_ee(project_id: str) -> tuple[bool, str]:
    if not project_id or not project_id.strip():
        return False, "❌ Project ID is empty."
    try:
        import ee
    except ImportError:
        return False, "❌ `earthengine-api` not installed. Run: `pip install earthengine-api`"
    try:
        ee.Initialize(project=project_id.strip())
        ee.ImageCollection(GEE_COLLECTION).filterDate("2020-01-01", "2020-01-03").limit(1).size().getInfo()
        return True, f"✅ Earth Engine initialised with project `{project_id.strip()}`"
    except Exception as exc:
        return False, (
            f"❌ Initialisation failed: {exc}\n\n"
            "**Checklist:**\n"
            "1. Run `earthengine authenticate` in your terminal\n"
            "2. Ensure the project ID matches a GEE-enabled Cloud project\n"
            "3. Accept the GEE Terms of Service at https://code.earthengine.google.com"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Time-series  (scalar daily mean per region)
# ─────────────────────────────────────────────────────────────────────────────

def get_smap_timeseries_gee(
    start_date:  str,
    end_date:    str,
    region_name: str,
    band:        str = "ssm",
    scale:       int = 9_000,
) -> tuple[Optional[pd.DataFrame], str]:
    try:
        import ee
    except ImportError:
        return None, "earthengine-api not installed."

    if region_name not in INDIA_REGIONS:
        return None, f"Region '{region_name}' not found."
    if band not in BAND_LABELS:
        return None, f"Band '{band}' invalid."

    try:
        _s = datetime.date.fromisoformat(start_date)
        _e = datetime.date.fromisoformat(end_date)
        edate_str = (_e + datetime.timedelta(days=1)).isoformat()
    except ValueError as exc:
        return None, f"Invalid date: {exc}"
    if _s > _e:
        return None, "Start date must be before or equal to end date."
    if (_e - _s).days > 3650:
        return None, "Date range > 10 years. Narrow your selection."

    bbox   = INDIA_REGIONS[region_name]
    region = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])
    col    = (ee.ImageCollection(GEE_COLLECTION)
              .filterDate(start_date, edate_str)
              .filterBounds(region)
              .select(band))

    def image_mean(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region,
            scale=scale, maxPixels=1e9, bestEffort=True)
        return ee.Feature(None, stats.set("date", img.date().format("YYYY-MM-dd")))

    feat_list = col.map(image_mean).toList(col.size())
    try:
        raw = feat_list.getInfo()
    except Exception as exc:
        return None, f"GEE server error: {exc}"

    records = []
    for feat in raw:
        props = feat.get("properties", {})
        d, v  = props.get("date"), props.get(band)
        if d and v is not None:
            records.append({"Date": d, band: round(float(v), 6)})

    if not records:
        return None, (
            f"No data for '{region_name}' between {start_date} and {end_date}. "
            "SMAP covers April 2015 onwards."
        )

    df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df.rename(columns={band: BAND_LABELS[band]}, inplace=True)
    return df, ""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-band time-series
# ─────────────────────────────────────────────────────────────────────────────

def get_smap_multiband_gee(
    start_date:  str,
    end_date:    str,
    region_name: str,
    bands:       list[str] | None = None,
    scale:       int = 9_000,
) -> tuple[Optional[pd.DataFrame], str]:
    if bands is None:
        bands = list(BAND_LABELS.keys())
    try:
        import ee
    except ImportError:
        return None, "earthengine-api not installed."
    if region_name not in INDIA_REGIONS:
        return None, f"Region '{region_name}' not found."

    bbox   = INDIA_REGIONS[region_name]
    region = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])
    col    = (ee.ImageCollection(GEE_COLLECTION)
              .filterDate(start_date, end_date)
              .filterBounds(region)
              .select(bands))

    def image_stats(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region,
            scale=scale, maxPixels=1e9, bestEffort=True)
        return ee.Feature(None, stats.set("date", img.date().format("YYYY-MM-dd")))

    feat_list = col.map(image_stats).toList(col.size())
    try:
        raw = feat_list.getInfo()
    except Exception as exc:
        return None, f"GEE server error: {exc}"

    records = []
    for feat in raw:
        props = feat.get("properties", {})
        d = props.get("date")
        if not d:
            continue
        row = {"Date": d}
        for b in bands:
            v = props.get(b)
            if v is not None:
                row[BAND_LABELS[b]] = round(float(v), 6)
        records.append(row)

    if not records:
        return None, "No data returned."

    df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df, ""


# ─────────────────────────────────────────────────────────────────────────────
# 4. Spatial grid fetch  (GeoTIFF → numpy)
# ─────────────────────────────────────────────────────────────────────────────

def get_smap_spatial_grid_gee(
    start_date:  str,
    end_date:    str,
    region_name: str,
    band:        str = "ssm",
    scale:       int = 9_000,
) -> tuple[Optional[dict], str]:
    """
    Fetch the period-mean SMAP spatial grid as numpy arrays.

    Returns
    -------
    (result_dict, error_str)

    result_dict keys:
        smap_grid  : np.ndarray shape (n_lats, n_lons) — ASCENDING lat order
        lats       : np.ndarray 1-D ASCENDING (south → north)
        lons       : np.ndarray 1-D ASCENDING (west  → east)
        bbox       : tuple (W, S, E, N)
        band       : str
        n_images   : int
        smap_mean  : float
        smap_min   : float
        smap_max   : float
        region     : str
        start_date : str
        end_date   : str
    """
    try:
        import ee
    except ImportError:
        return None, "earthengine-api not installed."

    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError:
        return None, (
            "`rasterio` is not installed.\n"
            "Run: `pip install rasterio`"
        )

    import requests

    if region_name not in INDIA_REGIONS:
        return None, f"Region '{region_name}' not found."
    if band not in BAND_LABELS:
        return None, f"Band '{band}' invalid."

    try:
        _e = datetime.date.fromisoformat(end_date)
        edate_str = (_e + datetime.timedelta(days=1)).isoformat()
    except Exception:
        edate_str = end_date

    bbox   = INDIA_REGIONS[region_name]
    region = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])

    col = (
        ee.ImageCollection(GEE_COLLECTION)
        .filterDate(start_date, edate_str)
        .filterBounds(region)
        .select(band)
    )

    n_images = col.size().getInfo()
    if n_images == 0:
        return None, (
            f"No SMAP images found for '{region_name}' between {start_date} and {end_date}. "
            "SMAP covers April 2015 onwards."
        )

    mean_img = col.mean().clip(region)

    try:
        url = mean_img.getDownloadURL({
            "bands"  : [band],
            "region" : region,
            "scale"  : scale,
            "format" : "GEO_TIFF",
            "crs"    : "EPSG:4326",
        })
    except Exception as exc:
        return None, f"Could not get download URL from GEE: {exc}"

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        tif_bytes = resp.content
    except requests.exceptions.Timeout:
        return None, "GeoTIFF download timed out (> 120 s)."
    except Exception as exc:
        return None, f"GeoTIFF download failed: {exc}"

    try:
        with MemoryFile(tif_bytes) as memfile:
            with memfile.open() as ds:
                raw_grid  = ds.read(1).astype(np.float32)
                transform = ds.transform
                nrows, ncols = raw_grid.shape

                lons_raw = np.array([transform.c + (j + 0.5) * transform.a
                                     for j in range(ncols)])
                lats_raw = np.array([transform.f + (i + 0.5) * transform.e
                                     for i in range(nrows)])
    except Exception as exc:
        return None, f"Could not read GeoTIFF: {exc}"

    # Normalise to ASCENDING lat order (south → north)
    if lats_raw[0] > lats_raw[-1]:
        lats_raw = lats_raw[::-1]
        raw_grid = raw_grid[::-1, :]

    if lons_raw[0] > lons_raw[-1]:
        lons_raw = lons_raw[::-1]
        raw_grid = raw_grid[:, ::-1]

    assert raw_grid.shape == (len(lats_raw), len(lons_raw)), (
        f"Shape mismatch after flip: grid={raw_grid.shape}, "
        f"lats={len(lats_raw)}, lons={len(lons_raw)}"
    )

    raw_grid[raw_grid <= 0]   = np.nan
    raw_grid[raw_grid > 1000] = np.nan

    valid = raw_grid[np.isfinite(raw_grid)]
    if valid.size == 0:
        return None, (
            "All pixels are masked — no valid SMAP data in this period/region."
        )

    return {
        "smap_grid" : raw_grid,
        "lats"      : lats_raw,
        "lons"      : lons_raw,
        "bbox"      : bbox,
        "band"      : band,
        "n_images"  : n_images,
        "smap_mean" : float(np.nanmean(valid)),
        "smap_min"  : float(np.nanmin(valid)),
        "smap_max"  : float(np.nanmax(valid)),
        "region"    : region_name,
        "start_date": start_date,
        "end_date"  : end_date,
    }, ""


# ─────────────────────────────────────────────────────────────────────────────
# 5. 3-panel validation map
#    Panel 1: AMSR Mean | Panel 2: SMAP Mean | Panel 3: Bias (AMSR − SMAP)
# ─────────────────────────────────────────────────────────────────────────────

def generate_gee_comparison_plot(
    gee_result:  dict,
    amsr_da,
    region_name: str,
    output_path: str = "gee_smap_comparison.png",
) -> tuple[bool, dict]:
    """
    3-panel figure matching reference image:
      Panel 1 — AMSR Mean        (YlGnBu)
      Panel 2 — SMAP Mean        (YlGnBu)
      Panel 3 — Bias (AMSR−SMAP) (RdBu_r: blue=pos, red=neg)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr
    from scipy import stats as sp_stats
    import os

    # Unpack SMAP result
    smap_grid = gee_result["smap_grid"]
    lats      = gee_result["lats"]
    lons      = gee_result["lons"]
    band      = gee_result["band"]
    unit      = "%" if band == "smp" else "m³/m³"

    n_lats, n_lons = smap_grid.shape

    # Regrid AMSR onto the SMAP lat/lon grid
    amsr_regridded = None

    if amsr_da is not None:
        try:
            if "time" in amsr_da.dims:
                amsr_2d = amsr_da.mean(dim="time")
            else:
                amsr_2d = amsr_da

            rename = {}
            if "x" in amsr_2d.dims and "lon" not in amsr_2d.dims:
                rename["x"] = "lon"
            if "y" in amsr_2d.dims and "lat" not in amsr_2d.dims:
                rename["y"] = "lat"
            if rename:
                amsr_2d = amsr_2d.rename(rename)

            amsr_2d = amsr_2d.transpose("lat", "lon").sortby("lat").sortby("lon")

            amsr_interp = amsr_2d.interp(
                lat    = xr.DataArray(lats, dims="lat"),
                lon    = xr.DataArray(lons, dims="lon"),
                method = "linear",
            )

            amsr_regridded = amsr_interp.values.astype(np.float32)

            if amsr_regridded.shape != smap_grid.shape:
                print(
                    f"⚠️  AMSR shape {amsr_regridded.shape} != "
                    f"SMAP shape {smap_grid.shape} after interp — skipping AMSR."
                )
                amsr_regridded = None

        except Exception as exc:
            print(f"⚠️  AMSR regrid failed: {exc}")
            amsr_regridded = None

    # Validation metrics
    metrics = {
        "region"     : region_name,
        "band"       : band,
        "n_images"   : gee_result["n_images"],
        "smap_mean"  : gee_result["smap_mean"],
        "amsr_mean"  : None,
        "bias"       : None,
        "rmse"       : None,
        "ubrmse"     : None,
        "correlation": None,
        "r_squared"  : None,
        "n"          : 0,
    }

    bias_grid = None

    if amsr_regridded is not None:
        mask = np.isfinite(smap_grid) & np.isfinite(amsr_regridded)
        n    = int(mask.sum())
        metrics["n"] = n
        if n >= 3:
            s_vals  = smap_grid[mask]
            a_vals  = amsr_regridded[mask]
            bias    = float(np.mean(a_vals - s_vals))
            rmse    = float(np.sqrt(np.mean((a_vals - s_vals) ** 2)))
            ubrmse  = float(np.sqrt(np.mean(((a_vals - bias) - s_vals) ** 2)))
            r, pval = sp_stats.pearsonr(s_vals, a_vals)
            metrics.update({
                "amsr_mean"  : float(np.mean(a_vals)),
                "bias"       : round(bias,   6),
                "rmse"       : round(rmse,   6),
                "ubrmse"     : round(ubrmse, 6),
                "correlation": round(float(r), 4),
                "r_squared"  : round(float(r ** 2), 4),
                "p_value"    : round(float(pval), 6),
            })
            bias_grid = np.where(mask, amsr_regridded - smap_grid, np.nan).astype(np.float32)

    # Shared colour limits
    valid_smap  = smap_grid[np.isfinite(smap_grid)]
    vmax_sm     = float(np.nanpercentile(valid_smap, 98)) if valid_smap.size else 0.5
    vmin_sm     = 0.0

    if amsr_regridded is not None:
        valid_am    = amsr_regridded[np.isfinite(amsr_regridded)]
        vmax_am     = float(np.nanpercentile(valid_am, 98)) if valid_am.size else vmax_sm
        vmax_shared = max(vmax_sm, vmax_am)
    else:
        vmax_shared = vmax_sm

    # Boundary overlay helper
    def _add_boundaries(ax):
        try:
            import geopandas as gpd
            for folder in [r"cache\shapefiles", "cache/shapefiles"]:
                if os.path.isdir(folder):
                    shps = [f for f in os.listdir(folder) if f.endswith(".shp")]
                    if shps:
                        gdf = gpd.read_file(os.path.join(folder, shps[0]))
                        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                            gdf = gdf.to_crs("EPSG:4326")
                        gdf.boundary.plot(
                            ax=ax, color="black", linewidth=0.6, alpha=0.8
                        )
                        return
        except Exception:
            pass

    # Single-panel plot helper
    def _plot_panel(ax, grid_2d, lats_1d, lons_1d, title,
                    cmap, vmin, vmax, cbar_label):
        def _edges(centres):
            d = np.diff(centres)
            return np.concatenate([
                [centres[0]  - d[0]  / 2],
                centres[:-1] + d      / 2,
                [centres[-1] + d[-1] / 2],
            ])

        lon_edges = _edges(lons_1d)
        lat_edges = _edges(lats_1d)
        LON, LAT  = np.meshgrid(lon_edges, lat_edges)

        masked = np.ma.masked_invalid(grid_2d)

        pcm = ax.pcolormesh(
            LON, LAT, masked,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading="flat",
        )
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, extend="both", pad=0.02)
        cbar.set_label(cbar_label, fontsize=8)

        _add_boundaries(ax)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude",  fontsize=8)
        ax.set_xlim(lons_1d.min() - 0.3, lons_1d.max() + 0.3)
        ax.set_ylim(lats_1d.min() - 0.3, lats_1d.max() + 0.3)
        ax.tick_params(labelsize=7)

    # Figure layout
    has_amsr = amsr_regridded is not None
    has_bias = bias_grid is not None
    n_panels = 3 if (has_amsr and has_bias) else (2 if has_amsr else 1)

    period_label = f"{gee_result['start_date']} → {gee_result['end_date']}"

    try:
        # ── Reduced size to save space ──
        fig, axes = plt.subplots(
            1, n_panels,
            figsize            = (4 * n_panels, 3.5),
            constrained_layout = True,
        )
        if n_panels == 1:
            axes = [axes]

        idx = 0

        # Panel 1 — AMSR Mean (left)
        if has_amsr:
            _plot_panel(
                axes[idx],
                grid_2d    = amsr_regridded,
                lats_1d    = lats,
                lons_1d    = lons,
                title      = f"AMSR Mean: {region_name}",
                cmap       = "YlGnBu",
                vmin       = vmin_sm,
                vmax       = vmax_shared,
                cbar_label = f"Soil Moisture ({unit})",
            )
            idx += 1

        # Panel 2 — SMAP Mean (centre) — always show the full original SMAP grid
        smap_display_grid = smap_grid

        _plot_panel(
            axes[idx],
            grid_2d    = smap_display_grid,
            lats_1d    = lats,
            lons_1d    = lons,
            title      = f"SMAP Mean: {region_name}",
            cmap       = "YlGnBu",
            vmin       = vmin_sm,
            vmax       = vmax_shared,
            cbar_label = f"Soil Moisture ({unit})",
        )
        idx += 1

        # Panel 3 — Bias (right)
        if has_bias:
            valid_b = bias_grid[np.isfinite(bias_grid)]
            bmax    = float(np.nanpercentile(np.abs(valid_b), 95)) if valid_b.size else 0.2
            bmax    = max(bmax, 0.01)
            _plot_panel(
                axes[idx],
                grid_2d    = bias_grid,
                lats_1d    = lats,
                lons_1d    = lons,
                title      = f"Bias (AMSR - SMAP): {region_name}",
                cmap       = "RdBu_r",
                vmin       = -bmax,
                vmax       =  bmax,
                cbar_label = f"Bias ({unit})",
            )

        fig.suptitle(period_label, fontsize=9, color="dimgray", y=1.01)

        # ── Lower DPI to reduce image size ──────────────
        plt.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"✅ GEE comparison plot saved → {output_path}")
        return True, metrics

    except Exception as exc:
        import traceback
        print(f"⚠️  Plot error: {exc}")
        traceback.print_exc()
        # Record the error for UI feedback
        if isinstance(metrics, dict):
            metrics["plot_error"] = str(exc)
        else:
            metrics = {"plot_error": str(exc)}
        plt.close("all")
        return False, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6. Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_collection_date_range() -> tuple[str, str]:
    return "2015-04-01", "present"


def list_regions() -> list[str]:
    regions = sorted(k for k in INDIA_REGIONS if k != "India")
    return ["India"] + regions