"""SM_Engine - Optimized Soil Moisture Analysis Engine

Handles operation-specific analysis with flexible output (scalar, map, or both).
"""

import matplotlib
matplotlib.use('Agg')   # Must be set BEFORE importing pyplot

import math
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
import rioxarray
from shapely.geometry import mapping
from datetime import datetime
import warnings
import traceback
warnings.filterwarnings('ignore')

from Config import (FIGURE_SETTINGS, OPERATIONS,
                    BORDER_SETTINGS, OUTPUT_PATH)
from utils import OutputFormatter, DateAnalyzer

# Normalise common spelling variants → exact shapefile STATE values
_STATE_SPELLING_NORM = {
    "chhattisgarh": "chhatisgarh",   # shapefile uses single-t spelling
    "chattisgarh":  "chhatisgarh",
}


# ============================================================================
# BORDER HELPER
# ============================================================================

def _draw_borders(ax, gdf, region_name=None):
    """Overlay actual shapefile borders in black on a map axes."""
    try:
        bs_all    = BORDER_SETTINGS['all_states_border']
        bs_region = BORDER_SETTINGS['region_border']

        gdf.boundary.plot(
            ax=ax, edgecolor='black',
            linewidth=bs_all['linewidth'],
            facecolor='none', zorder=bs_all['zorder']
        )

        if region_name and region_name.lower() != 'india':
            region_gdf = gdf[
                gdf['STATE'].str.strip().str.lower() == region_name.lower()
            ]
            if not region_gdf.empty:
                try:
                    region_gdf.boundary.plot(
                        ax=ax, edgecolor='black',
                        linewidth=bs_region['linewidth'],
                        facecolor='none', zorder=bs_region['zorder']
                    )
                except Exception:
                    pass

    except Exception as e:
        print(f"⚠️  Border drawing note: {e}")


# ============================================================================
# SLOPE MAP HELPER
# ============================================================================

def _compute_spatial_slope(data_array):
    """Compute a pixel-wise temporal slope map from a DataArray."""
    def get_px_slope(y):
        idx = np.arange(len(y))
        m   = ~np.isnan(y)
        return stats.linregress(idx[m], y[m])[0] if sum(m) > 2 else np.nan

    return xr.apply_ufunc(
        get_px_slope, data_array,
        input_core_dims=[['time']], vectorize=True
    )


# ============================================================================
# ENGINE
# ============================================================================

class SM_Engine:
    """
    Optimized Soil Moisture Analysis Engine.

    v2.6 fixes:
      - All comparison types (2-way region, N-way time) use _visualize_n_panel
        so they ALWAYS produce exactly ONE combined output image.
      - Map generation is more robust with full error reporting.

    v2.8 fixes:
      - Multi-period slope/trend queries produce ONE combined image.
      - New execute_analysis_batch() for main.py to collect N ranges together.
    """

    def __init__(self):
        from cloud.dataset_manager import get_full_dataset
        from cloud.shapefile_manager import get_shapefile_path

        print("📂 Loading Zarr datasets from local cache...")
        self.ds = get_full_dataset()

        for d in list(self.ds.dims):
            if 'lat' in d.lower():
                self.ds = self.ds.rename({d: 'y'})
            if 'lon' in d.lower():
                self.ds = self.ds.rename({d: 'x'})

        self.ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        if self.ds.rio.crs is None:
            self.ds.rio.write_crs("EPSG:4326", inplace=True)

        print("📍 Loading shapefile from local cache...")
        self.shp_path = get_shapefile_path()
        self.gdf = gpd.read_file(self.shp_path)
        if self.gdf.crs is None or self.gdf.crs.to_epsg() != 4326:
            self.gdf = self.gdf.to_crs("EPSG:4326")

        self.available_regions = (
            self.gdf['STATE'].str.strip().str.lower().unique()
        )
        print(f"✅ Engine ready! Available regions: {len(self.available_regions)}")

    # ------------------------------------------------------------------ #
    # PUBLIC: execute_analysis                                             #
    # ------------------------------------------------------------------ #

    def execute_analysis(self, region, start_date, end_date, operation,
                         output_type='both', comparison_info=None,
                         output_path=None):
        """
        Execute the requested analysis for a SINGLE date range.
        Returns (result_message: str, visualization_created: bool)

        output_path: if provided, the visualization is saved directly to this
                     path (avoids the shutil.copy race condition in app.py).
                     Defaults to OUTPUT_PATH (legacy 'latest_analysis.png').
        """
        _out = output_path or OUTPUT_PATH

        if start_date and len(start_date) == 7:
            start_date += "-01"
        if end_date and len(end_date) == 7:
            end_date += "-28"

        print(f"📊 Processing [{operation}] for [{region}]  "
              f"{start_date} → {end_date} | output_type={output_type} ...")

        if operation == 'comparison':
            return self._handle_comparison(
                region, start_date, end_date, output_type, comparison_info,
                output_path=_out
            )

        subset = self.ds.sel(time=slice(start_date, end_date)).compute()
        clipped, display_region, ok = self._clip_region(subset, region)
        if not ok:
            return clipped, False

        v = list(self.ds.data_vars)[0]

        if clipped.time.size == 0 or int(clipped[v].count()) == 0:
            return f"❌ No data available for {display_region} ({start_date} to {end_date})", False

        if operation == 'mean':
            return self._analyze_mean(
                clipped, v, display_region, region,
                start_date, end_date, output_type, _out)
        elif operation == 'slope':
            return self._analyze_slope(
                clipped, v, display_region, region,
                start_date, end_date, output_type, _out)
        elif operation == 'minimum':
            return self._analyze_minimum(
                clipped, v, display_region, region,
                start_date, end_date, output_type, _out)
        elif operation == 'maximum':
            return self._analyze_maximum(
                clipped, v, display_region, region,
                start_date, end_date, output_type, _out)
        else:
            return f"❌ Unknown operation: {operation}", False

    # ------------------------------------------------------------------ #
    # PUBLIC: execute_analysis_batch  (v2.8 — NEW)                        #
    # ------------------------------------------------------------------ #

    def execute_analysis_batch(self, region, date_ranges, operation,
                               output_type='both'):
        """
        Execute the same non-comparison operation over N date ranges and
        produce ONE combined output image (for slope/trend multi-period).

        Parameters
        ----------
        region      : str
        date_ranges : list of (start_date, end_date) tuples
        operation   : 'slope' | 'mean' | 'minimum' | 'maximum'
        output_type : 'scalar' | 'map' | 'both'

        Returns
        -------
        list of (result_message: str, viz_created: bool) — one per range.
        The visualization for ALL ranges is saved once to OUTPUT_PATH.
        """
        if not date_ranges:
            return []

        v = list(self.ds.data_vars)[0]

        # ── Collect per-period data + scalar results ───────────────────
        results        = []   # (msg, False) per period — viz handled jointly
        clipped_list   = []   # clipped DataArrays per period
        label_list     = []   # human-readable label per period
        display_region = region.title()

        for s, e in date_ranges:
            if s and len(s) == 7: s += "-01"
            if e and len(e) == 7: e += "-28"

            subset = self.ds.sel(time=slice(s, e)).compute()
            clipped, disp, ok = self._clip_region(subset, region)
            if not ok:
                results.append((clipped, False))
                clipped_list.append(None)
                label_list.append(f"{s} to {e}")
                continue

            display_region = disp
            clipped_list.append(clipped)
            label_list.append(f"{s}\nto {e}")

            # Scalar result for this period (no map yet)
            msg, _ = self._analyze_single_no_viz(
                clipped, v, disp, region, s, e, operation, output_type
            )
            results.append((msg, False))  # viz_created=False for now

        # ── Render ONE combined image for all periods ──────────────────
        viz_created = False
        if output_type in ['map', 'both']:
            valid_pairs = [
                (c, l) for c, l in zip(clipped_list, label_list) if c is not None
            ]
            if valid_pairs:
                valid_clipped, valid_labels = zip(*valid_pairs)
                if operation == 'slope':
                    viz_created = self._visualize_slope_n_panel(
                        clipped_list   = list(valid_clipped),
                        var_name       = v,
                        labels         = list(valid_labels),
                        region_name    = region,
                        display_region = display_region,
                    )
                else:
                    # For mean / min / max — use existing _visualize_n_panel
                    maps = [self._metric_map(c[v], operation) for c in valid_clipped]
                    viz_created = self._visualize_n_panel(
                        maps         = maps,
                        labels       = list(valid_labels),
                        region_name  = region,
                        region_names = None,
                        suptitle     = (
                            f"{len(valid_labels)}-Period {operation.upper()}: "
                            f"{display_region}"
                        ),
                        metric       = operation,
                    )

        # Mark the last result as viz_created so main.py prints one notice
        if results:
            last_msg, _ = results[-1]
            results[-1] = (last_msg, viz_created)

        return results

    # ── Internal: run analysis WITHOUT generating visualization ───────────

    def _analyze_single_no_viz(self, clipped, var_name, display_region,
                                raw_region, start_date, end_date,
                                operation, output_type):
        """
        Run a single-period analysis and return scalar output only.
        Visualization is suppressed — the caller handles combined rendering.
        """
        # Force scalar so no individual map is saved
        scalar_output_type = 'scalar' if output_type in ('map', 'both') else output_type

        if operation == 'slope':
            return self._analyze_slope(
                clipped, var_name, display_region, raw_region,
                start_date, end_date, scalar_output_type
            )
        elif operation == 'mean':
            return self._analyze_mean(
                clipped, var_name, display_region, raw_region,
                start_date, end_date, scalar_output_type
            )
        elif operation == 'minimum':
            return self._analyze_minimum(
                clipped, var_name, display_region, raw_region,
                start_date, end_date, scalar_output_type
            )
        elif operation == 'maximum':
            return self._analyze_maximum(
                clipped, var_name, display_region, raw_region,
                start_date, end_date, scalar_output_type
            )
        return "❌ Unknown operation.", False

    # ------------------------------------------------------------------ #
    # REGION CLIPPING                                                      #
    # ------------------------------------------------------------------ #

    def _clip_region(self, subset, region):
        """Clip dataset to a region. Returns (clipped, display_name, success)."""
        # Normalise common spelling variants to match shapefile STATE values
        norm_region = _STATE_SPELLING_NORM.get(region.lower(), region.lower())

        if norm_region == 'india':
            clipped = subset.rio.clip(
                self.gdf.geometry.apply(mapping),
                self.gdf.crs, drop=True
            )
            return clipped, "India (All States)", True

        region_gdf = self.gdf[
            self.gdf['STATE'].str.strip().str.lower() == norm_region
        ]
        if region_gdf.empty:
            available = ", ".join(sorted(self.available_regions)[:8])
            return (f"❌ Region '{region}' not found. "
                    f"Available (first 8): {available}...", None, False)

        clipped = subset.rio.clip(
            region_gdf.geometry.apply(mapping),
            self.gdf.crs, drop=True
        )
        return clipped, region.title(), True

    # ================================================================== #
    # SINGLE OPERATION HANDLERS
    # ================================================================== #

    def _analyze_mean(self, clipped, var_name, display_region, raw_region,
                      start_date, end_date, output_type, output_path=None):
        data  = clipped[var_name]
        total = int(data.size)
        null  = int(data.isnull().sum())

        expected_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
        spatial_size = total // max(1, data.time.size) if hasattr(data, 'time') and data.time.size > 0 else 0
        expected_total = expected_days * spatial_size

        missing_days_count = max(0, expected_days - (data.time.size if hasattr(data, 'time') else 1))
        true_null = null + (missing_days_count * spatial_size)
        true_total = expected_total if expected_total > 0 else total

        stats_dict = {
            'mean':        float(data.mean()),
            'count':       int(data.count()),
            'missing_pct': (true_null / true_total * 100) if true_total else 0
        }
        output = OutputFormatter.format_mean(
            stats_dict, output_type, display_region, start_date, end_date)

        viz_created = False
        if output_type in ['map', 'both']:
            viz_created = self._visualize_mean(
                clipped, var_name, display_region, raw_region,
                start_date, end_date, output_path or OUTPUT_PATH)

        return output, viz_created

    def _visualize_mean(self, clipped, var_name, display_region, raw_region,
                        start_date, end_date, output_path=None):
        _out = output_path or OUTPUT_PATH
        try:
            settings  = FIGURE_SETTINGS['mean']
            fig, ax   = plt.subplots(1, 1, figsize=settings['figsize'])
            mean_data = clipped[var_name].mean(dim='time')
            is_single_day = (start_date == end_date)
            cbar_label = 'Soil Moisture (m³/m³)' if is_single_day else 'Mean Soil Moisture (m³/m³)'
            mean_data.plot(ax=ax, x='x', y='y', cmap='YlGnBu',
                           cbar_kwargs={'label': cbar_label})
            _draw_borders(ax, self.gdf, raw_region)

            if is_single_day:
                title_str = f"Soil Moisture: {display_region}\n({start_date})"
            else:
                title_str = f"Mean Soil Moisture: {display_region}\n({start_date} to {end_date})"
            ax.set_title(title_str, fontsize=14, fontweight='bold')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.savefig(_out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Map saved → {_out}")
            return True
        except Exception as e:
            print(f"⚠️  Visualization error (mean): {e}")
            traceback.print_exc()
            plt.close('all')
            return False

    def _analyze_slope(self, clipped, var_name, display_region, raw_region,
                       start_date, end_date, output_type, output_path=None):
        ts       = clipped[var_name].mean(dim=['x', 'y'])
        mask     = ~np.isnan(ts)
        ts_clean = ts.where(mask, drop=True)
        x_idx    = np.arange(len(ts_clean))

        if len(ts_clean) > 1:
            slope, intercept, r_val, p_val, _ = stats.linregress(x_idx, ts_clean.values)
        else:
            slope, intercept, r_val, p_val = 0.0, float(ts_clean.mean() if len(ts_clean) > 0 else 0.0), 0.0, 1.0

        duration  = (datetime.strptime(end_date, '%Y-%m-%d') -
                     datetime.strptime(start_date, '%Y-%m-%d')).days + 1
        total_n   = len(ts)
        missing_n = int(total_n - len(ts_clean))

        stats_dict = {
            'slope':        float(slope),
            'p_value':      float(p_val),
            'r_squared':    float(r_val ** 2),
            'total_change': float(slope * duration),
            'count':        int(len(ts_clean)),
            'missing_pct':  (missing_n / total_n * 100) if total_n else 0
        }

        output = OutputFormatter.format_slope(
            stats_dict, output_type, display_region, start_date, end_date)

        viz_created = False
        if output_type in ['map', 'both']:
            viz_created = self._visualize_slope(
                clipped, var_name, ts_clean, slope, intercept, x_idx,
                display_region, raw_region, start_date, end_date,
                output_path or OUTPUT_PATH)

        return output, viz_created

    def _visualize_slope(self, clipped, var_name, ts_clean, slope, intercept,
                         x_idx, display_region, raw_region, start_date, end_date,
                         output_path=None):
        """Single-period slope visualization (2 panels: spatial map + trend graph)."""
        _out = output_path or OUTPUT_PATH
        try:
            settings = FIGURE_SETTINGS['slope']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=settings['figsize'])

            spatial_slope = _compute_spatial_slope(clipped[var_name])
            spatial_slope.plot(
                ax=ax1, x='x', y='y', cmap='RdBu', center=0,
                cbar_kwargs={'label': 'Slope (m³/m³/day)'})
            _draw_borders(ax1, self.gdf, raw_region)
            ax1.set_title(f"Spatial Trend: {display_region}",
                          fontsize=12, fontweight='bold')
            ax1.set_aspect('equal')

            ax2.plot(ts_clean.time.values, ts_clean.values,
                     color='#95a5a6', linewidth=1.5, alpha=0.6,
                     label='Daily Values')
            ax2.plot(ts_clean.time.values, intercept + slope * x_idx,
                     color='#e74c3c', linewidth=3, label='Trend Line')
            direction = "📈 Increasing" if slope > 0 else "📉 Decreasing"
            ax2.set_title(f"Temporal Trend: {direction}\n{start_date} to {end_date}",
                          fontsize=12, fontweight='bold')
            ax2.set_ylabel("Mean Soil Moisture (m³/m³)")
            ax2.legend()
            ax2.grid(True, linestyle=':', alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.savefig(_out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Map saved → {_out}")
            return True
        except Exception as e:
            print(f"⚠️  Visualization error (slope): {e}")
            traceback.print_exc()
            plt.close('all')
            return False

    # ------------------------------------------------------------------ #
    # SLOPE N-PANEL  (v2.8 — NEW)                                         #
    # ------------------------------------------------------------------ #

    def _visualize_slope_n_panel(self, clipped_list, var_name, labels,
                                  region_name, display_region,
                                  output_path=None):
        """
        Render N slope periods into ONE combined image.

        Layout per period (2 columns):
          Col 0: Spatial slope map
          Col 1: Temporal trend graph

        So for N periods → N rows × 2 columns grid.

        All spatial maps share the same RdBu colorscale (symmetric around 0)
        for easy comparison across periods.

        Parameters
        ----------
        clipped_list   : list of xr.Dataset — one per period
        var_name       : str — name of the soil moisture variable
        labels         : list of str — period label per entry (may contain \\n)
        region_name    : str — raw region name for border drawing
        display_region : str — display-friendly region name for titles
        output_path    : str — optional explicit save path (else OUTPUT_PATH)
        """
        _out = output_path or OUTPUT_PATH
        try:
            n = len(clipped_list)
            if n == 0:
                print("⚠️  No data to visualize.")
                return False

            # ── Figure layout: N rows, 2 cols (map | graph) ───────────
            ncols   = 2
            nrows   = n
            fig_w   = 16          # fixed width: 8 per column
            fig_h   = 5.5 * nrows # 5.5 inches per row

            fig, axes = plt.subplots(nrows, ncols,
                                      figsize=(fig_w, fig_h),
                                      squeeze=False)

            # ── Compute global vmin/vmax for spatial maps ──────────────
            # so all slope maps share a consistent colour scale
            slope_maps = []
            ts_data    = []   # (ts_clean, slope, intercept, x_idx, label) per period

            for clipped in clipped_list:
                sp = _compute_spatial_slope(clipped[var_name])
                slope_maps.append(sp)

                ts       = clipped[var_name].mean(dim=['x', 'y'])
                mask     = ~np.isnan(ts)
                ts_clean = ts.where(mask, drop=True)
                x_idx    = np.arange(len(ts_clean))
                if len(ts_clean) > 2:
                    slope_val, intercept, *_ = stats.linregress(x_idx, ts_clean.values)
                else:
                    slope_val, intercept = 0.0, float(ts_clean.mean())
                ts_data.append((ts_clean, slope_val, intercept, x_idx))

            # Symmetric vmin/vmax across all slope maps
            all_vals = []
            for sp in slope_maps:
                arr = sp.values.ravel()
                arr = arr[~np.isnan(arr)]
                if len(arr):
                    all_vals.extend([float(arr.min()), float(arr.max())])

            if not all_vals:
                print("⚠️  All slope data is NaN — cannot render.")
                plt.close('all')
                return False

            abs_max        = max(abs(min(all_vals)), abs(max(all_vals)))
            vmin, vmax     = -abs_max, abs_max
            if vmin == vmax:
                vmin -= 1e-9
                vmax += 1e-9

            map_kwargs = dict(
                x='x', y='y', cmap='RdBu', center=0,
                vmin=vmin, vmax=vmax,
                cbar_kwargs={'label': 'Slope (m³/m³/day)'},
                add_colorbar=True,
            )

            # ── Draw each row ─────────────────────────────────────────
            for i, (sp, (ts_clean, slope_val, intercept, x_idx), lbl) in enumerate(
                zip(slope_maps, ts_data, labels)
            ):
                ax_map   = axes[i][0]
                ax_graph = axes[i][1]

                flat_lbl = lbl.replace('\n', '  |  ')

                # Left: spatial slope map
                try:
                    sp.plot(ax=ax_map, **map_kwargs)
                except Exception as plot_err:
                    print(f"⚠️  Row {i+1} spatial plot error: {plot_err}")
                    ax_map.text(0.5, 0.5, f"No data\n{flat_lbl}",
                                ha='center', va='center',
                                transform=ax_map.transAxes)

                _draw_borders(ax_map, self.gdf, region_name)
                ax_map.set_title(f"Spatial Trend — {flat_lbl}",
                                  fontsize=10, fontweight='bold', pad=5)
                ax_map.set_aspect('equal')
                ax_map.set_xlabel("Longitude", fontsize=8)
                ax_map.set_ylabel("Latitude",  fontsize=8)
                ax_map.tick_params(labelsize=7)

                # Right: temporal trend graph
                if len(ts_clean) > 0:
                    ax_graph.plot(ts_clean.time.values, ts_clean.values,
                                  color='#95a5a6', linewidth=1.2,
                                  alpha=0.6, label='Daily Values')
                    ax_graph.plot(ts_clean.time.values,
                                  intercept + slope_val * x_idx,
                                  color='#e74c3c', linewidth=2.5,
                                  label='Trend Line')

                direction = "📈 Increasing" if slope_val > 0 else "📉 Decreasing"
                ax_graph.set_title(f"Temporal Trend — {flat_lbl}\n{direction}",
                                    fontsize=10, fontweight='bold', pad=5)
                ax_graph.set_ylabel("Mean Soil Moisture (m³/m³)", fontsize=8)
                ax_graph.tick_params(labelsize=7)
                ax_graph.legend(fontsize=8)
                ax_graph.grid(True, linestyle=':', alpha=0.3)
                fig.autofmt_xdate()

            fig.suptitle(
                f"Soil Moisture Trend Analysis — {display_region}  "
                f"({n} period{'s' if n > 1 else ''})",
                fontsize=13, fontweight='bold', y=1.01
            )
            plt.tight_layout()
            plt.savefig(_out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Combined slope map ({n} period{'s' if n > 1 else ''}) "
                  f"saved → {_out}")
            return True

        except Exception as e:
            print(f"⚠️  Visualization error (_visualize_slope_n_panel): {e}")
            traceback.print_exc()
            plt.close('all')
            return False

    def _analyze_minimum(self, clipped, var_name, display_region, raw_region,
                         start_date, end_date, output_type, output_path=None):
        data  = clipped[var_name]
        total = int(data.size)
        null  = int(data.isnull().sum())

        expected_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
        spatial_size = total // max(1, data.time.size) if hasattr(data, 'time') and data.time.size > 0 else 0
        expected_total = expected_days * spatial_size

        missing_days_count = max(0, expected_days - (data.time.size if hasattr(data, 'time') else 1))
        true_null = null + (missing_days_count * spatial_size)
        true_total = expected_total if expected_total > 0 else total

        stats_dict = {
            'min':         float(data.min()),
            'count':       int(data.count()),
            'missing_pct': (true_null / true_total * 100) if true_total else 0
        }
        output = OutputFormatter.format_minimum(
            stats_dict, output_type, display_region, start_date, end_date)

        viz_created = False
        if output_type in ['map', 'both']:
            viz_created = self._visualize_minimum(
                clipped, var_name, display_region, raw_region,
                start_date, end_date, output_path or OUTPUT_PATH)

        return output, viz_created

    def _visualize_minimum(self, clipped, var_name, display_region, raw_region,
                           start_date, end_date, output_path=None):
        _out = output_path or OUTPUT_PATH
        try:
            settings = FIGURE_SETTINGS['minimum']
            fig, ax  = plt.subplots(1, 1, figsize=settings['figsize'])
            min_data = clipped[var_name].min(dim='time')
            min_data.plot(
                ax=ax, x='x', y='y', cmap='RdYlGn',
                cbar_kwargs={'label': 'Minimum Soil Moisture (m³/m³)'})
            _draw_borders(ax, self.gdf, raw_region)
            ax.set_title(f"Minimum Soil Moisture: {display_region}\n"
                         f"({start_date} to {end_date})",
                         fontsize=14, fontweight='bold')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.savefig(_out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Map saved → {_out}")
            return True
        except Exception as e:
            print(f"⚠️  Visualization error (minimum): {e}")
            traceback.print_exc()
            plt.close('all')
            return False

    def _analyze_maximum(self, clipped, var_name, display_region, raw_region,
                         start_date, end_date, output_type, output_path=None):
        data  = clipped[var_name]
        total = int(data.size)
        null  = int(data.isnull().sum())

        expected_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
        spatial_size = total // max(1, data.time.size) if hasattr(data, 'time') and data.time.size > 0 else 0
        expected_total = expected_days * spatial_size

        missing_days_count = max(0, expected_days - (data.time.size if hasattr(data, 'time') else 1))
        true_null = null + (missing_days_count * spatial_size)
        true_total = expected_total if expected_total > 0 else total

        stats_dict = {
            'max':         float(data.max()),
            'count':       int(data.count()),
            'missing_pct': (true_null / true_total * 100) if true_total else 0
        }
        output = OutputFormatter.format_maximum(
            stats_dict, output_type, display_region, start_date, end_date)

        viz_created = False
        if output_type in ['map', 'both']:
            viz_created = self._visualize_maximum(
                clipped, var_name, display_region, raw_region,
                start_date, end_date, output_path or OUTPUT_PATH)

        return output, viz_created

    def _visualize_maximum(self, clipped, var_name, display_region, raw_region,
                           start_date, end_date, output_path=None):
        _out = output_path or OUTPUT_PATH
        try:
            settings = FIGURE_SETTINGS['maximum']
            fig, ax  = plt.subplots(1, 1, figsize=settings['figsize'])
            max_data = clipped[var_name].max(dim='time')
            max_data.plot(
                ax=ax, x='x', y='y', cmap='Blues',
                cbar_kwargs={'label': 'Maximum Soil Moisture (m³/m³)'})
            _draw_borders(ax, self.gdf, raw_region)
            ax.set_title(f"Maximum Soil Moisture: {display_region}\n"
                         f"({start_date} to {end_date})",
                         fontsize=14, fontweight='bold')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.savefig(_out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Map saved → {_out}")
            return True
        except Exception as e:
            print(f"⚠️  Visualization error (maximum): {e}")
            traceback.print_exc()
            plt.close('all')
            return False

    # ================================================================== #
    # COMPARISON HANDLER  (routes to N-way time or region comparison)
    # ================================================================== #

    def _handle_comparison(self, region, start_date, end_date,
                            output_type, comparison_info, output_path=None):
        if not comparison_info:
            return "❌ Comparison info not provided.", False

        comp_type   = comparison_info.get('comparison_type', 'time')
        comp_metric = comparison_info.get('comparison_metric', 'mean')

        if comp_type == 'region':
            return self._analyze_comparison_region(
                region,
                comparison_info.get('comparison_region2', ''),
                start_date, end_date,
                output_type, comp_metric,
                output_path=output_path
            )
        else:
            # N-way time comparison
            periods = comparison_info.get('comparison_periods', [])

            # Back-compat: if old keys used, reconstruct periods list
            if not periods:
                p1 = comparison_info.get('comparison_period1')
                p2 = comparison_info.get('comparison_period2')
                if p1 and p2:
                    periods = [p1, p2]

            if len(periods) < 2:
                return (
                    "❌ Two or more time periods are required for time comparison. "
                    "Please specify all periods clearly."
                ), False

            return self._analyze_comparison_n_periods(
                region, periods, output_type, comp_metric,
                output_path=output_path
            )

    # ── METRIC HELPERS ───────────────────────────────────────────────────

    @staticmethod
    def _compute_metric(data_array, metric: str) -> float:
        if metric == 'min':
            return float(data_array.min())
        elif metric == 'max':
            return float(data_array.max())
        elif metric == 'slope':
            ts    = data_array.mean(dim=['x', 'y'])
            mask  = ~np.isnan(ts)
            clean = ts.where(mask, drop=True)
            if len(clean) < 3:
                return float('nan')
            slope, *_ = stats.linregress(np.arange(len(clean)), clean.values)
            return float(slope)
        else:
            return float(data_array.mean())

    @staticmethod
    def _metric_map(data_array, metric: str):
        if metric == 'min':
            return data_array.min(dim='time')
        elif metric == 'max':
            return data_array.max(dim='time')
        elif metric == 'slope':
            return _compute_spatial_slope(data_array)
        else:
            return data_array.mean(dim='time')

    # ── N-WAY TIME COMPARISON ────────────────────────────────────────────

    def _analyze_comparison_n_periods(self, region, periods, output_type, comp_metric,
                                      output_path=None):
        """
        Compare the same region across N time periods.
        For 'slope' metric: ONE combined image with spatial map + trend graph per period.
        For other metrics: ONE combined image via _visualize_n_panel.
        """
        v = list(self.ds.data_vars)[0]

        period_data    = []   # list of clipped DataArrays (one per period)
        clipped_ds_list = []  # list of clipped Datasets — kept for slope visualizer
        period_labels  = []
        display_region = region.title()

        for idx, (s, e) in enumerate(periods):
            if len(s) == 7: s += "-01"
            if len(e) == 7: e += "-28"

            subset = self.ds.sel(time=slice(s, e)).compute()
            clipped, disp, ok = self._clip_region(subset, region)
            if not ok:
                return clipped, False

            if clipped[v].size == 0 or clipped[v].isnull().all():
                return (
                    f"❌ No valid data for {disp} in period {s} to {e}."
                ), False

            display_region = disp
            period_data.append(clipped[v])      # DataArray
            clipped_ds_list.append(clipped)     # Dataset (for _visualize_slope_n_panel)
            label = f"Period {idx+1}\n{s} to {e}"
            period_labels.append(label)

        values = [self._compute_metric(da, comp_metric) for da in period_data]

        output = OutputFormatter.format_comparison_n_periods(
            values        = values,
            period_labels = period_labels,
            region        = display_region,
            metric        = comp_metric,
            output_type   = output_type,
        )

        viz_created = False
        if output_type in ['map', 'both']:
            if comp_metric == 'slope':
                # _visualize_slope_n_panel renders both spatial slope map AND
                # temporal trend graph for every period in one combined image.
                viz_created = self._visualize_slope_n_panel(
                    clipped_list   = clipped_ds_list,
                    var_name       = v,
                    labels         = period_labels,
                    region_name    = region,
                    display_region = display_region,
                    output_path    = output_path,
                )
            else:
                maps = [self._metric_map(da, comp_metric) for da in period_data]
                viz_created = self._visualize_n_panel(
                    maps         = maps,
                    labels       = period_labels,
                    region_name  = region,
                    region_names = None,
                    suptitle     = (
                        f"{len(periods)}-Period Comparison: {display_region}  |  "
                        f"Metric: {comp_metric.upper()}"
                    ),
                    metric       = comp_metric,
                    output_path  = output_path,
                )

        return output, viz_created

    # ── REGION COMPARISON ────────────────────────────────────────────────

    def _analyze_comparison_region(self, region1, region2,
                                    start_date, end_date,
                                    output_type, comp_metric,
                                    output_path=None):
        """
        Compare two regions for the same time period.
        Always produces ONE combined image via _visualize_n_panel.
        """
        if not region2:
            return "❌ Second region not found. Please name two regions.", False

        subset = self.ds.sel(time=slice(start_date, end_date)).compute()

        clipped1, display1, ok = self._clip_region(subset, region1)
        if not ok:
            return clipped1, False

        clipped2, display2, ok = self._clip_region(subset, region2)
        if not ok:
            return clipped2, False

        v = list(self.ds.data_vars)[0]

        if clipped1[v].size == 0:
            return f"❌ No spatial data found for {display1} in this period.", False
        if clipped2[v].size == 0:
            return f"❌ No spatial data found for {display2} in this period.", False
        if clipped1[v].isnull().all():
            return f"❌ All values missing for {display1}.", False
        if clipped2[v].isnull().all():
            return f"❌ All values missing for {display2}.", False

        value1 = self._compute_metric(clipped1[v], comp_metric)
        value2 = self._compute_metric(clipped2[v], comp_metric)

        stats_dict = {'value1': value1, 'value2': value2}

        output = OutputFormatter.format_comparison_region(
            stats_dict, output_type, display1, display2,
            start_date, end_date, comp_metric
        )

        viz_created = False
        if output_type in ['map', 'both']:
            map1 = self._metric_map(clipped1[v], comp_metric)
            map2 = self._metric_map(clipped2[v], comp_metric)
            viz_created = self._visualize_n_panel(
                maps         = [map1, map2],
                labels       = [display1, display2],
                region_name  = None,
                region_names = [region1, region2],
                suptitle     = (
                    f"Regional Comparison: {display1} vs {display2}  |  "
                    f"{start_date} to {end_date}  |  "
                    f"Metric: {comp_metric.upper()}"
                ),
                metric       = comp_metric,
                output_path  = output_path,
            )

        return output, viz_created

    # ── UNIFIED N-PANEL VISUALIZATION ────────────────────────────────────

    def _visualize_n_panel(
        self,
        maps,
        labels,
        region_name,
        suptitle,
        metric,
        region_names=None,
        output_path=None,
    ):
        """
        Render a clean N-panel comparison map in a dynamic grid layout.
        ALWAYS saves to OUTPUT_PATH (latest_analysis.png) as ONE image.

        Layout rules:
          N=1  → 1×1
          N=2  → 1×2
          N=3  → 1×3
          N=4  → 2×2
          N=5  → 2×3  (one empty cell)
          N=6  → 2×3
          N=7+ → rows × 3  grid
        """
        try:
            n = len(maps)
            if n == 0:
                print("⚠️  No maps to visualize.")
                return False

            ncols = min(n, 3)
            nrows = math.ceil(n / ncols)

            fig_w = ncols * 7.0
            fig_h = nrows * 6.5

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(fig_w, fig_h), squeeze=False
            )

            # Shared colorscale
            valid_vals = []
            for m in maps:
                arr = m.values.ravel()
                arr = arr[~np.isnan(arr)]
                if len(arr):
                    valid_vals.extend([float(arr.min()), float(arr.max())])

            if not valid_vals:
                print("⚠️  All map data is NaN — cannot render.")
                plt.close('all')
                return False

            g_min = min(valid_vals)
            g_max = max(valid_vals)

            if metric == 'slope':
                cmap     = 'RdBu'
                cb_label = 'Slope (m³/m³/day)'
                center   = 0.0
                abs_max  = max(abs(g_min), abs(g_max))
                vmin, vmax = -abs_max, abs_max
            elif metric == 'min':
                cmap     = 'RdYlGn'
                cb_label = 'Minimum Soil Moisture (m³/m³)'
                center   = None
                vmin, vmax = g_min, g_max
            elif metric == 'max':
                cmap     = 'Blues'
                cb_label = 'Maximum Soil Moisture (m³/m³)'
                center   = None
                vmin, vmax = g_min, g_max
            else:
                cmap     = 'YlGnBu'
                cb_label = 'Mean Soil Moisture (m³/m³)'
                center   = None
                vmin, vmax = g_min, g_max

            if vmin == vmax:
                vmin -= 1e-6
                vmax += 1e-6

            plot_kwargs = dict(
                x='x', y='y', cmap=cmap,
                vmin=vmin, vmax=vmax,
                cbar_kwargs={'label': cb_label},
                add_colorbar=True,
            )
            if center is not None:
                plot_kwargs['center'] = center

            for i, (m, lbl) in enumerate(zip(maps, labels)):
                row, col = divmod(i, ncols)
                ax = axes[row][col]

                try:
                    m.plot(ax=ax, **plot_kwargs)
                except Exception as plot_err:
                    print(f"⚠️  Panel {i+1} plot error: {plot_err}")
                    ax.text(0.5, 0.5, f"No data\n{lbl}",
                            ha='center', va='center', transform=ax.transAxes)

                rname = (region_names[i] if region_names else region_name)
                _draw_borders(ax, self.gdf, rname)

                clean_lbl = lbl.replace('\n', '  |  ')
                ax.set_title(clean_lbl, fontsize=10, fontweight='bold', pad=6)
                ax.set_aspect('equal')
                ax.set_xlabel("Longitude", fontsize=8)
                ax.set_ylabel("Latitude",  fontsize=8)
                ax.tick_params(labelsize=7)

            total_cells = nrows * ncols
            for j in range(n, total_cells):
                row, col = divmod(j, ncols)
                axes[row][col].set_visible(False)

            _out = output_path or OUTPUT_PATH
            fig.suptitle(suptitle, fontsize=12, fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.savefig(_out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Combined comparison map ({n} panels) saved → {_out}")
            return True

        except Exception as e:
            print(f"⚠️  Visualization error (_visualize_n_panel): {e}")
            traceback.print_exc()
            plt.close('all')
            return False