"""
Master script: For each odd window size N (3 to 99):
  1. Smooth each morphometric variable with NxN moving window → PNG map
  2. Compute NxN KDE density raster of landslides → PNG map
  3. Scatter: KDE landslide density vs each morphometric variable at slide points
  4. Save master CSV

Morphometric variables:
  elevation, slope, aspect, curv_vert, curv_horiz,
  tpi, tri, flow_acc, spi, twi, ls_factor

Output directories:
  FIG/maps_morpho_kde/{var}_{N}x{N}.png        — smoothed morpho maps
  FIG/maps_kde_density/kde_density_{N}x{N}.png — landslide KDE density maps
  FIG/scatter_kde_vs_morpho/kde_{N}x{N}_{var}.png — scatter plots
  DATA/morpho_kde_scatter_data.csv
"""
import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, uniform_filter
from skimage.morphology import reconstruction
from pysheds.grid import Grid

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR         = "/home/edier/Documents/INVESTIGACION/PAPERS/ELABORACION/Espacial"
DATA_DIR         = os.path.join(ROOT_DIR, "DATA")
FIG_DIR_SCATTER  = os.path.join(ROOT_DIR, "FIG", "scatter_kde_vs_morpho")
FIG_DIR_MORPHO   = os.path.join(ROOT_DIR, "FIG", "maps_morpho_kde")
FIG_DIR_KDE      = os.path.join(ROOT_DIR, "FIG", "maps_kde_density")
DEM_PATH         = os.path.join(DATA_DIR, "dem_iguana_1m.tif")
POINTS_PATH      = os.path.join(DATA_DIR, "Deslizamientos_Iguana.gpkg")
TMP_FILLED       = os.path.join(DATA_DIR, "tmp_filled_master.tif")
OUT_CSV          = os.path.join(DATA_DIR, "morpho_kde_scatter_data.csv")

for d in [FIG_DIR_SCATTER, FIG_DIR_MORPHO, FIG_DIR_KDE]:
    os.makedirs(d, exist_ok=True)

# Window sizes: odd numbers 3 to 99
SIZES = list(range(3, 100, 2))

MORPH_VARS = [
    'elevation', 'slope', 'aspect',
    'curv_vert', 'curv_horiz',
    'tpi', 'tri',
    'flow_acc', 'spi', 'twi', 'ls_factor',
]

LABELS = {
    'elevation':  'Elevación (m)',
    'slope':      'Pendiente (°)',
    'aspect':     'Aspecto (°)',
    'curv_vert':  'Curvatura Vertical',
    'curv_horiz': 'Curvatura Horizontal',
    'tpi':        'TPI',
    'tri':        'TRI',
    'flow_acc':   'Acumulación de Flujo',
    'spi':        'SPI',
    'twi':        'TWI',
    'ls_factor':  'Factor LS',
}

CMAPS = {
    'elevation':  'terrain',
    'slope':      'YlOrRd',
    'aspect':     'twilight',
    'curv_vert':  'RdBu_r',
    'curv_horiz': 'RdBu_r',
    'tpi':        'RdBu_r',
    'tri':        'YlOrRd',
    'flow_acc':   'Blues',
    'spi':        'YlOrRd',
    'twi':        'Blues',
    'ls_factor':  'YlOrRd',
}

# ── Helper functions ───────────────────────────────────────────────────────────
def calc_derivatives(dem, dx, dy):
    kd = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]]) / (2 * dx)
    ke = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]]) / (2 * dy)
    kh = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]) / (2 * dx ** 2)
    kg = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]) / (2 * dy ** 2)
    kf = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / (4 * dx * dy)
    return (convolve(dem, kd), convolve(dem, ke),
            convolve(dem, kh), convolve(dem, kg), convolve(dem, kf))


def fill_sinks(dem):
    seed = np.copy(dem)
    seed[1:-1, 1:-1] = np.nanmax(dem)
    return reconstruction(seed, dem, method='erosion')


def neighborhood_mean(array, n):
    """NxN moving-window mean (full raster)."""
    return uniform_filter(array.astype(np.float64), size=n, mode='nearest')


def kde_density_raster(point_grid, n):
    """NxN moving-window landslide density (full raster)."""
    kernel = np.ones((n, n), dtype=np.float64) / float(n * n)
    return convolve(point_grid.astype(np.float64), kernel,
                    mode='constant', cval=0.0)


def extract_at_points(raster, rows, cols):
    r = np.clip(rows, 0, raster.shape[0] - 1)
    c = np.clip(cols, 0, raster.shape[1] - 1)
    return raster[r, c]


def pct_limits(arr, plo=2, phi=98):
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return None, None
    return float(np.percentile(valid, plo)), float(np.percentile(valid, phi))


def save_map(arr, title, cmap, out_path, extent,
             pt_x=None, pt_y=None, max_dim=2000):
    """Save a 2-D raster as a PNG map, downsampling display if needed."""
    display = arr.copy().astype(np.float64)
    h, w = display.shape
    step = max(1, max(h, w) // max_dim)
    display_ds = display[::step, ::step]

    vmin, vmax = pct_limits(display_ds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(display_ds, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin='upper', interpolation='bilinear',
                   aspect='equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if pt_x is not None:
        ax.scatter(pt_x, pt_y, c='blue', s=2, alpha=0.4,
                   linewidths=0, label='Deslizamientos')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


# ── 1. Load DEM and compute morphometric arrays ────────────────────────────────
print("Reading DEM...")
with rasterio.open(DEM_PATH) as src:
    dem    = src.read(1).astype(np.float64)
    trans  = src.transform
    nd     = src.nodata
    dx     = trans[0]
    dy     = abs(trans[4])
    meta   = src.meta.copy()
    height, width = dem.shape
    left   = trans.c
    right  = trans.c + width  * trans.a
    top    = trans.f
    bottom = trans.f + height * trans.e
    extent = [left, right, bottom, top]

dem[(dem == nd) | np.isnan(dem)] = np.nan
nan_mask = np.isnan(dem)
dem_fill0 = np.where(nan_mask, 0.0, dem)   # NaN → 0 for convolutions

print("Computing morphometric derivatives...")
D, E, H, G, F = calc_derivatives(dem_fill0, dx, dy)

slope_rad = np.arctan(np.sqrt(D ** 2 + E ** 2))
denom     = D ** 2 + E ** 2 + 1e-10

morph = {}
morph['elevation']  = dem.copy()
morph['slope']      = np.degrees(slope_rad)
morph['aspect']     = np.mod(180.0 - np.degrees(np.arctan2(E, D)), 360.0)
morph['curv_vert']  = -2 * (G * D ** 2 + H * E ** 2 + F * D * E) / denom
morph['curv_horiz'] = -2 * (G * E ** 2 + H * D ** 2 - F * D * E) / denom

k_tpi = np.ones((5, 5)); k_tpi[2, 2] = 0; k_tpi /= 24.0
morph['tpi'] = dem_fill0 - convolve(dem_fill0, k_tpi)
morph['tri'] = np.sqrt(np.abs(
    convolve(dem_fill0 ** 2, np.ones((3, 3))) -
    9 * (convolve(dem_fill0, np.ones((3, 3))) / 9.0) ** 2
))

print("Computing hydrological indices (fill sinks)...")
dem_for_fill = np.where(nan_mask, np.nanmin(dem) - 1.0, dem)
dem_filled   = fill_sinks(dem_for_fill)
meta2 = meta.copy(); meta2.update(dtype='float32', nodata=-9999)
dem_save = dem_filled.copy(); dem_save[nan_mask] = -9999
with rasterio.open(TMP_FILLED, 'w', **meta2) as dst:
    dst.write(dem_save.astype(np.float32), 1)

grid    = Grid.from_raster(TMP_FILLED)
dem_r   = grid.read_raster(TMP_FILLED)
d8_fdir = grid.flowdir(dem_r)
acc     = grid.accumulation(d8_fdir)
morph['flow_acc'] = np.array(acc, dtype=np.float64)

as_val    = (morph['flow_acc'] + 1) * dx * dy
tan_slope = np.where(np.tan(slope_rad) <= 0, 0.001,  np.tan(slope_rad))
sin_slope = np.where(np.sin(slope_rad) <= 0, 0.0001, np.sin(slope_rad))
morph['spi']       = np.log(as_val * tan_slope + 1e-10)
morph['twi']       = np.log(as_val / tan_slope)
morph['ls_factor'] = (as_val / 22.13) ** 0.4 * (sin_slope / 0.0896) ** 1.3

# Apply NaN mask to all morphometric arrays
for v in morph:
    morph[v][nan_mask] = np.nan

if os.path.exists(TMP_FILLED):
    os.remove(TMP_FILLED)

# ── 2. Load landslide points ───────────────────────────────────────────────────
print("Loading landslide points...")
gdf = gpd.read_file(POINTS_PATH)
pt_rows, pt_cols = rasterio.transform.rowcol(trans,
                                              gdf.geometry.x, gdf.geometry.y)
valid   = (
    (np.array(pt_rows) >= 0) & (np.array(pt_rows) < height) &
    (np.array(pt_cols) >= 0) & (np.array(pt_cols) < width)
)
pt_rows = np.array(pt_rows)[valid]
pt_cols = np.array(pt_cols)[valid]
n_pts   = len(pt_rows)
print(f"  {n_pts} valid landslide points")

# Point coordinates for map overlay
pt_x = trans.c + pt_cols * trans.a + trans.a / 2.0
pt_y = trans.f + pt_rows * trans.e + trans.e / 2.0

point_grid = np.zeros((height, width), dtype=np.float32)
np.add.at(point_grid, (pt_rows, pt_cols), 1)

# ── 3. Main loop over window sizes ─────────────────────────────────────────────
records = []

for n in SIZES:
    tag = f"{n}x{n}"
    print(f"\n[{tag}] Processing...", flush=True)

    # ── 3a. KDE landslide density raster & map ─────────────────────────────────
    kde_raster = kde_density_raster(point_grid, n)
    kde_raster[nan_mask] = np.nan

    save_map(
        kde_raster,
        f'Densidad KDE Deslizamientos ({tag})',
        'hot_r',
        os.path.join(FIG_DIR_KDE, f"kde_density_{tag}.png"),
        extent,
        pt_x=pt_x, pt_y=pt_y,
    )

    kde_vals = extract_at_points(kde_raster, pt_rows, pt_cols)

    # ── 3b. Smoothed morphometric maps + extract at points ─────────────────────
    morpho_vals = {}
    for var in MORPH_VARS:
        src_arr   = np.where(nan_mask, 0.0, np.nan_to_num(morph[var], nan=0.0))
        smoothed  = neighborhood_mean(src_arr, n)
        smoothed[nan_mask] = np.nan

        save_map(
            smoothed,
            f'{LABELS[var]} — Ventana {tag}',
            CMAPS[var],
            os.path.join(FIG_DIR_MORPHO, f"{var}_{tag}.png"),
            extent,
        )

        morpho_vals[var] = extract_at_points(smoothed, pt_rows, pt_cols)

    # ── 3c. Scatter plots ──────────────────────────────────────────────────────
    print(f"[{tag}] Generating {len(MORPH_VARS)} scatter plots...", flush=True)
    for var in MORPH_VARS:
        x = morpho_vals[var]
        y = kde_vals

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, alpha=0.5, s=20, color='#e74c3c', edgecolors='none')

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() > 2:
            z = np.polyfit(x[finite], y[finite], 1)
            xline = np.linspace(x[finite].min(), x[finite].max(), 200)
            ax.plot(xline, np.polyval(z, xline), 'k--', lw=1, alpha=0.7)

        ax.set_xlabel(LABELS[var], fontsize=10)
        ax.set_ylabel(f'KDE Densidad ({tag})', fontsize=10)
        ax.set_title(f'KDE {tag}  vs  {LABELS[var]}', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIG_DIR_SCATTER, f"kde_{tag}_{var}.png"),
            dpi=150, bbox_inches='tight',
        )
        plt.close()

    # ── 3d. Accumulate CSV records ─────────────────────────────────────────────
    for i in range(n_pts):
        row = {'window': n, 'point_id': i, 'kde_density': kde_vals[i]}
        for var in MORPH_VARS:
            row[var] = morpho_vals[var][i]
        records.append(row)

    print(f"[{tag}] Done.", flush=True)

# ── 4. Save master CSV ─────────────────────────────────────────────────────────
print("\nSaving master CSV...")
df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}  ({len(df)} rows)")

print("\n✓ All done.")
print(f"  Maps morpho : {FIG_DIR_MORPHO}")
print(f"  Maps KDE    : {FIG_DIR_KDE}")
print(f"  Scatters    : {FIG_DIR_SCATTER}")
