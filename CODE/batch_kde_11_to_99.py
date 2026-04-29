"""
Batch KDE calculation and figure generation for odd kernel sizes from 11 to 99.
Each kernel NxN uses N² cells and normalizes by N².
"""
import os
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.enums import Resampling
from scipy.ndimage import convolve
import geopandas as gpd
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = "/home/edier/Documents/INVESTIGACION/PAPERS/ELABORACION/Espacial"
DATA_DIR   = os.path.join(ROOT_DIR, "DATA")
FIG_DIR    = os.path.join(ROOT_DIR, "FIG")
DEM_PATH   = os.path.join(DATA_DIR, "dem_iguana_1m.tif")
POINTS_PATH = os.path.join(DATA_DIR, "Deslizamientos_Iguana.gpkg")
BASIN_PATH = os.path.join(DATA_DIR, "Cuenca_Iguana.shp")

os.makedirs(FIG_DIR, exist_ok=True)

# ── Kernel sizes: odd numbers from 11 to 99 ────────────────────────────────────
SIZES = list(range(11, 100, 2))   # [11, 13, 15, ..., 99]

# ── 1. Load common data once ───────────────────────────────────────────────────
print("Reading DEM metadata...")
with rasterio.open(DEM_PATH) as src:
    transform = src.transform
    meta      = src.meta.copy()
    width     = src.width
    height    = src.height

print("Loading landslide points...")
gdf = gpd.read_file(POINTS_PATH)

print("Rasterizing points...")
point_grid = np.zeros((height, width), dtype=np.float32)
rows, cols = rasterio.transform.rowcol(transform, gdf.geometry.x, gdf.geometry.y)
valid = (np.array(rows) >= 0) & (np.array(rows) < height) & \
        (np.array(cols) >= 0) & (np.array(cols) < width)
rows = np.array(rows)[valid]
cols = np.array(cols)[valid]
np.add.at(point_grid, (rows, cols), 1)

print("Loading basin boundary...")
basin = gpd.read_file(BASIN_PATH)

meta.update(dtype='float32', nodata=-9999)

# ── 2. Loop over kernel sizes ──────────────────────────────────────────────────
for n in SIZES:
    tag      = f"{n}x{n}"
    tif_path = os.path.join(DATA_DIR, f"kde_{tag}_iguana_1m.tif")
    fig_path = os.path.join(FIG_DIR,  f"kde_{tag}_density_map.png")
    n_cells  = n * n

    # ── 2a. Calculate KDE ──────────────────────────────────────────────────────
    print(f"[{tag}] Convolving with {tag} kernel ({n_cells} cells)...")
    kernel      = np.ones((n, n), dtype=np.float32)
    kde_grid    = convolve(point_grid, kernel, mode='constant', cval=0.0)
    kde_density = kde_grid / float(n_cells)

    print(f"[{tag}] Saving TIF -> {tif_path}")
    with rasterio.open(tif_path, 'w', **meta) as dst:
        dst.write(kde_density, 1)

    # ── 2b. Generate figure ────────────────────────────────────────────────────
    print(f"[{tag}] Generating figure...")
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    basin.plot(ax=ax, facecolor='lightgray', edgecolor='black', linewidth=1, alpha=0.5)

    with rasterio.open(tif_path) as src:
        data = src.read(
            1,
            out_shape=(
                int(src.height * 0.05),
                int(src.width  * 0.05)
            ),
            resampling=Resampling.average
        )
        t = src.transform * src.transform.scale(
            (src.width  / data.shape[1]),
            (src.height / data.shape[0])
        )
        data[data < 1e-5] = np.nan
        show(data, transform=t, ax=ax, cmap='Reds',
             vmin=0, vmax=np.nanpercentile(data, 99))

    ax.set_title(f"Movimientos en Masa: KDE {tag} Neighborhood Density")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[{tag}] Done -> {fig_path}")

print("\n✓ All KDE kernels processed successfully.")
