import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import convolve
from skimage.morphology import reconstruction
from pysheds.grid import Grid

# Set paths
DATA_DIR = "/home/edier/Documents/INVESTIGACION/PAPERS/ELABORACION/Espacial/DATA"
DEM_PATH = os.path.join(DATA_DIR, "dem_iguana_1m.tif")
POINTS_PATH = os.path.join(DATA_DIR, "Deslizamientos_Iguana.gpkg")
ORIG_CSV = os.path.join(DATA_DIR, "landslide_morphometrics.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "landslide_neighborhood_analysis.csv")
PLOT_PATH = os.path.join(DATA_DIR, "neighbor_comparison_plots.png")
TMP_FILLED = os.path.join(DATA_DIR, "tmp_filled_neighbor.tif")

def calculate_derivatives(dem, dx, dy):
    kd = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]]) / (2 * dx)
    ke = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]]) / (2 * dy)
    kh = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]) / (2 * dx**2)
    kg = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]) / (2 * dy**2)
    kf = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / (4 * dx * dy)
    D = convolve(dem, kd)
    E = convolve(dem, ke)
    H = convolve(dem, kh)
    G = convolve(dem, kg)
    F = convolve(dem, kf)
    return D, E, H, G, F

def fill_sinks_skimage(dem):
    mask = dem
    seed = np.copy(dem)
    seed[1:-1, 1:-1] = np.nanmax(dem)
    filled = reconstruction(seed, mask, method='erosion')
    return filled

def get_neighbor_mean(array, rows, cols):
    """
    Extracts the mean of the 8 neighbors for given row, col coordinates.
    """
    neighbor_means = []
    for r, c in zip(rows, cols):
        try:
            # 3x3 window around (r, c)
            window = array[r-1:r+2, c-1:c+2].flatten()
            if len(window) == 9:
                # Exclude center (index 4)
                neighbors = np.delete(window, 4)
                neighbor_means.append(np.nanmean(neighbors))
            else:
                neighbor_means.append(np.nan)
        except IndexError:
            neighbor_means.append(np.nan)
    return np.array(neighbor_means)

def main():
    print("Reading DEM and calculating indices...")
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        dx = transform[0]
        dy = abs(transform[4])
        meta = src.meta.copy()

    mask = (dem == nodata) | np.isnan(dem)
    dem[mask] = np.nan

    D, E, H, G, F = calculate_derivatives(dem, dx, dy)
    
    results = {}
    results['elevation'] = dem
    slope_rad = np.arctan(np.sqrt(D**2 + E**2))
    results['slope'] = np.degrees(slope_rad)
    results['aspect'] = np.mod(180.0 - np.degrees(np.arctan2(E, D)), 360)
    denom = D**2 + E**2
    results['curv_vert'] = -2 * (G * D**2 + H * E**2 + F * D * E) / (denom + 1e-10)
    results['curv_horiz'] = -2 * (G * E**2 + H * D**2 - F * D * E) / (denom + 1e-10)
    k_tpi = np.ones((5, 5)); k_tpi[2, 2] = 0; k_tpi /= 24.0
    results['tpi'] = dem - convolve(dem, k_tpi)
    results['tri'] = np.sqrt(np.abs(convolve(dem**2, np.ones((3,3))) - 9*(convolve(dem, np.ones((3,3)))/9.0)**2))

    print("Hydrological indices...")
    dem_filled = fill_sinks_skimage(dem)
    meta.update(dtype='float32', nodata=-9999)
    with rasterio.open(TMP_FILLED, 'w', **meta) as dst:
        to_save = dem_filled.copy(); to_save[np.isnan(to_save)] = -9999; dst.write(to_save, 1)
    
    grid = Grid.from_raster(TMP_FILLED)
    dem_raster = grid.read_raster(TMP_FILLED)
    d8_fdir = grid.flowdir(dem_raster)
    acc = grid.accumulation(d8_fdir)
    results['flow_acc'] = acc
    as_val = (acc + 1) * dx * dy
    tan_slope = np.where(np.tan(slope_rad) <= 0, 0.001, np.tan(slope_rad))
    results['spi'] = np.log(as_val * tan_slope + 1e-10)
    results['twi'] = np.log(as_val / tan_slope)
    results['ls_factor'] = (as_val / 22.13)**0.4 * (np.where(np.sin(slope_rad) <= 0, 0.0001, np.sin(slope_rad)) / 0.0896)**1.3

    print("Extracting points and neighbor averages...")
    gdf = gpd.read_file(POINTS_PATH)
    rows, cols = rasterio.transform.rowcol(transform, gdf.geometry.x, gdf.geometry.y)
    rows = np.clip(rows, 1, dem.shape[0] - 2) # Ensure space for 3x3
    cols = np.clip(cols, 1, dem.shape[1] - 2)
    
    data = gdf[['id', 'Name']].copy()
    for name, array in results.items():
        print(f"Processing {name}...")
        data[name] = array[rows, cols]
        data[f"{name}_neighbor_avg"] = get_neighbor_mean(array, rows, cols)

    print("Cleaning up...")
    if os.path.exists(TMP_FILLED): os.remove(TMP_FILLED)
    data.to_csv(OUTPUT_CSV, index=False)

    print("Generating scatter plots...")
    vars_to_plot = list(results.keys())
    n_vars = len(vars_to_plot)
    cols_plot = 3
    rows_plot = (n_vars + cols_plot - 1) // cols_plot
    
    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(15, 5 * rows_plot))
    axes = axes.flatten()
    
    sns.set_style("whitegrid")
    for i, var in enumerate(vars_to_plot):
        ax = axes[i]
        sns.scatterplot(data=data, x=var, y=f"{var}_neighbor_avg", ax=ax, alpha=0.6, color='#2ecc71')
        # Add 1:1 line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        ax.set_title(f"{var.capitalize()}: Original vs Neighbor Avg", fontweight='bold')
        ax.set_xlabel("Original Value")
        ax.set_ylabel("Mean of 8 Neighbors")

    # Hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"Plots saved to {PLOT_PATH}")
    print("Done!")

if __name__ == "__main__":
    main()
