import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from scipy.ndimage import convolve
from skimage.morphology import reconstruction
from pysheds.grid import Grid

# Set paths
DATA_DIR = "/home/edier/Documents/INVESTIGACION/PAPERS/ELABORACION/Espacial/DATA"
DEM_PATH = os.path.join(DATA_DIR, "dem_iguana_1m.tif")
POINTS_PATH = os.path.join(DATA_DIR, "Deslizamientos_Iguana.gpkg")
OUTPUT_CSV = os.path.join(DATA_DIR, "landslide_morphometrics.csv")
TMP_FILLED = os.path.join(DATA_DIR, "tmp_filled.tif")

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

def main():
    print("Reading DEM...")
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        dx = transform[0]
        dy = abs(transform[4])
        meta = src.meta.copy()

    mask = (dem == nodata) | np.isnan(dem)
    dem[mask] = np.nan

    print("Calculating derivatives (Slope, Aspect, Curvatures)...")
    D, E, H, G, F = calculate_derivatives(dem, dx, dy)
    
    results = {}
    results['elevation'] = dem
    
    slope_rad = np.arctan(np.sqrt(D**2 + E**2))
    results['slope'] = np.degrees(slope_rad)
    
    aspect = 180.0 - np.degrees(np.arctan2(E, D))
    aspect = np.mod(aspect, 360)
    results['aspect'] = aspect
    
    denom = D**2 + E**2
    results['curv_vert'] = -2 * (G * D**2 + H * E**2 + F * D * E) / (denom + 1e-10)
    results['curv_horiz'] = -2 * (G * E**2 + H * D**2 - F * D * E) / (denom + 1e-10)
    
    # TPI and TRI
    k_tpi = np.ones((5, 5))
    k_tpi[2, 2] = 0
    k_tpi /= 24.0
    results['tpi'] = dem - convolve(dem, k_tpi)
    results['tri'] = np.sqrt(np.abs(convolve(dem**2, np.ones((3,3))) - 9*(convolve(dem, np.ones((3,3)))/9.0)**2))

    print("Filling sinks with skimage...")
    dem_filled = fill_sinks_skimage(dem)
    
    # Save filled DEM for pysheds compatibility
    meta.update(dtype='float32', nodata=-9999)
    with rasterio.open(TMP_FILLED, 'w', **meta) as dst:
        to_save = dem_filled.copy()
        to_save[np.isnan(to_save)] = -9999
        dst.write(to_save, 1)
    
    print("Loading Grid in pysheds...")
    grid = Grid.from_raster(TMP_FILLED)
    dem_raster = grid.read_raster(TMP_FILLED)
    
    print("Calculating Flow Direction...")
    # Pass the actual raster object
    d8_fdir = grid.flowdir(dem_raster, out_name='dir')
    
    print("Calculating Flow Accumulation...")
    acc = grid.accumulation(d8_fdir, out_name='acc')
    results['flow_acc'] = acc
    
    print("Calculating SPI, TWI, and LS-Factor...")
    as_val = (acc + 1) * dx * dy
    tan_slope = np.tan(slope_rad)
    tan_slope = np.where(tan_slope <= 0, 0.001, tan_slope)
    
    results['spi'] = np.log(as_val * tan_slope + 1e-10)
    results['twi'] = np.log(as_val / tan_slope)
    results['ls_factor'] = (as_val / 22.13)**0.4 * (np.where(np.sin(slope_rad) <= 0, 0.0001, np.sin(slope_rad)) / 0.0896)**1.3

    print("Extracting values at landslide points...")
    gdf = gpd.read_file(POINTS_PATH)
    rows, cols = rasterio.transform.rowcol(transform, gdf.geometry.x, gdf.geometry.y)
    rows = np.clip(rows, 0, dem.shape[0] - 1)
    cols = np.clip(cols, 0, dem.shape[1] - 1)
    
    data_points = gdf[['id', 'Name']].copy()
    for name, array in results.items():
        print(f"Sampling {name}...")
        data_points[name] = array[rows, cols]

    print(f"Saving results to {OUTPUT_CSV}...")
    data_points.to_csv(OUTPUT_CSV, index=False)
    
    # Cleanup
    if os.path.exists(TMP_FILLED):
        os.remove(TMP_FILLED)
    
    print("Done!")

if __name__ == "__main__":
    main()
