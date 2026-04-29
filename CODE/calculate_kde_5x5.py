import os
import numpy as np
import rasterio
import geopandas as gpd
from scipy.ndimage import convolve

DATA_DIR = "/home/edier/Documents/INVESTIGACION/PAPERS/ELABORACION/Espacial/DATA"
DEM_PATH = os.path.join(DATA_DIR, "dem_iguana_1m.tif")
POINTS_PATH = os.path.join(DATA_DIR, "Deslizamientos_Iguana.gpkg")
OUTPUT_TIF = os.path.join(DATA_DIR, "kde_5x5_iguana_1m.tif")

def main():
    print("Reading DEM metadata for reference grid...")
    with rasterio.open(DEM_PATH) as src:
        transform = src.transform
        meta = src.meta.copy()
        width = src.width
        height = src.height

    print("Loading landslide points...")
    gdf = gpd.read_file(POINTS_PATH)
    
    print("Rasterizing points...")
    point_grid = np.zeros((height, width), dtype=np.float32)
    rows, cols = rasterio.transform.rowcol(transform, gdf.geometry.x, gdf.geometry.y)
    
    # Ensure they are within bounds
    valid_mask = (np.array(rows) >= 0) & (np.array(rows) < height) & \
                 (np.array(cols) >= 0) & (np.array(cols) < width)
    rows = np.array(rows)[valid_mask]
    cols = np.array(cols)[valid_mask]

    # Add +1 for each point falling in a cell
    np.add.at(point_grid, (rows, cols), 1)

    print("Calculating 5x5 KDE (Cell + neighbors + neighbors of neighbors)...")
    kernel = np.ones((5, 5), dtype=np.float32)
    
    # Convolution with 5x5 square matrix exactly represents
    # the 8 direct neighbors and their adjacent neighbors up to 2 steps away
    kde_grid = convolve(point_grid, kernel, mode='constant', cval=0.0)

    # Average number of points per cell in the local 25m^2 area
    kde_density = kde_grid / 25.0

    print(f"Saving output to {OUTPUT_TIF}...")
    meta.update(dtype='float32', nodata=-9999)
    with rasterio.open(OUTPUT_TIF, 'w', **meta) as dst:
        dst.write(kde_density, 1)

    print("Done!")

if __name__ == "__main__":
    main()
