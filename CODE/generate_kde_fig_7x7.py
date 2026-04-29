import os
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.enums import Resampling
import geopandas as gpd
import matplotlib.pyplot as plt

def main():
    root_dir = "/home/edier/Documents/INVESTIGACION/PAPERS/ELABORACION/Espacial"
    data_dir = os.path.join(root_dir, "DATA")
    fig_dir = os.path.join(root_dir, "FIG")

    os.makedirs(fig_dir, exist_ok=True)

    basin_path = os.path.join(data_dir, "Cuenca_Iguana.shp")
    kde_path = os.path.join(data_dir, "kde_7x7_iguana_1m.tif")
    out_fig = os.path.join(fig_dir, "kde_7x7_density_map.png")

    print("Loading basin boundary...")
    basin = gpd.read_file(basin_path)

    print("Preparing figure...")
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    basin.plot(ax=ax, facecolor='lightgray', edgecolor='black', linewidth=1, alpha=0.5)

    print("Loading KDE raster (fast downsampling with average resampling)...")
    scale_factor = 0.05  # downsample to ~5% size for performance

    with rasterio.open(kde_path) as src:
        data = src.read(
            1,
            out_shape=(
                int(src.height * scale_factor),
                int(src.width * scale_factor)
            ),
            resampling=Resampling.average
        )

        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0])
        )

        # Mask zeros / near-zero float-precision artifacts
        data[data < 1e-5] = np.nan

        print("Plotting KDE overlay...")
        show(data, transform=transform, ax=ax, cmap='Reds',
             vmin=0, vmax=np.nanpercentile(data, 99))

    ax.set_title("Movimientos en Masa: KDE 7x7 Neighborhood Density")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.grid(True, linestyle='--', alpha=0.5)

    print(f"Saving figure to {out_fig}...")
    plt.tight_layout()
    plt.savefig(out_fig, bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure successfully generated.")

if __name__ == "__main__":
    main()

