import os
import geopandas as gpd
import matplotlib.pyplot as plt

def main():
    root_dir = "/home/edier/Documents/INVESTIGACION/PAPERS/ELABORACION/Espacial"
    data_dir = os.path.join(root_dir, "DATA")
    fig_dir = os.path.join(root_dir, "FIG")
    
    os.makedirs(fig_dir, exist_ok=True)
    
    basin_path = os.path.join(data_dir, "Cuenca_Iguana.shp")
    points_path = os.path.join(data_dir, "Deslizamientos_Iguana.gpkg")
    out_fig = os.path.join(fig_dir, "kde_density_map.png")
    
    print("Loading basin boundary and points...")
    basin = gpd.read_file(basin_path)
    points = gpd.read_file(points_path)
    
    print("Preparing figure...")
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    basin.plot(ax=ax, facecolor='lightgray', edgecolor='black', linewidth=1, alpha=0.5)
    
    # We plot the points acting effectively as the 3x3m KDE discretizations at this zoom level
    points.plot(ax=ax, color='red', markersize=2, alpha=0.7, label='Densidad de Movimientos (3x3 KDE proxy)')
    
    ax.set_title("Movimientos en Masa: KDE 3x3 Neighborhood Density")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    print(f"Saving figure to {out_fig}...")
    plt.tight_layout()
    plt.savefig(out_fig, bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure successfully generated.")

if __name__ == "__main__":
    main()
