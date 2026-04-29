import pcfingerprint
from pathlib import Path
import time

start_time = time.time()
# load data
data_folder = Path("data/forest")
report_folder = Path("output_report_tls_virtual_forest")
class_mapping = {0: "grass", 1: "ground", 2: "trunk", 3: "canopy"}
pc_pair = pcfingerprint.PointCloudPair(
    standardize_intensity=False, shift_z=True, class_mapping=class_mapping
)
pc_pair.load_real_world(data_folder / "plot_10_original_test.laz")
pc_pair.load_simulated(list(data_folder.glob("plot_10_simulated_test.laz")))
print(f"Data loaded in {time.time() - start_time:.2f} seconds.")

pdens_start_time = time.time()
# compute density and geometric features
pc_pair.compute_geometric_features(
    feature_names=["density"], radius=0.2, backend="jakteristics"
)
print(f"Density computed in {time.time() - pdens_start_time:.2f} seconds.")

geom_start_time = time.time()
pc_pair.compute_geometric_features(
    feature_names=["linearity"], radius=0.2, backend="pgeof"
)
print(f"Geometric features computed in {time.time() - geom_start_time:.2f} seconds.")

voxel_start_time = time.time()
pc_pair.compute_voxel_features(feature_names=["linearity"], voxel_size=0.2)
print(f"Voxel features computed in {time.time() - voxel_start_time:.2f} seconds.")

report_start_time = time.time()
# generate project report
pc_pair.generate_report(
    output_folder=report_folder,
    attribute_list=[
        "z",
        "density_r0.2",
        "linearity_r0.2",
        "linearity_v0.2",
        "classification",
    ],
    summary_statistics=True,
    class_distribution=True,
    return_number_metrics=False,
    visualizations=[
        "histogram_overlay",
        "3d_scatter",
        "histogram_per_class",
    ],
    vis_backend="plotly",
)
print(f"Report generated in {time.time() - report_start_time:.2f} seconds.")

# write point clouds to disk
pc_outfolder = Path("output_pcs/")
pc_pair.write_point_clouds(pc_outfolder)
print(f"Point clouds written to {pc_outfolder}")
