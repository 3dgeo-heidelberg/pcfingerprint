import pcfingerprint
from pathlib import Path
import time

start_time = time.time()
# load data
data_folder = Path("data/uls_tree_voxels")
report_folder = Path("output_report_uls_tree_voxels")
pc_pair = pcfingerprint.PointCloudPair(standardize_intensity=False, shift_z=True)
pc_pair.load_real_world(data_folder / "uls_reference.laz")
pc_pair.load_simulated(list(data_folder.glob("uls_detailed_voxel*.laz")))
print(f"Data loaded in {time.time() - start_time:.2f} seconds.")

pdens_start_time = time.time()
# compute point density and geometric features
pc_pair.compute_point_density(radius=0.5, backend="jakteristics")
print("Point density computed.")
print(f"Point density computed in {time.time() - pdens_start_time:.2f} seconds.")

geom_start_time = time.time()
pc_pair.compute_geometric_features(
    feature_names=["linearity"], radius=1.0, backend="pgeof"
)
print(f"Geometric features computed in {time.time() - geom_start_time:.2f} seconds.")

report_start_time = time.time()
# generate project report
pc_pair.generate_report(
    output_path=report_folder / "test_report.html",
    attribute_list=["z", "point_density", "linearity", "intensity"],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=[
        "histogram_overlay",
        "3d_scatter",
    ],  # "histogram" will instead plot histograms side by side
    vis_backend="matplotlib",
)
print(f"Report generated in {time.time() - report_start_time:.2f} seconds.")

# write point clouds to disk
pc_outfolder = Path("output_pcs/")
pc_pair.write_point_clouds(pc_outfolder)
print(f"Point clouds written to {pc_outfolder}")
