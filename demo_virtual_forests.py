import pcfingerprint
from pathlib import Path
import time

start_time = time.time()
# load data
data_folder = Path("data/forest")
report_folder = Path("output_report_tls_virtual_forest")
pc_pair = pcfingerprint.PointCloudPair(standardize_intensity=False, shift_z=True)
pc_pair.load_real_world(data_folder / "plot_10_original_s.laz")
pc_pair.load_simulated(list(data_folder.glob("plot_10_simulated_s.laz")))
print(f"Data loaded in {time.time() - start_time:.2f} seconds.")

pdens_start_time = time.time()
# compute point density and geometric features
pc_pair.compute_point_density(radius=0.2, backend="jakteristics")
print(f"Point density computed in {time.time() - pdens_start_time:.2f} seconds.")

geom_start_time = time.time()
pc_pair.compute_geometric_features(
    feature_names=["linearity"], radius=0.2, backend="pgeof"
)
print(f"Geometric features computed in {time.time() - geom_start_time:.2f} seconds.")

assert (
    pc_pair.real_world.point_density is not None
), "Point density not computed for real world point cloud."
assert (
    pc_pair.simulated[0].point_density is not None
), "Point density not computed for simulated point cloud."

report_start_time = time.time()
# generate project report
pc_pair.generate_report(
    output_folder=report_folder,
    attribute_list=["z", "point_density", "linearity"],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=[
        "histogram_overlay",
        "3d_scatter",
        "histogram_per_class",
    ],
    vis_backend="matplotlib",
)
print(f"Report generated in {time.time() - report_start_time:.2f} seconds.")

pc_pair_filtered = pc_pair.filter_by_class([1])  # filter to only class 1
report_folder_class1 = Path("output_report_tls_virtual_forest_class1")
# generate report only for one class
pc_pair_filtered.generate_report(
    output_folder=report_folder_class1,
    attribute_list=["z", "point_density", "linearity"],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=[
        "histogram_overlay",
        "3d_scatter",
    ],
    vis_backend="matplotlib",
)
print(f"Report generated in {time.time() - report_start_time:.2f} seconds.")

# write point clouds to disk
pc_outfolder = Path("output_pcs/")
pc_pair.write_point_clouds(pc_outfolder)
print(f"Point clouds written to {pc_outfolder}")
