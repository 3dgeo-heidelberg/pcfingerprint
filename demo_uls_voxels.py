import pcfingerprint
from pathlib import Path
import time

start_time = time.time()
# load data
data_folder = Path("data/uls_tree_voxels")
report_folder = Path("output_report_uls_tree_voxels")
pc_pair = pcfingerprint.PointCloudPair(standardize_intensity=False, shift_z=True)
pc_pair.load_reference(data_folder / "uls_reference.laz")
pc_pair.load_compare(list(data_folder.glob("uls_detailed_voxel*.laz")))

pdens_start_time = time.time()
# compute density and geometric features
pc_pair.compute_geometric_features(
    feature_names=["density"], radius=0.5, backend="jakteristics"
)

geom_start_time = time.time()
pc_pair.compute_geometric_features(
    feature_names=["linearity"], radius=1.0, backend="pgeof"
)

report_start_time = time.time()
# generate project report
pc_pair.generate_report(
    output_folder=report_folder,
    attribute_list=["z", "density_r0.5", "linearity_r1.0", "intensity"],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=[
        "histogram_overlay",
        "3d_scatter",
    ],  # "histogram" will instead plot histograms side by side
    vis_backend="plotly",
)

report_folder = Path("output_report_uls_tree_voxels_matplotlib")
report_start_time = time.time()
# generate project report
pc_pair.generate_report(
    output_folder=report_folder,
    attribute_list=["z", "density_r0.5", "linearity_r1.0", "intensity"],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=[
        "histogram_overlay",
        "3d_scatter",
    ],  # "histogram" will instead plot histograms side by side
    vis_backend="matplotlib",
    preview=False,
)

# write point clouds to disk
pc_outfolder = Path("output_pcs/")
pc_pair.write_point_clouds(pc_outfolder)
