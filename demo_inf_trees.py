import pcfingerprint
from pathlib import Path
import time

start_time = time.time()
# load data
data_folder = Path("data/inf_trees")
report_folder = Path("output_report_inf_trees")
pc_pair = pcfingerprint.PointCloudPair(standardize_intensity=True, shift_z=False)
pc_pair.load_reference(data_folder / "INF_trees_VZ600i_260317_merged_local.laz")
pc_pair.load_compare(
    list(data_folder.glob("output_tree1_mesh_0.10ns*_merged_trees.laz"))
)

geom_start_time = time.time()
pc_pair.compute_geometric_features(
    feature_names=["linearity"], radius=0.1, backend="pgeof"
)

voxel_start_time = time.time()
pc_pair.compute_voxel_features(
    feature_names=["density", "linearity", "planarity"], voxel_size=0.2
)
pc_pair.compute_voxel_features(
    feature_names=["linearity", "planarity"], voxel_size=0.05
)


report_start_time = time.time()
# generate project report
pc_pair.generate_report(
    output_folder=report_folder,
    attribute_list=[
        "z",
        "return_number",
        "number_of_returns",
        "density_v0.2",
        "linearity_v0.2",
        "planarity_v0.2",
        "linearity_v0.05",
        "planarity_v0.05",
        "linearity_r0.1",
    ],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=[
        "histogram_overlay",
        "3d_scatter",
    ],  # "histogram" will instead plot histograms side by side
    vis_backend="plotly",
)

# write point clouds to disk
pc_outfolder = Path("output_pcs/")
pc_pair.write_point_clouds(pc_outfolder)
