import pcfingerprint
from pathlib import Path

# define input and output paths
script_path = Path(__file__).parent
data_folder = script_path / "../data/inf_trees"
report_folder = script_path / "../output_report_inf_trees"
pc_outfolder = script_path / "../output_pcs/"

# initialize PointCloudPair and load data
pc_pair = pcfingerprint.PointCloudPair(standardize_intensity=True, shift_z=False)
pc_pair.load_reference(data_folder / "INF_trees_VZ600i_260317_merged_local.laz")
pc_pair.load_compare(
    list(data_folder.glob("output_tree1_mesh_0.10ns*_merged_trees.laz"))
)

pc_pair.compute_c2c_distances(max_distance=1.0)

# pc_pair.compute_geometric_features(
#    feature_names=["linearity"], radius=0.1, backend="pgeof"
# )

pc_pair.compute_voxel_features(
    feature_names=["density", "linearity", "planarity"], voxel_size=0.2
)
pc_pair.compute_voxel_features(
    feature_names=["linearity", "planarity"], voxel_size=0.05
)

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
        # "linearity_r0.1",
    ],
    distance_list=["c2c"],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=[
        "histogram_overlay",
        "3d_scatter",
    ],
    vis_backend="plotly",
)

# write point clouds to disk
pc_pair.write_point_clouds(pc_outfolder)
