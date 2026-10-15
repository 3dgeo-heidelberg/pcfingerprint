from pathlib import Path

import pcfingerprint

# define input and output paths
script_path = Path(__file__).parent
data_folder = script_path / "../tests/data/uls_tree_voxels"
report_folder = script_path / "../output_report_uls_tree_voxels"
report_folder_matplotlib = script_path / "../output_report_uls_tree_voxels_matplotlib"
pc_outfolder = script_path / "../output_pcs/"

# initialize PointCloudPair and load data
pc_pair = pcfingerprint.PointCloudPair(standardize_intensity=False, shift_z=True)
pc_pair.load_reference(data_folder / "uls_reference.laz")
pc_pair.load_compare(list(data_folder.glob("uls_detailed_voxel*.laz")))

# compute density and geometric features
pc_pair.compute_geometric_features(
    feature_names=["density"], radius=0.5, backend="jakteristics"
)

pc_pair.compute_geometric_features(
    feature_names=["linearity"], radius=1.0, backend="pgeof"
)

# generate project report
pc_pair.generate_report(
    output_folder=report_folder,
    attribute_list=["z", "density_r0.5", "linearity_r1.0", "intensity"],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=[
        "histogram_overlay",
        "3d_scatter",
    ],
    vis_backend="plotly",
)

# generate project report
pc_pair.generate_report(
    output_folder=report_folder_matplotlib,
    attribute_list=["z", "density_r0.5", "linearity_r1.0", "intensity"],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=[
        "histogram_overlay",
        "3d_scatter",
    ],
    vis_backend="matplotlib",
    preview=False,
)

# write point clouds to disk
pc_pair.write_point_clouds(pc_outfolder)
