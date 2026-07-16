import pcfingerprint
from pathlib import Path

# define input and output paths
script_path = Path(__file__).parent
data_folder = script_path / "../data/forest"
report_folder = script_path / "../output_report_tls_virtual_forest"
pc_outfolder = script_path / "../output_pcs/"

# define class mapping for the point cloud classification labels
class_mapping = {0: "grass", 1: "ground", 2: "trunk", 3: "canopy"}

# initialize PointCloudPair and load data
pc_pair = pcfingerprint.PointCloudPair(
    standardize_intensity=False, shift_z=True, class_mapping=class_mapping
)
pc_pair.load_reference(data_folder / "plot_10_original_s.laz")
pc_pair.load_compare(list(data_folder.glob("plot_10_simulated_s.laz")))

# compute density and geometric features
pc_pair.compute_geometric_features(
    feature_names=["density"],
    radius=0.2,
    backend="jakteristics",
    chunking=True,
    max_points_per_chunk=1_000_000,
)

pc_pair.compute_geometric_features(
    feature_names=["linearity", "planarity"],
    radius=0.2,
    backend="pgeof",
    chunking=True,
    max_points_per_chunk=1_000_000,
)

pc_pair.compute_voxel_features(feature_names=["linearity"], voxel_size=0.2)

# generate project report
pc_pair.generate_report(
    output_folder=report_folder,
    attribute_list=[
        "z",
        "density_r0.2",
        "linearity_r0.2",
        "planarity_r0.2",
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

# write point clouds to disk
pc_pair.write_point_clouds(pc_outfolder)
