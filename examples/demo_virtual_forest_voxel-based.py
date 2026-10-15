from pathlib import Path

import pcfingerprint
from pcfingerprint.pcfingerprint import write_voxelized_point_clouds

# define input and output paths
script_path = Path(__file__).parent
data_folder = script_path / "../data/forest"
report_folder = script_path / "../output_report_tls_virtual_forest_voxel"
pc_outfolder = script_path / "../output_pcs_forest_voxel/"

# define class mapping for the point cloud classification labels
class_mapping = {0: "grass", 1: "ground", 2: "trunk", 3: "canopy", 4: "deadwood"}

# initialize PointCloudPair and load data
pc_pair = pcfingerprint.PointCloudPair(
    standardize_intensity=False, shift_z=True, class_mapping=class_mapping
)
pc_pair.load_reference(data_folder / "plot_09_original_1.laz")
pc_pair.load_compare(list(data_folder.glob("plot_09_simulated_1.laz")))

# compute voxel features
voxel_size = 0.5
vapc_reference, vapc_compare = pc_pair.compute_voxel_features(
    voxel_size=voxel_size,
    voxel_origin=[0, 0, 0],
    feature_names=["density", "linearity", "sum_of_eigenvalues"],
)

# generate project report
pc_pair.generate_report(
    output_folder=report_folder,
    attribute_list=[
        f"density_v{voxel_size}",
        f"linearity_v{voxel_size}",
        "z",
        "classification",
    ],
    summary_statistics=True,
    return_number_metrics=False,
    visualizations=[
        "histogram_overlay",
        "histogram_per_class",
        "3d_scatter",
    ],
    vis_backend="plotly",
    class_distribution=True,
    preview=True,
)

# write point clouds to disk
pc_outfolder.mkdir(exist_ok=True, parents=True)
write_voxelized_point_clouds(
    vapc_reference,
    pc_outfolder / f"{pc_pair.reference.source.name}",
    reduce_to_feature="closest_to_centroids",
)
for ivapc, vapc in enumerate(vapc_compare):
    write_voxelized_point_clouds(
        vapc,
        pc_outfolder / f"{pc_pair.compare[ivapc].source.name}",
        reduce_to_feature="closest_to_centroids",
    )
