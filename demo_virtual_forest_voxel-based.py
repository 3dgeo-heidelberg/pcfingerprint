import pcfingerprint
from pathlib import Path
import time
from pcfingerprint.pcfingerprint import write_voxelized_point_clouds

start_time = time.time()
# load data
data_folder = Path("data/forest")
report_folder = Path("output_report_tls_virtual_forest_voxel")
class_mapping = {0: "grass", 1: "ground", 2: "trunk", 3: "canopy", 4: "deadwood"}
pc_pair = pcfingerprint.PointCloudPair(
    standardize_intensity=False, shift_z=True, class_mapping=class_mapping
)
pc_pair.load_real_world(data_folder / "plot_09_original_1.laz")
pc_pair.load_simulated(list(data_folder.glob("plot_09_simulated_1.laz")))
print(f"Data loaded in {time.time() - start_time:.2f} seconds.")

# compute voxel features
voxel_size = 1.0
voxel_start_time = time.time()
vapc_real, vapc_sim = pc_pair.compute_voxel_features(
    voxel_size=voxel_size,
    voxel_origin=[0, 0, 0],
    feature_names=["density", "linearity", "sum_of_eigenvalues"],
)
print(f"Voxel features computed in {time.time() - voxel_start_time:.2f} seconds.")

report_start_time = time.time()
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
print(f"Report generated in {time.time() - report_start_time:.2f} seconds.")

# write point clouds to disk
pc_outfolder = Path("output_pcs_forest_voxel/")
pc_outfolder.mkdir(exist_ok=True, parents=True)
write_voxelized_point_clouds(
    vapc_real,
    pc_outfolder / f"{pc_pair.real_world.source.name}",
    reduce_to_feature="closest_to_centroids",
)
for ivapc, vapc in enumerate(vapc_sim):
    write_voxelized_point_clouds(
        vapc,
        pc_outfolder / f"{pc_pair.simulated[ivapc].source.name}",
        reduce_to_feature="closest_to_centroids",
    )
print(f"Point clouds written to {pc_outfolder}")
