import pcfingerprint
from pathlib import Path
import time
from pcfingerprint.pcfingerprint import write_voxelized_point_clouds

start_time = time.time()
# load data
data_folder = Path("data/forest")
report_folder = Path("output_report_tls_virtual_forest_voxel")
pc_pair = pcfingerprint.PointCloudPair(standardize_intensity=False, shift_z=True)
pc_pair.load_real_world(data_folder / "plot_09_original_1.laz")
pc_pair.load_simulated(list(data_folder.glob("plot_09_simulated_1.laz")))
print(f"Data loaded in {time.time() - start_time:.2f} seconds.")

# compute voxel features
voxel_size = 0.5
geom_start_time = time.time()
vapc_real, vapc_sim = pc_pair.compute_voxel_features(
    voxel_size=0.5,
    voxel_origin=[0, 0, 0],
    feature_names=["density", "linearity", "sum_of_eigenvalues"],
)
print(f"Voxel features computed in {time.time() - geom_start_time:.2f} seconds.")

print(pc_pair.real_world.voxel_features.keys())

assert (
    pc_pair.real_world.voxel_features[f"density_v{voxel_size}"] is not None
), "Voxel density not computed for real world point cloud."
assert (
    pc_pair.simulated[0].voxel_features[f"density_v{voxel_size}"] is not None
), "Voxel density not computed for simulated point cloud."
assert any(
    "linearity_v" in key for key in pc_pair.real_world.voxel_features
), "Voxel linearity not computed for real world point cloud."
assert any(
    "linearity_v" in key for key in pc_pair.simulated[0].voxel_features
), "Voxel linearity not computed for simulated point cloud."

report_start_time = time.time()
# generate project report
pc_pair.generate_report(
    output_folder=report_folder,
    attribute_list=[f"density_v{voxel_size}", f"linearity_v{voxel_size}", "z"],
    summary_statistics=True,
    return_number_metrics=False,
    visualizations=[
        "histogram_overlay",
        "histogram_per_class",
        "3d_scatter",
    ],
    vis_backend="plotly",
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
