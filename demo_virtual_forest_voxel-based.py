import pcfingerprint
from pathlib import Path
import time
from pcfingerprint.pcfingerprint import write_voxelized_point_clouds

start_time = time.time()
# load data
data_folder = Path("data/forest")
report_folder = Path("output_report_tls_virtual_forest_voxel")
pc_pair = pcfingerprint.PointCloudPair(standardize_intensity=False, shift_z=True)
pc_pair.load_real_world(data_folder / "plot_10_original_s.laz")
pc_pair.load_simulated(list(data_folder.glob("plot_10_simulated_s.laz")))
print(f"Data loaded in {time.time() - start_time:.2f} seconds.")

# compute voxel features
geom_start_time = time.time()
vapc_real, vapc_sim = pc_pair.compute_voxel_features(
    voxel_size=0.2,
    voxel_origin=[0, 0, 0],
    feature_names=["density", "linearity", "sum_of_eigenvalues"],
)
print(f"Voxel features computed in {time.time() - geom_start_time:.2f} seconds.")

assert (
    pc_pair.real_world.geometric_features["density_v0.2"] is not None
), "Voxel density not computed for real world point cloud."
assert (
    pc_pair.simulated[0].geometric_features["density_v0.2"] is not None
), "Voxel density not computed for simulated point cloud."
assert any(
    "linearity_v" in key for key in pc_pair.real_world.geometric_features
), "Voxel linearity not computed for real world point cloud."
assert any(
    "linearity_v" in key for key in pc_pair.simulated[0].geometric_features
), "Voxel linearity not computed for simulated point cloud."

report_start_time = time.time()
# generate project report
pc_pair.generate_report(
    output_path=report_folder / "test_report.html",
    attribute_list=["density", "linearity"],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=[
        "histogram_overlay",
    ],
    vis_backend="matplotlib",
)
print(f"Report generated in {time.time() - report_start_time:.2f} seconds.")

# filter to only class 1 on a fresh point-wise pair before voxel computations
pc_pair_filtered = pcfingerprint.PointCloudPair(
    standardize_intensity=False, shift_z=True
)
pc_pair_filtered.load_real_world(data_folder / "plot_10_original_s.laz")
pc_pair_filtered.load_simulated(list(data_folder.glob("plot_10_simulated_s.laz")))
pc_pair_filtered = pc_pair_filtered.filter_by_class([1])
report_folder_class1 = Path("output_report_tls_virtual_forest_voxel_class1")

geom_start_time = time.time()
vapc_real_class1, vapc_sim_class1 = pc_pair_filtered.compute_voxel_features(
    voxel_size=0.05,
    voxel_origin=[0, 0, 0],
    feature_names=[
        "density",
        "linearity",
        "planarity",
        "sphericity",
        "sum_of_eigenvalues",
        "omnivariance",
        "eigenentropy",
        "anisotropy",
        "surface_variation",
    ],
)
print(
    f"Voxel features (class 1) computed in {time.time() - geom_start_time:.2f} seconds."
)

# generate report only for one class
pc_pair_filtered.generate_report(
    output_path=report_folder_class1 / "test_report.html",
    attribute_list=[
        "density",
        "linearity",
        "planarity",
        "sphericity",
        "sum_of_eigenvalues",
        "omnivariance",
        "eigenentropy",
        "anisotropy",
        "surface_variation",
    ],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=[
        "histogram_overlay",
    ],
    vis_backend="matplotlib",
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
