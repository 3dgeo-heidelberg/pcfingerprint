import pcfingerprint
from pathlib import Path

# load data
data_folder = Path("data/uls_tree_voxels")
report_folder = Path("output_report_uls_tree_voxels")
pc_pair = pcfingerprint.PointCloudPair(standardize_intensity=False, shift_z=True)
pc_pair.load_real_world(data_folder / "uls_reference.laz")
pc_pair.load_simulated(list(data_folder.glob("uls_detailed_voxel*.laz")))

# compute point density and geometric features
pc_pair.compute_point_density(radius=0.5, backend="pgeof")
print("Point density computed.")
pc_pair.compute_geometric_features(
    feature_names=[
        "linearity"
    ],
    radius=1.0,
    backend="pgeof"
)
print("Geometric features computed.")

# generate project report
pc_pair.generate_report(
    output_path=report_folder / "test_report.html",
    attribute_list=["z", "point_density", "linearity", "intensity"],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=["histogram_overlay", "3d_scatter"],  # "histogram" will instead plot histograms side by side
    vis_backend="matplotlib"
)
print("Report generated.")

# write point clouds to disk
pc_outfolder = Path("output_pcs/")
pc_pair.write_point_clouds(pc_outfolder)
print(f"Point clouds written to {pc_outfolder}")
