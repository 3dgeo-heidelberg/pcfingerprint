import pcfingerprint
from pathlib import Path

# load data
pc_pair = pcfingerprint.PointCloudPair(standardize_intensity=False, shift_z=True)
pc_pair.load_real_world("data/uls_reference.laz")
pc_pair.load_simulated(list(Path("data").glob("uls_detailed_voxel*.laz")))

# compute point density and geometric features
pc_pair.compute_point_density(radius=0.5)
print("Point density computed.")
pc_pair.compute_geometric_features(
    feature_names=[
        "linearity"
    ],
    radius=1.0,
)
print("Geometric features computed.")

# generate project report
pc_pair.generate_report(
    output_path="output_report/test_report.html",
    attribute_list=["z", "point_density", "linearity", "intensity"],
    summary_statistics=True,
    return_number_metrics=True,
    visualizations=["histogram_overlay", "3d_scatter"],  # "histogram" will instead plot histograms side by side
    vis_backend="matplotlib"
)
print("Report generated.")

# write point clouds to disk
pc_outfolder = "output_pcs/"
pc_pair.write_point_clouds(pc_outfolder)
print(f"Point clouds written to {pc_outfolder}")
