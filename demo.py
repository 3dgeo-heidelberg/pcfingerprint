import pcfingerprint
from pathlib import Path

# load data
pc_pair = pcfingerprint.PointCloudPair()
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
    attribute_list=["z", "point_density", "linearity", "number_of_returns"],
    summary_statistics=True,
    visualizations=["histogram"],
    vis_backend="matplotlib"
)
