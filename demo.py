import pcfingerprint

# load data
pc_pair = pcfingerprint.PointCloudPair()
pc_pair.load_real_world("data/real_world.laz")
pc_pair.load_simulated("data/simulated.laz")

# compute point density and geometric features
pc_pair.compute_point_density(radius=0.05)
print("Point density computed.")
pc_pair.compute_geometric_features(
    feature_names=[
        "linearity"
    ],
    radius=0.2,
)
print("Geometric features computed.")

# generate project report
pc_pair.generate_report(
    sim_idx=0,
    output_path="output_report/test_report.html",
    attribute_list=["z", "point_density", "linearity"],
    summary_statistics=True,
    visualizations=["histogram"],
    vis_backend="matplotlib"
)
