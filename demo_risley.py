import pcfingerprint
from pathlib import Path

# load data
data_folder = Path("data/livox_avia")
report_folder = Path("output_report_livox_avia")
pc_pair = pcfingerprint.PointCloudPair(shift_gps_time=True)
pc_pair.load_reference(data_folder / "avia_office_wall_real-world.csv")
pc_pair.load_compare(list(data_folder.glob("avia*.las")))

# generate project report
pc_pair.generate_report(
    output_folder=report_folder,
    attribute_list=["gps_time"],
    summary_statistics=True,
    visualizations=[
        "histogram_overlay",
        "3d_scatter",
    ],  # "histogram" will instead plot histograms side by side
    vis_backend="plotly",
)
print("Report generated.")

# write point clouds to disk
pc_outfolder = Path("output_pcs_risley/")
pc_pair.write_point_clouds(pc_outfolder)
print(f"Point clouds written to {pc_outfolder}")
