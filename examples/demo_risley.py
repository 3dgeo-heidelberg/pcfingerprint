import pcfingerprint
from pathlib import Path

# define input and output paths
script_path = Path(__file__).parent
data_folder = script_path / "../data/livox_avia"
report_folder = script_path / "../output_report_livox_avia"
pc_outfolder = script_path / "../output_pcs_risley/"

# initialize PointCloudPair and load data
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
    ],
    vis_backend="plotly",
)

# write point clouds to disk
pc_pair.write_point_clouds(pc_outfolder)
