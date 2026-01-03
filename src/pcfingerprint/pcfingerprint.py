import warnings
import numpy as np
import laspy
from pcfingerprint.geometric_features import compute_point_density, compute_geom_feat
from pcfingerprint.report import generate_report
from pathlib import Path


class PointCloud:
    def __init__(self, 
    coords=None,
    gps_time=None,
    return_number=None,
    number_of_returns=None,
    intensity=None):
        self.coords = coords  # Nx3 array of point coordinates
        self.source = None  # source file path
        self.kd_tree = None  # KDTree for spatial queries
        self.gps_time = gps_time  # float array
        self.return_number = return_number  # int array
        self.number_of_returns = number_of_returns  # int array
        self.intensity = intensity  # int array
        self.point_density = None  # float array
        self.point_density_radius = None  # radius used for point density
        self.normals = None  # Nx3 array of normal vectors
        self.normal_radius = None  # radius used for normal estimation
        self.geometric_features = {}  # dict of feature name to float array
        self.voxel_size = None  # size of voxels (float) 
        self.voxel_indices = None  # Nx3 array of voxel indices (int)
        self.is_outlier = None  # boolean array


class PointCloudPair:
    def __init__(self):
        self.real_world = PointCloud()
        # simulated is a list of PointClouds (minimum 1)
        self.simulated = []
        self.trajectory = None  # Optional trajectory data
    
    def drop_attributes(self):
        # drop attributes that are not present in all point clouds
        all_pcs = [self.real_world] + self.simulated
        for attr in [
            "gps_time",
            "return_number",
            "number_of_returns",
            "intensity",
        ]:
            if any(getattr(pc, attr) is None for pc in all_pcs):
                for pc in all_pcs:
                    setattr(pc, attr, None)

    def load_real_world(self, filepath: str):
        self.real_world = read_las_file(filepath)
        self.real_world.source = filepath
        # if simulated point cloud is already loaded, drop non-overlapping attributes
        if self.simulated != []:
            self.drop_attributes()
    
    def load_simulated(self, filepaths: list[str] | str):
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        for filepath in filepaths:
            self.simulated.append(read_las_file(filepath))
            self.simulated[-1].source = filepath
        # if real world point cloud is already loaded, drop non-overlapping attributes
        if self.real_world.coords is not None:
            self.drop_attributes()

    def compute_point_density(self, radius: float = 2.0, nthreads: int = -1):
        # Compute point density for all point clouds

        # check if point density already computed
        radius = np.round(radius, 3)
        if self.real_world.point_density_radius == radius and self.real_world.point_density is not None:
            return  # point density already computed
        else:
            compute_point_density(self.real_world, radius, nthreads)
            for sim_pc in self.simulated:
                compute_point_density(sim_pc, radius, nthreads)
    
    def compute_geometric_features(self, feature_names: list[str], radius: float = 2.0, nthreads: int = -1):
        # Compute geometric features for all point clouds
        
        # check if features already computed
        for f in feature_names:
            radius = np.round(radius, 3)
            f_with_radius = f + f"_r{radius}"
            if f_with_radius in self.real_world.geometric_features:
                # drop from feature_names
                feature_names.remove(f)
        if feature_names == []:
            return  # all features already computed

        compute_geom_feat(self.real_world, feature_names, radius, nthreads)
        for sim_pc in self.simulated:
            compute_geom_feat(sim_pc, feature_names, radius, nthreads)
    
    def plot_histogram(self, attribute: str, bins: int = 100, backend: str = "matplotlib"):
        if backend == "matplotlib":
            from pcfingerprint.vis_matplotlib import plot_histogram
            fig = plot_histogram(self.real_world, self.simulated[0], attribute, bins)
        elif backend == "plotly":
            pass
        else:
            raise ValueError(f"Backend {backend} not supported.")
        return fig
    
    def generate_report(self, output_path: str, attribute_list: list[str], summary_statistics: bool = True, visualizations: list[str] = ["histogram"], vis_backend: str = "matplotlib"):
        # generate html report with histograms for each attribute in attribute_list
        if len(self.simulated) > 5:
            warnings.warn("More than 5 simulated point clouds provided. Report will only consider the first five.")
            simulated = self.simulated[:5]
        else:
            simulated = self.simulated
        generate_report(self.real_world, simulated, output_path, attribute_list, summary_statistics=summary_statistics, visualizations=visualizations, vis_backend=vis_backend)

    def write_point_clouds(self, output_folder: str):
        # write real world and simulated point clouds (with new attributes) to output folder
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        # write real world
        real_outpath = output_folder / Path(self.real_world.source).name
        write_las(self.real_world, real_outpath)
        # write simulated
        for sim_pc in self.simulated:
            sim_outpath = output_folder / Path(sim_pc.source).name
            write_las(sim_pc, sim_outpath)


def read_las_file(filepath):
    """Create a PointCloud from a LAS/LAZ file

    :param filepath:
        The filename to read from. It is expected to be in LAS/LAZ format
        and will be processed using laspy.
    :type filepath: str
    :return:
        A PointCloud object containing the data from the file.
    :rtype: PointCloud
    """

    # Read the lasfile using laspy
    las = laspy.read(filepath)

    # get gpstime, return number, number of returns, intensity if available
    gps_time = las.gps_time if "gps_time" in las.point_format.dimension_names else None
    return_number = las.return_number if "return_number" in las.point_format.dimension_names else None
    number_of_returns = las.number_of_returns if "number_of_returns" in las.point_format.dimension_names else None
    intensity = las.intensity if "intensity" in las.point_format.dimension_names else None

    # create new PointCloud and assign attributes
    pc = PointCloud()
    pc.coords = np.vstack((las.x, las.y, las.z)).T
    pc.gps_time = gps_time
    pc.return_number = return_number
    pc.number_of_returns = number_of_returns
    pc.intensity = intensity

    return pc


def write_las(
        pc: PointCloud,
        outfile: str,
        las_point_format=7,
        las_version="1.4",
        las_scales=None,
        las_offset=None,
    ):
    """
    Docstring for write_las
    Modified after vapc: https://github.com/3dgeo-heidelberg/vapc/blob/main/src/vapc/datahandler.py#L97-L187
    
    :param pc: Description
    :type pc: PointCloud
    :param outfile: Description
    :type outfile: str
    :param las_point_format: Description
    :param las_version: Description
    :param las_scales: Description
    :param las_offset: Description
    """
    if las_scales is None:
        las_scales = [0.00025, 0.00025, 0.00025]
    if las_offset is None:
        las_offset = np.min(pc.coords, axis=0)
    
    new_header = laspy.LasHeader(point_format=las_point_format, version=las_version)
    new_header.offsets = las_offset
    new_header.scales = las_scales

    las = laspy.LasData(new_header)
    las.x = pc.coords[:, 0]
    las.y = pc.coords[:, 1]
    las.z = pc.coords[:, 2]

    # standard las fields: gps_time, return_number, number_of_returns, intensity
    standard_fields = ["gps_time", "return_number", "number_of_returns", "intensity"]
    for field in standard_fields:
        data = getattr(pc, field)
        if data is not None:
            las[field] = data
    # some attributes first require attribute name handling by adding a suffix (e.g., search radius)
    # point density and normals
    if pc.point_density is not None:
        pdens_name = f"point_density_r{pc.point_density_radius}"
        add_dimension_to_laz(las, pc.point_density, pdens_name)
    if pc.normals is not None:
        normal_x_name = f"normal_x_r{pc.normal_radius}"
        normal_y_name = f"normal_y_r{pc.normal_radius}"
        normal_z_name = f"normal_z_r{pc.normal_radius}"
        add_dimension_to_laz(las, pc.normals[:, 0], normal_x_name)
        add_dimension_to_laz(las, pc.normals[:, 1], normal_y_name)
        add_dimension_to_laz(las, pc.normals[:, 2], normal_z_name)
    # geometric features (here the feature name already includes the radius suffix)
    for feat_name, feat_data in pc.geometric_features.items():
        add_dimension_to_laz(las, feat_data, feat_name)

    las.write(outfile)


def add_dimension_to_laz(las_file, array, name):
        """
        Adds a new attribute dimension to the LAZ file.

        Writes a new attribute to the LAZ file by adding an extra dimension with the specified name and data type.

        :param array: The data array to be added as a new attribute.
        :param name: The name of the new attribute.
        :type array: np.ndarray
        :type name: str

        :return: None
        :rtype: None
        """
        las_file.add_extra_dim(
            laspy.ExtraBytesParams(name=name, type=array.dtype, description=name)
        )
        las_file[name] = array


def get_attribute_data(pc: PointCloud, attribute: str):
    """
    Function to get the data for a given attribute from a PointCloud.
    The attribute may be 
    - a direct attribute of the PointCloud object,
    - one of the coordinate axes ("x", "y", "z"),
    - a geometric feature (e.g. "linearity_0.2")
    
    :param pc: PointCloud object
    :type pc: PointCloud
    :param attribute: Name of the attribute to retreive
    :type attribute: str
    """
    # make attribute lower case
    attribute = attribute.lower()

    # search for attribute in all possible locations
    if hasattr(pc, attribute):
        data = getattr(pc, attribute)
    elif attribute in ["x", "y", "z"]:  # similarly for normals?
        coord_index = {"x": 0, "y": 1, "z": 2}[attribute]
        data = pc.coords[:, coord_index]
    elif any(attribute in key for key in pc.geometric_features.keys()):
        geom_feats = [key for key in pc.geometric_features.keys() if attribute in key]
        if len(geom_feats) > 1:
            warnings.warn(f"Multiple geometric features match attribute {attribute}: {geom_feats}. Using first match.\nConsider using full feature name with radius suffix.\n"
                          f"Available geometric features: {list(pc.geometric_features.keys())}")
        data = pc.geometric_features[geom_feats[0]]
    else:
        raise ValueError(f"Attribute {attribute} not found in point clouds.")

    return data
