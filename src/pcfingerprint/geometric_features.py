"""
Module for calculating geometric features on point clouds.
"""

import numpy as np
import warnings


def compute_point_density_jakteristics(
    point_cloud, radius: float, max_knn: int = 50_000
) -> None:
    """
    Compute (volume) point density using Jakteristics.

    :param point_cloud: PointCloud object
    :type point_cloud: PointCloud
    :param radius: Radius for the neighborhood search
    :type radius: float
    :param max_knn: Maximum number of nearest neighbors to consider
    :type max_knn: int
    """
    import jakteristics

    # limit radius to 3 decimal places
    radius = np.round(radius, 3)
    if not f"number_of_neighbors_r{radius}" in point_cloud.geometric_features:
        feats = jakteristics.compute_features(
            point_cloud.coords,
            search_radius=radius,
            max_k_neighbors=max_knn,
            feature_names=["number_of_neighbors"],
        )
        point_cloud.geometric_features[f"number_of_neighbors_r{radius}"] = (
            feats.reshape(-1)
        )
    point_cloud.point_density = point_cloud.geometric_features[
        f"number_of_neighbors_r{radius}"
    ] / (4 / 3 * np.pi * radius**3)
    point_cloud.point_density_radius = radius


def compute_point_density_pgeof(
    point_cloud, radius: float, max_knn: int = 50_000
) -> None:
    """
    Compute (volume) point density using PGEOF.

    :param point_cloud: PointCloud object
    :type point_cloud: PointCloud
    :param radius: Radius for the neighborhood search
    :type radius: float
    :param max_knn: Maximum number of nearest neighbors to consider
    :type max_knn: int
    """
    import pgeof

    # limit radius to 3 decimal places
    radius = np.round(radius, 3)
    # check if max_knn is larger than point cloud size
    max_knn = min(max_knn, point_cloud.coords.shape[0] - 1)

    if not f"number_of_neighbors_r{radius}" in point_cloud.geometric_features:
        indices, _ = pgeof.radius_search(
            point_cloud.coords.astype(np.float32),
            point_cloud.coords.astype(np.float32),
            search_radius=radius,
            max_knn=max_knn,
        )
        # number of neighbors is the number of valid indices returned for each point - invalid indices have value -1
        num_neigh = (indices != -1).sum(axis=1)
        point_cloud.geometric_features[f"number_of_neighbors_r{radius}"] = num_neigh
    point_cloud.point_density = point_cloud.geometric_features[
        f"number_of_neighbors_r{radius}"
    ] / (4 / 3 * np.pi * radius**3)
    point_cloud.point_density_radius = radius


def compute_geom_feat_jakteristics(
    point_cloud, feature_names: list[str], radius: float, max_knn: int = 50_000
) -> None:
    """
    Compute geometric features using Jakteristics.

    :param point_cloud: PointCloud object
    :type point_cloud: PointCloud
    :param feature_names: List of feature names to compute
    :type feature_names: list[str]
    :param radius: Radius for the neighborhood search
    :type radius: float
    :param max_knn: Maximum number of nearest neighbors to consider
    :type max_knn: int
    """
    import jakteristics

    # limit radius to 3 decimal places
    radius = np.round(radius, 3)
    # Compute geometric features
    feats = jakteristics.compute_features(
        point_cloud.coords,
        search_radius=radius,
        feature_names=feature_names,
        max_k_neighbors=max_knn,
    )
    if len(feature_names) == 1:
        feats = [feats]
    for i, f_name in enumerate(feature_names):
        point_cloud.geometric_features[f"{f_name}_r{radius}"] = feats[i].reshape(-1)


def compute_geom_feat_pgeof(
    point_cloud, feature_names: list[str], radius: float, max_knn: int = 50_000
) -> None:
    """
    Compute geometric features using PGEOF.

    :param point_cloud: PointCloud object
    :type point_cloud: PointCloud
    :param feature_names: List of feature names to compute
    :type feature_names: list[str]
        Available features: "linearity", "planarity", "scattering", "verticalitypgeof", "normalx", "normaly", "normalz", "verticality", "eigentropy"
    :param radius: Radius for the neighborhood search
    :type radius: float
    :param max_knn: Maximum number of nearest neighbors to consider
    :type max_knn: int
    """
    import pgeof
    from pgeof import EFeatureID

    # map from string feature names to pgeof EFeatureID
    feature_map = {
        "linearity": EFeatureID.Linearity,
        "planarity": EFeatureID.Planarity,
        "scattering": EFeatureID.Scattering,
        "verticalitypgeof": EFeatureID.VerticalityPGEOF,
        "normalx": EFeatureID.Normal_x,
        "normaly": EFeatureID.Normal_y,
        "normalz": EFeatureID.Normal_z,
        "curvature": EFeatureID.Curvature,
        "verticality": EFeatureID.Verticality,
        "eigentropy": EFeatureID.Eigentropy,
    }

    sel_feats = [feature_map.get(f_name) for f_name in feature_names]

    if None in sel_feats:
        warnings.warn(
            f"The following feature names are not recognized: {[f_name for f_name, f in zip(feature_names, sel_feats) if f is None]}. Available features: {list(feature_map.keys())}."
        )
        sel_feats = [f for f in sel_feats if f is not None]

    if len(sel_feats) == 0:
        raise ValueError(
            f"No valid feature names provided. Available features: {list(feature_map.keys())}. Provided feature names: {feature_names}"
        )

    # limit radius to 3 decimal places
    radius = np.round(radius, 3)
    # Compute geometric features
    feats = pgeof.compute_features_selected(
        point_cloud.coords,
        search_radius=radius,
        max_knn=max_knn,
        selected_features=sel_feats,
    )
    if len(feature_names) == 1:
        feats = [feats]
    for i, f_name in enumerate(feature_names):
        point_cloud.geometric_features[f"{f_name}_r{radius}"] = feats[i].reshape(-1)

####################################### Voxel-based computations #######################################

def compute_point_density_vapc(point_cloud, voxel_size: float, voxel_origin: np.ndarray) -> None:
    """
    Compute voxel-based point density using VAPC.

    :param point_cloud: PointCloud object
    :type point_cloud: PointCloud
    :param voxel_size: Size of the voxels
    :type voxel_size: float
    :param voxel_origin: Origin of the voxel grid (3D coordinates)
    :type voxel_origin: np.ndarray
    """
    import py4dgeo
    epoch = py4dgeo.Epoch(point_cloud.coords)
    vapc = py4dgeo.Vapc(epoch, voxel_size=voxel_size, origin=voxel_origin)
    voxel_density = vapc._compute_density()  # compute density using VAPC's internal method
    point_cloud.voxel_density = voxel_density
    point_cloud.voxel_size = voxel_size
    point_cloud.voxel_origin = voxel_origin
    point_cloud.voxel_coords = vapc.unique_voxels  # store voxel coordinates for potential future use
