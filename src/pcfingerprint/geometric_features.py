"""
Module for calculating geometric features on point clouds.
"""

import numpy as np
import warnings
from pcfingerprint.utils import point_cloud_to_vapc


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


def compute_point_density_vapc(
    point_cloud, voxel_size: float, voxel_origin: list[float] | None
):
    """
    Compute voxel-based point density using VAPC.

    :param point_cloud: PointCloud object
    :type point_cloud: PointCloud
    :param voxel_size: Size of the voxels
    :type voxel_size: float
    :param voxel_origin: Origin of the voxel grid (3D coordinates). If None, [0.0, 0.0, 0.0] is used.
    :type voxel_origin: list[float] or None

    :return: VAPC instance
    :rtype: py4dgeo.Vapc
    """
    import py4dgeo

    py4dgeo.enable_timeit(False)
    py4dgeo.enable_trace(False)
    resolved_voxel_origin = (
        [0.0, 0.0, 0.0] if voxel_origin is None else list(voxel_origin)
    )
    vapc = point_cloud_to_vapc(
        point_cloud,
        voxel_size=voxel_size,
        voxel_origin=resolved_voxel_origin,
        include_additional_dimensions=False,
    )
    voxel_density = (
        vapc._compute_density()
    )  # compute density using VAPC's internal method
    # store computed voxel density under geometric_features
    point_cloud.geometric_features[f"voxel_density_v{voxel_size}"] = voxel_density

    return vapc


def compute_geom_feat_vapc(
    point_cloud,
    voxel_size: float,
    voxel_origin: list[float] | None,
    feature_names: list[str],
):
    """
    Compute voxel-based geometric features using VAPC.

    :param point_cloud: PointCloud object
    :type point_cloud: PointCloud
    :param voxel_size: Size of the voxels
    :type voxel_size: float
    :param voxel_origin: Origin of the voxel grid (3D coordinates). If None, [0.0, 0.0, 0.0] is used.
    :type voxel_origin: list[float] or None
    :param feature_names: List of feature names to compute. Supported features depend on the underlying VAPC implementation.
    :type feature_names: list[str]

    :return: VAPC instance with computed features
    :rtype: py4dgeo.Vapc

    Available VAPC features (py4dgeo v1.1):
    "count", "density", "centroid", "voxel_center", "local_centroid", "covariance",
    "closest_to_voxel_centers", "closest_to_centroids", "eigenvalues", "linearity",
    "planarity", "sphericity", "sum_of_eigenvalues", "omnivariance", "eigenentropy",
    "anisotropy", "surface_variation".
    """
    import py4dgeo

    py4dgeo.enable_timeit(False)
    py4dgeo.enable_trace(False)
    resolved_voxel_origin = (
        [0.0, 0.0, 0.0] if voxel_origin is None else list(voxel_origin)
    )

    vapc_available_features = [
        "count",
        "density",
        "centroid",
        "voxel_center",
        "local_centroid",
        "covariance",
        "closest_to_voxel_centers",
        "closest_to_centroids",
        "eigenvalues",
        "linearity",
        "planarity",
        "sphericity",
        "sum_of_eigenvalues",
        "omnivariance",
        "eigenentropy",
        "anisotropy",
        "surface_variation",
    ]

    if len(feature_names) == 0:
        feature_names = vapc_available_features

    sel_feats = [f for f in feature_names if f in vapc_available_features]
    invalid_features = [f for f in feature_names if f not in vapc_available_features]
    if invalid_features:
        warnings.warn(
            f"Unsupported VAPC features ignored: {invalid_features}. "
            f"Available features: {vapc_available_features}",
            UserWarning,
        )
    if len(sel_feats) == 0:
        raise ValueError(
            "No valid VAPC features selected. "
            f"Available features: {vapc_available_features}"
        )

    vapc = point_cloud_to_vapc(
        point_cloud,
        voxel_size=voxel_size,
        voxel_origin=resolved_voxel_origin,
        include_additional_dimensions=True,
    )
    vapc.compute_features(feature_names=sel_feats)  # populates vapc.out

    # Store computed VAPC outputs under geometric_features
    # TODO: Either
    # a) map voxel-based features to all points in point cloud and then store as point-wise geometric features
    # OR b) store voxel-based features separately in the PointCloud object (e.g. in a new attribute voxel_features) instead of mixing with point-wise geometric features
    # background: avoid shape mismatches, which cause errors in, e.g., subsampling
    for feat_name, feat_data in vapc.out.items():
        point_cloud.geometric_features[f"{feat_name}_v{voxel_size}"] = feat_data

    return vapc
