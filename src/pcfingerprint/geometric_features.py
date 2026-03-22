"""
Module for calculating geometric features on point clouds.
"""
import numpy as np


def compute_point_density_jakteristics(point_cloud, radius: float, nthreads: int = -1) -> None:
    """
    Compute (volume) point density using Jakteristics.

    :param point_cloud: PointCloud object
    :type point_cloud: PointCloud
    :param radius: Radius for the neighborhood search
    :type radius: float
    :param nthreads: Number of threads to use
    :type nthreads: int
    """
    import jakteristics

    # limit radius to 3 decimal places
    radius = np.round(radius, 3)
    if not f"number_of_neighbors_r{radius}" in point_cloud.geometric_features:
        feats = jakteristics.compute_features(
        point_cloud.coords,
        search_radius=radius,
        feature_names=["number_of_neighbors"],
        num_threads=nthreads
        )
        point_cloud.geometric_features[f"number_of_neighbors_r{radius}"] = feats.reshape(-1)
    point_cloud.point_density = point_cloud.geometric_features[f"number_of_neighbors_r{radius}"] / (4/3 * np.pi * radius**3)
    point_cloud.point_density_radius = radius


def compute_point_density_pgeof(point_cloud, radius: float, max_knn: int = 1000) -> None:
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
    if not f"number_of_neighbors_r{radius}" in point_cloud.geometric_features:
        indices, sqrd_dists = pgeof.radius_search(
            point_cloud.coords.astype(np.float32),
            point_cloud.coords.astype(np.float32),
            search_radius=radius,
            max_knn=max_knn
        )
        # number of neighbors is the number of valid indices returned for each point - invalid indices have value -1
        num_neigh = np.array([np.sum(indices[i] != -1) for i in range(len(point_cloud.coords))])
        point_cloud.geometric_features[f"number_of_neighbors_r{radius}"] = num_neigh
    point_cloud.point_density = point_cloud.geometric_features[f"number_of_neighbors_r{radius}"] / (4/3 * np.pi * radius**3)
    point_cloud.point_density_radius = radius


def compute_geom_feat_jakteristics(point_cloud, feature_names: list[str], radius: float, nthreads: int = -1) -> None:
    """
    Compute geometric features using Jakteristics.
    
    :param point_cloud: PointCloud object
    :type point_cloud: PointCloud
    :param feature_names: List of feature names to compute
    :type feature_names: list[str]
    :param radius: Radius for the neighborhood search
    :type radius: float
    :param nthreads: Number of threads to use
    :type nthreads: int
    """
    import jakteristics

    # limit radius to 3 decimal places
    radius = np.round(radius, 3)
    # Compute geometric features
    feats = jakteristics.compute_features(
        point_cloud.coords,
        search_radius=radius,
        feature_names=feature_names,
        num_threads=nthreads
    )
    if len(feature_names) == 1:
        feats = [feats]
    for i, f_name in enumerate(feature_names):
        point_cloud.geometric_features[f"{f_name}_r{radius}"] = feats[i].reshape(-1)


def compute_geom_feat_pgeof(point_cloud, feature_names: list[str], radius: float, max_knn: int = 1000) -> None:
    """
    Compute geometric features using PGEOF.
    
    :param point_cloud: PointCloud object
    :type point_cloud: PointCloud
    :param feature_names: List of feature names to compute
    :type feature_names: list[str]
        Available features: "Linearity", "Planarity", "Scattering", "VerticalityPGEOF", "NormalX", "NormalY", "NormalZ", "Verticality", "Eigentropy"
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
        "eigentropy": EFeatureID.Eigentropy
    }

    # limit radius to 3 decimal places
    radius = np.round(radius, 3)
    # Compute geometric features
    feats = pgeof.compute_features_selected(
        point_cloud.coords,
        search_radius=radius,
        max_knn=max_knn,
        selected_features=[feature_map.get(f_name) for f_name in feature_names],
    )
    if len(feature_names) == 1:
        feats = [feats]
    for i, f_name in enumerate(feature_names):
        point_cloud.geometric_features[f"{f_name}_r{radius}"] = feats[i].reshape(-1)