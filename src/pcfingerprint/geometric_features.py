"""
Module for calculating geometric features on point clouds.
"""
import numpy as np
import jakteristics


def compute_point_density(point_cloud, radius: float, nthreads: int = -1) -> None:
    """
    Compute (volume) point density using Jakteristics.

    :param point_cloud: PointCloud object
    :type point_cloud: PointCloud
    :param radius: Radius for the neighborhood search
    :type radius: float
    :param nthreads: Number of threads to use
    :type nthreads: int
    """
    # limit radius to 3 decimal places
    radius = np.round(radius, 3)
    if not "number_of_neighbors" in point_cloud.geometric_features:
        feats = jakteristics.compute_features(
        point_cloud.coords,
        search_radius=radius,
        feature_names=["number_of_neighbors"],
        num_threads=nthreads
        )
        point_cloud.geometric_features["number_of_neighbors"] = feats.reshape(-1)
    point_cloud.point_density = point_cloud.geometric_features["number_of_neighbors"] / (4/3 * np.pi * radius**3)
    point_cloud.point_density_radius = radius



def compute_geom_feat(point_cloud, feature_names: list[str], radius: float, nthreads: int = -1) -> None:
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
