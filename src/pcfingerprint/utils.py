import numpy as np


def shift_to_zero(array):
    """
    Shift the array to have a minimum value of zero.
    :param array: The input array to be shifted.
    :type array: np.ndarray

    :return: The shifted array with a minimum value of zero.
    :rtype: np.ndarray
    """
    return array - np.min(array)


def standardize(array):
    """
    Standardize the array to have a mean of zero and a standard deviation of one.
    :param array: The input array to be standardized.

    :return: The standardized array.
    :rtype: np.ndarray
    """
    mean = np.mean(array)
    std = np.std(array)
    return (array - mean) / std


def scale_robust(array):
    """
    Scale the array using the interquartile range (IQR) to reduce the influence of outliers (robust scaling).
    :param array: The input array to be scaled.

    :return: The robustly scaled array.
    :rtype: np.ndarray
    """
    median = np.median(array)
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    iqr = q3 - q1
    return (array - median) / iqr


def _build_structured_additional_dimensions(point_cloud):
    """Build py4dgeo-compatible structured per-point dimensions or return None."""
    extra_dim_names = (
        "gps_time",
        "return_number",
        "number_of_returns",
        "intensity",
        "classification",
    )
    n_points = point_cloud.coords.shape[0]
    dtype_fields = []
    values_by_name = {}
    for dim in extra_dim_names:
        values = getattr(point_cloud, dim, None)
        if values is None:
            continue
        values = np.asarray(values).reshape(-1)
        if values.shape[0] != n_points:
            continue
        dtype_fields.append((dim, values.dtype))
        values_by_name[dim] = values

    if not dtype_fields:
        return None

    additional_dimensions = np.empty(n_points, dtype=dtype_fields)
    for dim, _ in dtype_fields:
        additional_dimensions[dim] = values_by_name[dim]
    return additional_dimensions


def point_cloud_to_vapc(
    point_cloud,
    voxel_size: float,
    voxel_origin: list[float] | None,
    include_additional_dimensions: bool = True,
):
    """
    Convert a PointCloud to a configured py4dgeo Vapc object.

    :param point_cloud: PointCloud object with coords and optional per-point attributes.
    :param voxel_size: Size of the voxels.
    :param voxel_origin: Origin of the voxel grid (3D coordinates). If None, [0.0, 0.0, 0.0] is used.
    :param include_additional_dimensions: Include structured per-point dimensions if available.
    :return: py4dgeo.Vapc instance.
    """
    import py4dgeo

    additional_dimensions = None
    if include_additional_dimensions:
        additional_dimensions = _build_structured_additional_dimensions(point_cloud)
    epoch = py4dgeo.Epoch(
        point_cloud.coords, additional_dimensions=additional_dimensions
    )
    if voxel_origin is None:
        voxel_origin = [0.0, 0.0, 0.0]
    return py4dgeo.Vapc(epoch, voxel_size=voxel_size, origin=voxel_origin)


def subsample_points(pc, max_n_points: int = 500_000):
    """
    Randomly subsample the point cloud to a maximum number of points.
    If the point cloud has less points than the specified maximum, it will be returned unchanged.

    :param pc: The input point cloud to be subsampled.
    :type pc: PointCloud
    :param max_n_points: The maximum number of points to subsample to.
    :type max_n_points: int

    :return: The subsampled point cloud and the indices of the points that were kept.
    :rtype: tuple(PointCloud, np.ndarray)
    """
    num_points = pc.coords.shape[0]
    if num_points > max_n_points:
        indices = np.random.choice(num_points, max_n_points, replace=False)
        pc = pc.mask(indices)
        print(
            f"Subsampled from {num_points} to {pc.coords.shape[0]} points for visualization."
        )
    else:
        indices = np.arange(num_points)
    return pc, indices


def _validate_classes(pc):
    try:
        getattr(pc, "classification")
    except AttributeError:
        raise ValueError(
            f"Point cloud does not have 'classification' attribute, cannot plot histogram per class."
        )
    if pc.classification is None or len(pc.classification) == 0:
        raise ValueError(
            f"Point cloud has empty 'classification' attribute, cannot plot histogram per class."
        )
