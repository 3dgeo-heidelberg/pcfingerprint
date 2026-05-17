from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger("pcfingerprint")


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


def match_representative_points_to_indices(point_coords, representative_coords):
    """Return original point indices for representative voxel points.

    The representative coordinates produced by VAPC are copied from the
    original point cloud, so an exact coordinate match is the preferred path.
    If a coordinate appears multiple times, the first unused match is chosen.
    """
    point_coords = np.ascontiguousarray(np.asarray(point_coords))
    representative_coords = np.ascontiguousarray(np.asarray(representative_coords))
    if representative_coords.size == 0:
        return np.array([], dtype=int)
    row_dtype = np.dtype((np.void, point_coords.dtype.itemsize * point_coords.shape[1]))
    point_row_keys = point_coords.view(row_dtype).ravel()
    representative_row_keys = representative_coords.view(row_dtype).ravel()
    sort_order = np.argsort(point_row_keys, kind="mergesort")
    sorted_point_row_keys = point_row_keys[sort_order]
    representative_indices = np.empty(representative_coords.shape[0], dtype=int)
    used_exact_matches = {}
    for rep_index, (coord, row_key) in enumerate(
        zip(representative_coords, representative_row_keys)
    ):
        left = np.searchsorted(sorted_point_row_keys, row_key, side="left")
        right = np.searchsorted(sorted_point_row_keys, row_key, side="right")
        if right > left:
            key_bytes = bytes(row_key)
            used_count = used_exact_matches.get(key_bytes, 0)
            if left + used_count < right:
                representative_indices[rep_index] = sort_order[left + used_count]
                used_exact_matches[key_bytes] = used_count + 1
                continue

        close_matches = np.where(np.all(np.isclose(point_coords, coord), axis=1))[0]
        if close_matches.size == 0:
            raise ValueError(
                "Could not map representative voxel coordinates back to point indices."
            )
        representative_indices[rep_index] = close_matches[0]

    return representative_indices


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


def subsample_points_random(pc, max_n_points: int = 500_000):
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
        logger.info(
            f"Subsampled {Path(pc.source).name} from {num_points} to {pc.coords.shape[0]} points for visualization."
        )
    else:
        indices = np.arange(num_points)
    return pc, indices


def subsample_points_voxel(
    pc, voxel_size: float, mode: str = "closest_to_centroids"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Voxel-based subsamplint using py4dgeo's VAPC.

    :param pc: The input PointCloud object to be subsampled.
    :type pc: PointCloud
    :param voxel_size: Size of the voxels.
    :type voxel_size: float
    :param mode: Mode for reducing points within each voxel. Options are "closest_to_centroids" or "closest_to_voxel_centers".
    :type mode: str

    :return: The subsampled point cloud and the indices of the points that were kept.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    vapc = point_cloud_to_vapc(
        pc, voxel_size=voxel_size, include_additional_dimensions=False
    )
    reduced_vapc = vapc.reduce_to_feature(mode)
    indices = reduced_vapc.original_point_cloud_indices
    pc = pc.mask(indices)

    return pc, indices


def _estimate_voxel_size_for_target_voxel_count(pc, target_voxel_count: int) -> float:
    """
    Based on the bounding box of the point cloud and the target number of voxels, estimate a voxel size that would yield approximately the target number of voxels when subsampling with VAPC.

    :param pc: The input PointCloud object for which to estimate the voxel size.
    :type pc: PointCloud
    :param target_voxel_count: The target number of voxels to aim for when subsampling.
    :type target_voxel_count: int

    :return: The estimated voxel size to achieve the target number of voxels.
    :rtype: float
    """
    min_coords = np.min(pc.coords, axis=0)
    max_coords = np.max(pc.coords, axis=0)
    bounding_box_size = max_coords - min_coords
    voxel_size = (bounding_box_size.prod() / target_voxel_count) ** (1 / 3)

    # round to a reasonable number of decimal places for typical point cloud coordinate scales
    voxel_size = round(voxel_size, 4)
    return voxel_size


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
