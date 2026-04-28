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
