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
