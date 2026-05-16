import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("pcfingerprint")


def compute_statistics(
    data: np.ndarray, location_stats=True, spread_stats=True, shape_stats=False
) -> dict:
    """Compute summary statistics for a given data array

    :param data: Input data array
    :type data: np.ndarray
    :param location_stats: Whether to compute location statistics (mean, median, mode)
    :type location_stats: bool
    :param spread_stats: Whether to compute spread statistics (std dev, variance, min, max, range, iqr, cov)
    :type spread_stats: bool
    :param shape_stats: Whether to compute shape statistics (skewness, kurtosis)
    :type shape_stats: bool

    :return: Dictionary of computed statistics
    :rtype: dict
    """
    mean = np.nanmean(data)
    stats = {}
    if location_stats:
        location_stats = {
            "mean": mean,
            "median": np.nanmedian(data),
            # only compute mode if data is type int
            "mode": int(pd.Series(data).mode()[0]) if data.dtype == int else None,
        }
        stats.update(location_stats)
    if spread_stats:
        std = np.nanstd(data)
        min = np.nanmin(data)
        max = np.nanmax(data)
        spread_stats = {
            "std dev": std,
            "variance": np.nanvar(data),
            "min": min,
            "max": max,
            "range": max - min,
            "iqr": np.nanpercentile(data, 75) - np.nanpercentile(data, 25),
            "cov": std / mean if mean != 0 else np.nan,
        }
        stats.update(spread_stats)
    if shape_stats:
        shape_stats = {
            "skewness": pd.Series(data).skew(),
            "kurtosis": pd.Series(data).kurtosis(),
        }
        stats.update(shape_stats)
    return stats


def compute_return_number_metrics(return_number, number_of_returns) -> dict:
    """
    Function to compute
    - fraction of first returns
    - fraction of intermediate returns
    - fraction of multiple returns

    :param return_number: Array of return numbers
    :type return_number: sequence or np.ndarray (or laspy SubFieldView)
    :param number_of_returns: Array of number of returns
    :type number_of_returns: sequence or np.ndarray (or laspy SubFieldView)

    :return: Dictionary of return number metrics
    :rtype: dict
    """
    assert (
        return_number.shape == number_of_returns.shape
    ), "return_number and number_of_returns must have the same shape"
    # convert to numpy arrays if not already
    return_number = np.array(return_number)
    number_of_returns = np.array(number_of_returns)
    total_points = int(return_number.shape[0])
    first_returns = int(np.sum(return_number == 1))
    last_returns = int(np.sum(return_number == number_of_returns))
    intermediate_returns = int(
        np.sum((return_number > 1) & (return_number < number_of_returns))
    )
    multiple_returns = int(np.sum(number_of_returns > 1))
    single_returns = int(np.sum(number_of_returns == 1))
    metrics = {
        "number of returns": total_points,
        "number of first returns": first_returns,
        "number of intermediate returns": intermediate_returns,
        "number of last returns": last_returns,
        "number of single returns": single_returns,
        "number of multiple returns": multiple_returns,
        "fraction of first returns": first_returns / total_points,
        "fraction of intermediate returns": intermediate_returns / total_points,
        "fraction of last returns": last_returns / total_points,
        "fraction of single returns": single_returns / total_points,
        "fraction of multiple returns": multiple_returns / total_points,
    }

    return metrics


def compute_class_distribution_metrics(
    class_labels, class_mapping: dict[int, str] | None = None
) -> dict:
    """
    Function to compute simple class distribution metrics, including:
    - number of points in each class
    - fraction of points in each class

    :param class_labels: Array of class labels
    :type class_labels: sequence or np.ndarray (or laspy SubFieldView)
    :param class_mapping: Optional dict mapping class values to class names
    :type class_mapping: dict[int, str] | None

    :return: Dictionary of class distribution metrics
    :rtype: dict
    """
    class_labels = np.array(class_labels)
    total_points = int(class_labels.shape[0])
    unique_classes, counts = np.unique(class_labels, return_counts=True)
    metrics = {"total points": total_points}
    for cls, count in zip(unique_classes, counts):
        class_name = (
            class_mapping.get(cls, f"Class {cls}") if class_mapping else f"Class {cls}"
        )
        metrics[f"number of points in class {class_name}"] = count
        metrics[f"fraction of points in class {class_name}"] = count / total_points
    return metrics
