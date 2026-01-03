import numpy as np
import pandas as pd


def compute_statistics(data: np.ndarray, location_stats=True, spread_stats=True, shape_stats=False) -> dict:
    """Compute summary statistics for a given data array"""
    mean = np.nanmean(data)
    stats = {}
    if location_stats:
        location_stats = {
            'mean': mean,
            'median': np.nanmedian(data),
            # only compute mode if data is type int
            'mode': int(pd.Series(data).mode()[0]) if data.dtype == int else None,
        }
        stats.update(location_stats)
    if spread_stats:
        std = np.nanstd(data)
        min = np.nanmin(data)
        max = np.nanmax(data)
        spread_stats = {
            'std dev': std,
            'variance': np.nanvar(data),
            'min': min,
            'max': max,
            'range': max - min,
            'iqr': np.nanpercentile(data, 75) - np.nanpercentile(data, 25),
            'cov': std / mean if mean != 0 else np.nan,
        }
        stats.update(spread_stats)
    if shape_stats:
        shape_stats = {
            'skewness': pd.Series(data).skew(),
            'kurtosis': pd.Series(data).kurtosis()
        }
        stats.update(shape_stats)
    return stats