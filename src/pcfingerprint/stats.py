import numpy as np
import pandas as pd


def compute_statistics(data: np.ndarray) -> dict:
    """Compute summary statistics for a given data array"""

    stats = {
        'mean': np.nanmean(data),
        'median': np.nanmedian(data),
        'std_dev': np.nanstd(data)
    }
    return stats