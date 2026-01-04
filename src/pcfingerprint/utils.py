import numpy as np


def shift_to_zero(array):
    return array - np.min(array)

def standardize(array):
    mean = np.mean(array)
    std = np.std(array)
    return (array - mean) / std

def scale_robust(array):
    median = np.median(array)
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    iqr = q3 - q1
    return (array - median) / iqr