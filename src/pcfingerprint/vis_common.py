"""
Shared visualization helpers for pcfingerprint renderers.

This module centralises common data-preparation and small utility
functions used by both `vis_matplotlib.py` and `vis_plotly.py` to
reduce duplication and keep renderer modules thin.
"""

from typing import List, Optional, Tuple
import math

import numpy as np
import plotly.express as px

import pcfingerprint
from pcfingerprint.utils import _validate_classes


def prepare_attribute_data(
    reference_pc,
    compare_pcs: List,
    attribute: Optional[str] = None,
    max_n_points: int = 50_000,
) -> Tuple[
    np.ndarray,
    List[np.ndarray],
    np.ndarray,
    List[np.ndarray],
    Optional[float],
    Optional[float],
    Optional[np.ndarray],
    Optional[List[str]],
]:
    """Prepare coordinates and attribute arrays for rendering.

    Returns:
      reference_coords, compare_coords_list, reference_data, compare_data,
      vmin, vmax, class_values, mapped_colors

    Notes:
      - If `attribute` is voxel-based (exists in `reference_pc.voxel_features`)
        the centroid points are returned as coords and attribute values are
        taken from the voxel_features arrays. Otherwise point subsampling is
        applied via `pcfingerprint.utils.subsample_points`.
    """
    # lazy import to avoid circular imports in some environments
    from pcfingerprint.utils import subsample_points

    # decide whether to use voxel centroids or raw points
    if attribute is not None and attribute in getattr(
        reference_pc, "voxel_features", {}
    ):
        reference_coords = reference_pc.voxel_features[attribute][1]
        compare_coords_list = [pc.voxel_features[attribute][1] for pc in compare_pcs]
        reference_data = pcfingerprint.pcfingerprint.get_attribute_data(
            reference_pc, attribute
        )
        compare_data = [
            pcfingerprint.pcfingerprint.get_attribute_data(compare_pc, attribute)
            for compare_pc in compare_pcs
        ]
    else:
        reference_pc_sub, _ = subsample_points(reference_pc, max_n_points)
        compare_sub = [subsample_points(pc, max_n_points)[0] for pc in compare_pcs]
        reference_coords = reference_pc_sub.coords
        compare_coords_list = [pc.coords for pc in compare_sub]

        # obtain attribute arrays from the same subsampled clouds used for plotting
        reference_data = pcfingerprint.pcfingerprint.get_attribute_data(
            reference_pc_sub, attribute
        )
        compare_data = [
            pcfingerprint.pcfingerprint.get_attribute_data(compare_pc, attribute)
            for compare_pc in compare_sub
        ]

    # classification handling
    class_values = None
    mapped_colors = None
    if attribute == "classification":
        class_values = np.unique(
            np.concatenate(
                [
                    np.asarray(reference_data).ravel(),
                    *[np.asarray(d).ravel() for d in compare_data],
                ]
            )
        )
        colors = px.colors.qualitative.Set1
        mapped_colors = [colors[i % len(colors)] for i in range(len(class_values))]

    # compute vmin/vmax for numeric attributes
    vmin = None
    vmax = None
    if attribute != "classification":
        try:
            arrs = [np.asarray(reference_data)] + [np.asarray(d) for d in compare_data]
            finite = np.concatenate([a[np.isfinite(a)] for a in arrs if a.size])
            if finite.size:
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
        except Exception:
            vmin = None
            vmax = None

    return (
        reference_coords,
        compare_coords_list,
        reference_data,
        compare_data,
        vmin,
        vmax,
        class_values,
        mapped_colors,
    )


def prepare_histogram_data_per_class(
    series: List[Tuple],
    reference: Optional[Tuple] = None,
):
    """Split each ``(pc, attribute)`` series by class value found on its own point cloud.

    :param series: List of ``(PointCloud, attribute)`` pairs to split by class.
    :type series: list[tuple[PointCloud, str]]
    :param reference: Optional ``(PointCloud, attribute)`` pair for the reference series.
    :type reference: tuple[PointCloud, str] | None

    :return: ``(classes, reference_per_class, compare_per_class)`` where
        ``reference_per_class`` is a list of ``(pc, attribute)`` pairs (or ``None``
        entries if ``reference`` was ``None``) and ``compare_per_class`` is a list of
        lists of ``(pc, attribute)`` pairs, both indexed in parallel with ``classes``.
        Each ``pc`` is masked by its own ``classification`` attribute.
    """
    all_pcs = [pc for pc, _ in series] + (
        [reference[0]] if reference is not None else []
    )
    for pc in all_pcs:
        _validate_classes(pc)

    classes = np.unique(
        np.concatenate([np.unique(pc.classification) for pc in all_pcs])
    )

    reference_per_class = []
    compare_per_class = []
    for class_value in classes:
        if reference is not None:
            ref_pc, ref_attr = reference
            reference_per_class.append(
                (ref_pc.mask(ref_pc.classification == class_value), ref_attr)
            )
        else:
            reference_per_class.append(None)
        compare_per_class.append(
            [(pc.mask(pc.classification == class_value), attr) for pc, attr in series]
        )

    return classes, reference_per_class, compare_per_class


def plot_grid_layout(
    num_plots: int, all_comparison: bool = False
) -> Tuple[int, int, List[str]]:
    """Return nrows, ncols and subplot titles for a grid with `num_plots` slots.

    Uses the same simple heuristic as the original modules: 2 columns up to
    four plots, otherwise 3 columns.
    """
    ncols = 2 if num_plots <= 4 else 3
    nrows = math.ceil(num_plots / ncols)
    subplot_titles = [""] * (nrows * ncols)
    if all_comparison:
        for i in range(num_plots):
            subplot_titles[i] = f"Comparison Point Cloud {i + 1}"
    else:
        if num_plots > 0:
            subplot_titles[0] = "Reference Point Cloud"
        for i in range(num_plots - 1):
            subplot_titles[i + 1] = f"Comparison Point Cloud {i + 1}"

    return nrows, ncols, subplot_titles


def prepare_distance_scatter_ref_data(
    pc,
    attribute_list: List[str],
    max_n_points: int = 50_000,
):
    """Prepare reference point cloud data for distance attribute scatter plots."""
    from pcfingerprint.utils import subsample_points

    pc_sub, _ = subsample_points(pc, max_n_points)
    pc_pl = pc_sub.coords
    values = [np.asarray(pc_sub.c2c[attribute]) for attribute in attribute_list]

    finite_values = np.concatenate(
        [val[np.isfinite(val)] for val in values if val.size]
    )
    if finite_values.size:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
    else:
        vmin = None
        vmax = None

    return pc_pl, values, vmin, vmax


def prepare_distance_scatter_comp_data(
    pc_list: List,
    attribute: str,
    max_n_points: int = 50_000,
):
    """Prepare comparison point cloud data for distance attribute scatter plots."""
    from pcfingerprint.utils import subsample_points

    pc_list_sub = [subsample_points(pc, max_n_points)[0] for pc in pc_list]
    coords_list = [pc.coords for pc in pc_list_sub]
    values = [np.asarray(pc.c2c[attribute]) for pc in pc_list_sub]

    finite_values = np.concatenate(
        [val[np.isfinite(val)] for val in values if val.size]
    )
    if finite_values.size:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
    else:
        vmin = None
        vmax = None

    return coords_list, values, vmin, vmax


def histogram_bin_edges(values_list: List, bins: int) -> np.ndarray:
    """Return shared histogram bin edges for multiple data series."""
    arrays = [np.asarray(values) for values in values_list if values is not None]
    if not arrays:
        return np.linspace(0.0, 1.0, bins + 1)

    combined = np.concatenate([array[np.isfinite(array)] for array in arrays])
    if combined.size == 0:
        return np.linspace(0.0, 1.0, bins + 1)

    min_value = float(np.min(combined))
    max_value = float(np.max(combined))
    if min_value == max_value:
        padding = 0.5 if min_value == 0 else max(abs(min_value) * 0.05, 0.5)
        min_value -= padding
        max_value += padding
    elif np.issubdtype(combined.dtype, np.integer):
        min_value -= 0.5
        max_value += 0.5

    return np.linspace(min_value, max_value, bins + 1)


def classification_color_mapping(values_list: List) -> Tuple[np.ndarray, List[str]]:
    """Return class values and a list of mapped hex colors for discrete classes."""
    class_values = np.unique(
        np.concatenate([np.asarray(values).ravel() for values in values_list])
    )
    colors = px.colors.qualitative.Set1
    mapped_colors = [colors[i % len(colors)] for i in range(len(class_values))]
    return class_values, mapped_colors
