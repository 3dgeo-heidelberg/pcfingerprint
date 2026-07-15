"""
Module for visualisations on comparisons between point clouds.
"""

import math

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Template
import pcfingerprint
from pcfingerprint.utils import subsample_points, _validate_classes
from pcfingerprint.vis_common import (
    prepare_attribute_data,
    plot_grid_layout,
    prepare_distance_scatter_comp_data,
    prepare_distance_scatter_ref_data,
    histogram_bin_edges,
    classification_color_mapping,
    prepare_histogram_data_per_class,
)

import logging

logger = logging.getLogger("pcfingerprint")


def _linked_3d_camera_post_script() -> str:
    """Return JavaScript that syncs camera updates across all 3D scenes."""
    return """
const gd = document.getElementById('{plot_id}');
if (!gd) {
    return;
}

let syncing = false;

function getSceneKeys(layout) {
    return Object.keys(layout || {}).filter((key) => /^scene\\d*$/.test(key));
}

gd.on('plotly_relayout', (eventData) => {
    if (syncing || !eventData) {
        return;
    }

    let sourceScene = null;
    let camera = null;

    for (const [key, value] of Object.entries(eventData)) {
        const match = key.match(/^(scene\\d*)\\.camera$/);
        if (match) {
            sourceScene = match[1];
            camera = value;
            break;
        }
    }

    if (!camera) {
        for (const key of Object.keys(eventData)) {
            const match = key.match(/^(scene\\d*)\\.camera\\./);
            if (match) {
                sourceScene = match[1];
                break;
            }
        }
        if (sourceScene && gd.layout[sourceScene] && gd.layout[sourceScene].camera) {
            camera = gd.layout[sourceScene].camera;
        }
    }

    if (!camera) {
        return;
    }

    const updates = {};
    for (const scene of getSceneKeys(gd.layout)) {
        if (scene !== sourceScene) {
            updates[`${scene}.camera`] = camera;
        }
    }

    if (Object.keys(updates).length === 0) {
        return;
    }

    syncing = true;
    Plotly.relayout(gd, updates)
        .finally(() => {
            syncing = false;
        });
});
"""


def write_html_with_linked_3d_camera(fig: go.Figure, file, **kwargs):
    """Write a Plotly figure to standalone HTML with linked 3D camera behavior."""
    existing_post_script = kwargs.pop("post_script", None)
    linked_script = _linked_3d_camera_post_script()

    kwargs.setdefault("config", {"responsive": True})

    if existing_post_script is None:
        post_script = linked_script
    elif isinstance(existing_post_script, str):
        post_script = [existing_post_script, linked_script]
    else:
        post_script = [*existing_post_script, linked_script]

    fig.write_html(file, post_script=post_script, **kwargs)


def _histogram_bar_trace(
    values,
    bin_edges: np.ndarray,
    name: str,
    marker_color: str,
    opacity: float,
    histnorm: str = None,
    showlegend: bool = True,
) -> go.Bar:
    """Create a lightweight bar trace that mimics a Plotly histogram."""
    array = np.asarray(values)
    array = array[np.isfinite(array)]

    if array.size == 0:
        counts = np.zeros(len(bin_edges) - 1)
    else:
        counts, _ = np.histogram(array, bins=bin_edges)

    counts = counts.astype(float)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2

    if histnorm == "probability density":
        total = counts.sum()
        if total > 0:
            counts = counts / (total * bin_widths)

    return go.Bar(
        x=bin_centers,
        y=counts,
        width=bin_widths,
        name=name,
        marker_color=marker_color,
        opacity=opacity,
        showlegend=showlegend,
        marker_line_width=0,
    )


def plot_3d_scatter(
    reference_pc, compare_pcs: list, attribute: str = None, max_n_points: int = 50_000
):

    if attribute is None:
        logger.info(
            "No attribute specified for coloring. Plotting 3D scatter plot colored by z coordinate instead."
        )
        attribute = "z"
    logger.info(f"Generating 3D scatter plot colored by {attribute}...")

    # prepare data and coords using the shared helper
    (
        reference_pc_pl,
        compare_pc_pl,
        reference_data,
        compare_data,
        vmin,
        vmax,
        class_values,
        mapped_colors,
    ) = prepare_attribute_data(
        reference_pc, compare_pcs, attribute=attribute, max_n_points=max_n_points
    )
    # determine plot grid based on inputs
    total_plots = len(compare_pcs) + 1
    nrows, ncols, subplot_titles = plot_grid_layout(total_plots)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        specs=[[{"type": "scatter3d"} for _ in range(ncols)] for _ in range(nrows)],
    )

    # compute global vmin/vmax for a consistent color scale across subplots
    if attribute == "classification":
        # build colorscale for plotly from mapped colors
        if class_values is None:
            class_values, mapped_colors = classification_color_mapping(
                [reference_data, *compare_data]
            )
        reference_data = np.searchsorted(class_values, np.asarray(reference_data))
        compare_data = [
            np.searchsorted(class_values, np.asarray(data)) for data in compare_data
        ]
        vmin = 0
        vmax = max(len(class_values) - 1, 0)
        colorscale = []
        n_colors = len(mapped_colors)
        for i, color in enumerate(mapped_colors):
            start = i / n_colors
            end = (i + 1) / n_colors
            colorscale.extend([[start, color], [end, color]])
        colorbar = dict(
            title=attribute,
            tickmode="array",
            tickvals=list(range(len(class_values))),
            ticktext=[str(value) for value in class_values],
        )
    else:
        # ensure vmin/vmax set by helper are used; otherwise attempt to infer
        try:
            if vmin is None or vmax is None:
                vmin = float(np.min(reference_data))
                vmax = float(np.max(reference_data))
                for data in compare_data:
                    vmin = min(vmin, float(np.min(data)))
                    vmax = max(vmax, float(np.max(data)))
        except Exception:
            vmin = None
            vmax = None
        colorscale = "Viridis"
        colorbar = dict(title=attribute)

    # reference world
    fig.add_trace(
        go.Scatter3d(
            x=reference_pc_pl[:, 0],
            y=reference_pc_pl[:, 1],
            z=reference_pc_pl[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=reference_data,
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                opacity=0.8,
                colorbar=colorbar,
            ),
            name="Reference Point Cloud",
        ),
        row=1,
        col=1,
    )

    # comparison
    for i, compare_pc in enumerate(compare_pcs):
        subplot_index = i + 1
        row = subplot_index // ncols + 1
        col = subplot_index % ncols + 1
        fig.add_trace(
            go.Scatter3d(
                x=compare_pc_pl[i][:, 0],
                y=compare_pc_pl[i][:, 1],
                z=compare_pc_pl[i][:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=compare_data[i],
                    colorscale=colorscale,
                    cmin=vmin,
                    cmax=vmax,
                    opacity=0.8,
                ),
                name=f"Comparison Point Cloud {i + 1}",
            ),
            row=row,
            col=col,
        )
    fig.update_scenes(aspectmode="data")
    fig.update_layout(template="plotly_white", showlegend=False)

    return fig


def plot_3d_scatter_dist_ref(
    pc, attribute_list: list, max_n_points: int = 50_000, vmax: float = None
):
    pc_pl, values, vmin, inferred_vmax = prepare_distance_scatter_ref_data(
        pc, attribute_list, max_n_points=max_n_points
    )
    if vmax is not None:
        inferred_vmax = vmax

    nrows, ncols, subplot_titles = plot_grid_layout(
        len(attribute_list), all_comparison=True
    )
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        specs=[[{"type": "scatter3d"} for _ in range(ncols)] for _ in range(nrows)],
    )

    colorscale = "Viridis"
    colorbar = dict(title="Distance")

    for i in range(len(attribute_list)):
        row = i // ncols + 1
        col = i % ncols + 1
        fig.add_trace(
            go.Scatter3d(
                x=pc_pl[:, 0],
                y=pc_pl[:, 1],
                z=pc_pl[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=values[i],
                    colorscale=colorscale,
                    cmin=vmin,
                    cmax=inferred_vmax,
                    opacity=0.8,
                    colorbar=colorbar,
                ),
                name="Reference Point Cloud",
            ),
            row=row,
            col=col,
        )

    fig.update_scenes(aspectmode="data")
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_3d_scatter_dist_comp(
    pc_list: list, attribute: str, max_n_points: int = 50_000, vmax: float = None
):
    pc_pl_list, values, vmin, inferred_vmax = prepare_distance_scatter_comp_data(
        pc_list, attribute, max_n_points=max_n_points
    )
    if vmax is not None:
        inferred_vmax = vmax

    nrows, ncols, subplot_titles = plot_grid_layout(len(pc_list), all_comparison=True)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        specs=[[{"type": "scatter3d"} for _ in range(ncols)] for _ in range(nrows)],
    )

    colorscale = "Viridis"
    colorbar = dict(title="Distance")

    for i in range(len(pc_pl_list)):
        row = i // ncols + 1
        col = i % ncols + 1
        fig.add_trace(
            go.Scatter3d(
                x=pc_pl_list[i][:, 0],
                y=pc_pl_list[i][:, 1],
                z=pc_pl_list[i][:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=values[i],
                    colorscale=colorscale,
                    cmin=vmin,
                    cmax=inferred_vmax,
                    opacity=0.8,
                    colorbar=colorbar,
                ),
                name=f"Comparison Point Cloud {i + 1}",
            ),
            row=row,
            col=col,
        )

    fig.update_scenes(aspectmode="data")
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_histogram(
    compare_pcs: list,
    attribute: str | list[str],
    reference_pc=None,
    bins: int = 100,
    overlay: bool = False,
    normalize: bool = False,
):
    """
    Plot histograms of an attribute for reference and comparison point clouds side by side.

    :param compare_pcs: List of PointCloud objects representing the comparison point clouds
    :type compare_pcs: list
    :param attribute: Name of the attribute to plot or list of names of attributes to plot per point cloud
    :type attribute: str or list
    :param reference_pc: PointCloud object representing the reference point cloud
    :type compare_pcs: PointCloud
    :param bins: Number of bins for the histogram
    :type bins: int
    :param overlay: Whether to overlay the histograms.
    :type overlay: bool
    :param normalize: Whether to normalize the histogram values to show probability density instead of counts.
    :type normalize: bool
    """
    # y-label based on normalization
    if normalize:
        ylabel = "Density"
        histnorm = "probability density"
    else:
        ylabel = "Frequency"
        histnorm = None

    if attribute == "classification":
        bins = np.unique(reference_pc.classification).size

    # get attribute data
    all_data = []
    if reference_pc:
        reference_data = pcfingerprint.pcfingerprint.get_attribute_data(
            reference_pc, attribute
        )
        all_data.append(reference_data)
    else:
        reference_data = None

    if isinstance(attribute, str):
        compare_data = [
            pcfingerprint.pcfingerprint.get_attribute_data(compare_pc, attribute)
            for compare_pc in compare_pcs
        ]
        attr_name = attribute
    elif isinstance(attribute, list):
        compare_data = [
            pcfingerprint.pcfingerprint.get_attribute_data(compare_pc, attr)
            for compare_pc in compare_pcs
            for attr in attribute
        ]
        attr_name = attribute[0].split("_")[0]
    all_data.extend(compare_data)
    bin_edges = histogram_bin_edges(all_data, bins)

    if overlay:  # overlay histograms with transparency
        fig = go.Figure()
        n_colours = len(all_data)
        colors = (
            px.colors.qualitative.Plotly
            if n_colours <= len(px.colors.qualitative.Plotly)
            else px.colors.qualitative.Dark24
        )
        # reference
        if reference_data is not None:
            fig.add_trace(
                _histogram_bar_trace(
                    reference_data,
                    bin_edges,
                    name="Reference Point Cloud",
                    marker_color=colors[0],
                    opacity=0.8,
                    histnorm=histnorm,
                )
            )
        # comparison
        for i, data in enumerate(compare_data):
            color = colors[i + 1]
            fig.add_trace(
                _histogram_bar_trace(
                    data,
                    bin_edges,
                    name=f"Comparison Point Cloud {i+1}",
                    marker_color=color,
                    opacity=0.2,
                    histnorm=histnorm,
                )
            )
        fig.update_layout(barmode="overlay", xaxis_title=attr_name, yaxis_title=ylabel)

        return fig
    else:  # plot histograms side by side
        # determine plot grid based on inputs
        total_plots = len(all_data)
        has_ref = reference_data != None
        nrows, ncols, subplot_titles = plot_grid_layout(total_plots)
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles)

        # get max frequency to scale y-axis equally
        histnorm = "probability density" if normalize else None
        max_freq = 0
        for data in all_data:
            counts, _ = np.histogram(data, bins=bin_edges)
            max_freq = max(max_freq, float(np.max(counts)) if counts.size else 0)

        # ax 0 is always reference pc - if present
        if reference_data is not None:
            fig.add_trace(
                _histogram_bar_trace(
                    reference_data,
                    bin_edges,
                    name="Reference Point Cloud",
                    marker_color="blue",
                    opacity=0.7,
                    histnorm=histnorm,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.update_xaxes(title_text="", row=1, col=1)
            fig.update_yaxes(title_text=ylabel, row=1, col=1)
            if not normalize:
                fig.update_yaxes(range=[0, max_freq * 1.1], row=1, col=1)
        for i, data in enumerate(compare_data):
            subplot_index = i + has_ref
            row = subplot_index // ncols + has_ref
            col = subplot_index % ncols + has_ref
            fig.add_trace(
                _histogram_bar_trace(
                    data,
                    bin_edges,
                    name=f"Comparison Point Cloud {i + 1}",
                    marker_color=px.colors.qualitative.Plotly[
                        i % len(px.colors.qualitative.Plotly)
                    ],
                    opacity=0.7,
                    histnorm=histnorm,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text="", row=row, col=col)
            fig.update_yaxes(title_text=ylabel, row=row, col=col)
            if not normalize:
                fig.update_yaxes(range=[0, max_freq * 1.1], row=row, col=col)

        # hide any unused subplots
        for subplot_index in range(total_plots, nrows * ncols):
            row = subplot_index // ncols + 1
            col = subplot_index % ncols + 1
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)

        fig.update_layout(template="plotly_white", showlegend=False)

        return fig


def plot_histogram_per_class(
    reference_pc,
    compare_pcs: list,
    attribute: str,
    bins: int = 100,
    normalize: bool = False,
):
    """
    Plot histograms of an attribute for reference and comparison point clouds side by side, separated by class.
    If a class is present in the reference point cloud but not in any of the comparison point clouds, or vice versa,
    there will be one empty histogram.

    :param reference_pc: PointCloud object representing the reference point cloud
    :param compare_pcs: List of PointCloud objects representing the comparison point clouds
    :param attribute: Name of the attribute to plot
    :type attribute: str
    :param bins: Number of bins for the histogram
    :type bins: int
    :param normalize: Whether to normalize the histogram values to show probability density instead of counts.
    :type normalize: bool
    """
    classes, hists_reference, hists_compare = prepare_histogram_data_per_class(
        reference_pc, compare_pcs
    )
    figs = []

    for class_value, hist_reference_class, hist_compare_class in zip(
        classes, hists_reference, hists_compare
    ):
        fig = plot_histogram(
            hist_reference_class,
            hist_compare_class,
            attribute=attribute,
            bins=bins,
            overlay=True,
            normalize=normalize,
        )
        fig.update_layout(title_text=f"Class {class_value}")
        figs.append(fig)

    return figs, classes
