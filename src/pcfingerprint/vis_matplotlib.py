"""
Module for visualisations on comparisons between point clouds.
"""

import matplotlib.pyplot as plt
import pcfingerprint
from pcfingerprint.utils import subsample_points, _validate_classes
from pcfingerprint.vis_common import (
    prepare_attribute_data,
    plot_grid_layout,
    prepare_distance_scatter_comp_data,
    prepare_distance_scatter_ref_data,
    prepare_histogram_data_per_class,
)
import numpy as np
import logging

logger = logging.getLogger("pcfingerprint")

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.grid"] = True
# grid for minor ticks
plt.rcParams["axes.grid.which"] = "both"
plt.rcParams["grid.alpha"] = 1
plt.rcParams["grid.color"] = "#cccccc"


def style_axis(ax):
    """Utility function to style the axes of the 3D scatter plot for a cleaner look."""
    ax.set_xlabel("X", labelpad=-10)
    ax.set_ylabel("Y", labelpad=-10)
    ax.set_zlabel("Z", labelpad=-10)
    # disable tick labels for cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def plot_3d_scatter(
    reference_pc,
    compare_pcs: list,
    attribute: str = None,
    max_n_points: int = 500_000,
    overlay: bool = False,
):
    """
    Create a 3D scatter plot of the reference and comparison point clouds, colored by the specified attribute if provided.

    :param reference_pc: PointCloud object representing the reference point cloud
    :param compare_pcs: List of PointCloud objects representing the comparison point clouds
    :param attribute: Name of the attribute to color by. If None, points will not be colored by z-coordinate.
    :type attribute: str
    :param max_n_points: Maximum number of points to plot for each point cloud. If a point cloud has more points, it will be randomly subsampled to this number for visualization purposes.
    :type max_n_points: int
    :param overlay: Whether to overlay all point clouds in one plot (True) or plot them side by side (False). Overlaying only makes sense for two point clouds.
    :type overlay: bool

    :return: The matplotlib figure object containing the 3D scatter plot.
    :rtype: matplotlib.figure.Figure
    """
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

    if attribute is None:
        logger.info(
            "No attribute specified for coloring. Plotting 3D scatter plot colored by z coordinate instead."
        )
        attribute = "z"
    logger.info(f"Generating 3D scatter plot colored by {attribute}...")

    if overlay:
        # overlay all point clouds in one plot - only really makes sense for two point clouds
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
        n_colours = len(compare_pcs) + 1
        colors = plt.cm.get_cmap("tab10", n_colours)
        # reference
        p = ax.scatter(
            reference_pc_pl[:, 0],
            reference_pc_pl[:, 1],
            reference_pc_pl[:, 2],
            c=colors(0),
            s=0.1,
            label="Reference Point Cloud",
        )
        # comparison
        for i in range(len(compare_pcs)):
            color = colors(i + 1)
            ax.scatter(
                compare_pc_pl[i][:, 0],
                compare_pc_pl[i][:, 1],
                compare_pc_pl[i][:, 2],
                c=color,
                s=0.1,
                label=f"Comparison Point Cloud {i+1}",
            )
        ax.set_title("3D Scatter Plot of Point Clouds")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect(
            (
                np.ptp(reference_pc_pl[:, 0]),
                np.ptp(reference_pc_pl[:, 1]),
                np.ptp(reference_pc_pl[:, 2]),
            )
        )
        plt.legend()
    else:
        # determine plot grid based on inputs
        total_plots = len(compare_pcs) + 1
        nrows, ncols, _ = plot_grid_layout(total_plots)
        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(14, 9),
            dpi=100,
            subplot_kw={"projection": "3d"},
            layout="constrained",
        )

        # Ensure axs is always 2D for consistent indexing
        if nrows == 1:
            axs = axs.reshape(1, -1)

        # compute vmin/vmax (fall back to prepared values)
        vmin = (
            vmin
            if vmin is not None
            else min(reference_data.min(), min([data.min() for data in compare_data]))
        )
        vmax = (
            vmax
            if vmax is not None
            else max(reference_data.max(), max([data.max() for data in compare_data]))
        )

        # viridis for most attributes, but use a different colorscale for classification to better distinguish classes
        if attribute == "classification":
            cmap = plt.get_cmap(
                "Set1",
                (
                    class_values.size
                    if class_values is not None
                    else np.unique(reference_data).size
                ),
            )
        else:
            cmap = "viridis"

        # ax 0 is always reference pc
        p = axs[0, 0].scatter(
            reference_pc_pl[:, 0],
            reference_pc_pl[:, 1],
            reference_pc_pl[:, 2],
            c=reference_data,
            cmap=cmap,
            s=0.2,
            vmin=vmin,
            vmax=vmax,
        )
        axs[0, 0].set_title("Reference Point Cloud")
        style_axis(axs[0, 0])
        # equal scaling
        axs[0, 0].set_aspect("equal")
        for i in range(len(compare_pcs)):
            row = (i + 1) // ncols
            col = (i + 1) % ncols
            axs[row, col].scatter(
                compare_pc_pl[i][:, 0],
                compare_pc_pl[i][:, 1],
                compare_pc_pl[i][:, 2],
                c=compare_data[i],
                cmap=cmap,
                s=0.2,
                vmin=vmin,
                vmax=vmax,
            )
            axs[row, col].set_title(f"Comparison Point Cloud {i+1}")
            style_axis(axs[row, col])
            # equal scaling
            axs[row, col].set_aspect("equal")
        # colorbar to the right side spanning all rows
        if attribute == "classification":
            cbar = fig.colorbar(
                p, label=attribute, ax=axs[:, :], location="right", ticks=[]
            )
            cbar.ax.set_yticklabels([])
        else:
            fig.colorbar(p, label=attribute, ax=axs[:, :], location="right")
        # hide any unused subplots
        if len(compare_pcs) + 1 < nrows * ncols:
            axs[row, -1].axis("off")
    return fig


def plot_3d_scatter_dist_ref(
    pc, attribute_list: list, max_n_points: int = 50_000, vmax: float = None
):
    """
    Create a 3D scatter plot of the reference point cloud, colored by the specified distance attributes.

    :param pc: PointCloud object representing the reference point cloud
    :param attribute_list: List of distance attributes to color by
    :type attribute_list: list
    :param max_n_points: Maximum number of points to plot. If the point cloud has more points, it will be randomly subsampled to this number for visualization purposes.
    :type max_n_points: int
    :param vmax: Maximum value for the color scale. If None, it will be determined from the data.
    :type vmax: float

    :return: The matplotlib figure object containing the 3D scatter plot.
    :rtype: matplotlib.figure.Figure
    """
    pc_pl, values, vmin, inferred_vmax = prepare_distance_scatter_ref_data(
        pc, attribute_list, max_n_points=max_n_points
    )
    if vmax is not None:
        inferred_vmax = vmax

    colorscale = "viridis"

    # determine plot grid based on number of attributes
    if len(attribute_list) == 1:
        nrows, ncols = 1, 1
    elif len(attribute_list) <= 4:
        nrows, ncols = 2, 2
    elif len(attribute_list) <= 6:
        nrows, ncols = 2, 3
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 9),
        dpi=100,
        subplot_kw={"projection": "3d"},
        layout="constrained",
    )
    # Ensure axs is always 2D for consistent indexing
    if nrows == 1:
        axs = axs.reshape(1, -1)

    for i, attribute in enumerate(attribute_list):
        row = i // ncols
        col = i % ncols
        p = axs[row, col].scatter(
            pc_pl[:, 0],
            pc_pl[:, 1],
            pc_pl[:, 2],
            c=values[i],
            cmap=colorscale,
            s=0.2,
            vmin=vmin,
            vmax=inferred_vmax,
        )
        style_axis(axs[row, col])
        axs[row, col].set_title(f"Reference Point Cloud - {attribute}")
        axs[row, col].set_aspect("equal")
    # colorbar to the right side spanning all rows
    fig.colorbar(p, label="Distance", ax=axs[:, :], location="right")
    # hide any unused subplots
    if len(attribute_list) < nrows * ncols:
        axs[row, -1].axis("off")
    return fig


def plot_3d_scatter_dist_comp(
    pc_list: list, attribute: str, max_n_points: int = 50_000, vmax: float = None
):
    """
    Create a 3D scatter plot of the comparison point clouds, colored by the specified distance attribute.

    :param pc_list: List of PointCloud objects representing the comparison point clouds
    :param attribute: Name of the distance attribute to color by
    :type attribute: str
    :param max_n_points: Maximum number of points to plot for each point cloud. If a point cloud has more points, it will be randomly subsampled to this number for visualization purposes.
    :type max_n_points: int
    :param vmax: Maximum value for the color scale. If None, it will be determined from the data.
    :type vmax: float

    :return: The matplotlib figure object containing the 3D scatter plot.
    :rtype: matplotlib.figure.Figure
    """
    pc_pl_list, values, vmin, inferred_vmax = prepare_distance_scatter_comp_data(
        pc_list, attribute, max_n_points=max_n_points
    )
    if vmax is not None:
        inferred_vmax = vmax

    colorscale = "viridis"

    # determine plot grid based on number of comparison point clouds
    if len(pc_list) == 1:
        nrows, ncols = 1, 1
    elif len(pc_list) <= 4:
        nrows, ncols = 2, 2
    elif len(pc_list) <= 6:
        nrows, ncols = 2, 3
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 9),
        dpi=100,
        subplot_kw={"projection": "3d"},
        layout="constrained",
    )
    # Ensure axs is always 2D for consistent indexing
    if nrows == 1:
        axs = axs.reshape(1, -1)

    for i in range(len(pc_pl_list)):
        row = i // ncols
        col = i % ncols
        p = axs[row, col].scatter(
            pc_pl_list[i][:, 0],
            pc_pl_list[i][:, 1],
            pc_pl_list[i][:, 2],
            c=values[i],
            cmap=colorscale,
            s=0.2,
            vmin=vmin,
            vmax=inferred_vmax,
        )
        style_axis(axs[row, col])
        axs[row, col].set_title(f"Comparison Point Cloud {i+1} - {attribute}")
        axs[row, col].set_aspect("equal")

    # colorbar to the right side spanning all rows
    fig.colorbar(p, label="Distance", ax=axs[:, :], location="right")
    # hide any unused subplots
    if len(pc_list) < nrows * ncols:
        axs[row, -1].axis("off")
    return fig


def plot_histogram(
    compare_pcs: list,
    attribute: str,
    reference_pc=None,
    bins: int = 100,
    overlay: bool = False,
    normalize: bool = False,
):
    """
    Plot histograms of an attribute for reference and comparison point clouds side by side.

    :param reference_pc: PointCloud object representing the reference point cloud
    :param compare_pcs: List of PointCloud objects representing the comparison point clouds
    :param attribute: Name of the attribute to plot
    :type attribute: str
    :param overlay: Whether to overlay the histograms (True) or plot them side by side (False)
    :type overlay: bool
    :param bins: Number of bins for the histogram
    :type bins: int
    :param normalize: Whether to normalize the histogram values to show probability density instead of counts.
    :type normalize: bool
    """
    # y-label based on normalization
    if normalize:
        ylabel = "Density"
    else:
        ylabel = "Frequency"

    # get attribute data
    all_data = []
    if reference_pc:
        reference_data = pcfingerprint.pcfingerprint.get_attribute_data(
            reference_pc, attribute
        )
        all_data.append(reference_data)
    else:
        reference_data = None
    compare_data = [
        pcfingerprint.pcfingerprint.get_attribute_data(compare_pc, attribute)
        for compare_pc in compare_pcs
    ]
    all_data.extend(compare_data)
    total_n_plots = len(all_data)

    if attribute == "classification":
        bins = np.unique(reference_pc.classification).size

    if overlay:  # overlay histograms with transparency
        fig, ax = plt.subplots(figsize=(8, 6))
        n_colours = total_n_plots
        colors = plt.cm.get_cmap("tab10", n_colours)
        # reference
        if reference_data is not None:
            ax.hist(
                reference_data,
                bins=bins,
                color="blue",
                alpha=0.7,
                label="Reference Point Cloud",
                histtype="step",
                density=normalize,
            )
        # comparison
        for i, data in enumerate(compare_data):
            color = colors(i + 1)
            # 1. draw: inner area with alpha
            ax.hist(
                data,
                bins=bins,
                color=color,
                alpha=0.3,
                label=f"Comparison Point Cloud {i+1}",
                density=normalize,
            )
            # 2. draw: only edges with full color
            ax.hist(
                data,
                bins=bins,
                color=color,
                edgecolor=color,
                fc="None",
                lw=1,
                histtype="step",
                density=normalize,
            )
        ax.set_axisbelow(True)
        plt.title(f"Histogram of {attribute}")
        plt.xlabel(attribute)
        plt.ylabel(ylabel)
        # add minor ticks
        plt.minorticks_on()
        plt.legend()
        plt.tight_layout()

        return plt.gcf()
    else:  # plot histograms side by side
        # determine plot grid based on inputs
        has_reference = reference_data is not None
        nrows, ncols = plot_grid_layout(total_n_plots, all_comparison=not has_reference)
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 5))

        # Ensure axs is always 2D for consistent indexing
        if nrows == 1:
            axs = axs.reshape(1, -1)

        # get max frequency to scale y-axis equally
        max_freq = 0
        if reference_data:
            max_freq = max(max_freq, max(np.histogram(reference_data, bins=bins)[0]))
        for data in compare_data:
            max_freq = max(max_freq, max(np.histogram(data, bins=bins)[0]))
        # ax 0 is always reference pc
        if has_reference:
            axs[0, 0].hist(
                reference_data, bins=bins, color="blue", alpha=0.7, density=normalize
            )
            axs[0, 0].set_title("Reference Point Cloud")
            axs[0, 0].set_xlabel(attribute)
            axs[0, 0].set_ylabel(ylabel)
            # add minor ticks
            axs[0, 0].minorticks_on()
            if not normalize:
                axs[0, 0].set_ylim(0, max_freq * 1.1)
            axs[0, 0].set_axisbelow(True)
        for i, data in enumerate(compare_data):
            row = (i + has_reference) // ncols
            col = (i + has_reference) % ncols
            axs[row, col].hist(
                data, bins=bins, color="orange", alpha=0.7, density=normalize
            )
            axs[row, col].set_title(f"Comparison Point Cloud {i+1}")
            axs[row, col].set_ylabel(ylabel)
            # add minor ticks
            axs[row, col].minorticks_on()
            if not normalize:
                axs[row, col].set_ylim(0, max_freq * 1.1)
            axs[row, col].set_axisbelow(True)
        # hide any unused subplots
        if len(compare_pcs) + 1 < nrows * ncols:
            axs[row, -1].axis("off")
        plt.tight_layout()

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

    for hist_reference_class, hist_compare_class in zip(hists_reference, hists_compare):

        fig = plot_histogram(
            compare_pcs=hist_compare_class,
            attribute=attribute,
            reference_pc=hist_reference_class,
            bins=bins,
            overlay=True,
            normalize=normalize,
        )
        figs.append(fig)

    return figs, classes
