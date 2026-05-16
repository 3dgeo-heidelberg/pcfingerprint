"""
Module for visualisations on comparisons between point clouds.
"""

import matplotlib.pyplot as plt
import pcfingerprint
from pcfingerprint.utils import subsample_points, _validate_classes
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
    # only do subsampling if attributes is not voxel-based
    if attribute not in reference_pc.voxel_features:
        # subsample points if necessary
        reference_pc, _ = subsample_points(reference_pc, max_n_points)
        compare_pc_sub = [
            subsample_points(compare_pc, max_n_points)
            for i, compare_pc in enumerate(compare_pcs)
        ]
        compare_pcs, _ = zip(*compare_pc_sub)
        reference_pc_pl = reference_pc.coords
        compare_pc_pl = [compare_pc.coords for compare_pc in compare_pcs]

        if attribute is None:
            logger.info(
                "No attribute specified for coloring. Plotting 3D scatter plot colored by z coordinate instead."
            )
            attribute = "z"
        logger.info(f"Generating 3D scatter plot colored by {attribute}...")
    else:
        reference_pc_pl = reference_pc.voxel_features[attribute][
            1
        ]  # closest_to_centroids
        compare_pc_pl = [
            compare_pc.voxel_features[attribute][1] for compare_pc in compare_pcs
        ]  # closest_to_centroids

    # get attribute data
    reference_data = pcfingerprint.pcfingerprint.get_attribute_data(
        reference_pc, attribute
    )
    compare_data = [
        pcfingerprint.pcfingerprint.get_attribute_data(compare_pc, attribute)
        for compare_pc in compare_pcs
    ]

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
        if attribute is None:
            logger.info(
                "No attribute specified for coloring. Plotting 3D scatter plot colored by z coordinate instead."
            )
            attribute = "z"
        logger.info(f"Generating 3D scatter plot colored by {attribute}...")

        # determine plot grid based on inputs
        if len(compare_pcs) == 1:
            nrows, ncols = 1, 2
        elif len(compare_pcs) <= 3:
            nrows, ncols = 2, 2
        elif len(compare_pcs) <= 5:
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

        # compute global vmin and vmax for color scale
        vmin = min(reference_data.min(), min([data.min() for data in compare_data]))
        vmax = max(reference_data.max(), max([data.max() for data in compare_data]))

        # viridis for most attributes, but use a different colorscale for classification to better distinguish classes
        if attribute == "classification":
            cmap = plt.get_cmap("Set1", np.unique(reference_data).size)
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
        axs[0, 0].set_box_aspect(
            (
                np.ptp(reference_pc_pl[:, 0]),
                np.ptp(reference_pc_pl[:, 1]),
                np.ptp(reference_pc_pl[:, 2]),
            )
        )
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
            axs[row, col].set_box_aspect(
                (
                    np.ptp(compare_pc_pl[i][:, 0]),
                    np.ptp(compare_pc_pl[i][:, 1]),
                    np.ptp(compare_pc_pl[i][:, 2]),
                )
            )
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


def plot_histogram(
    reference_pc,
    compare_pcs: list,
    attribute: str,
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
    reference_data = pcfingerprint.pcfingerprint.get_attribute_data(
        reference_pc, attribute
    )
    compare_data = [
        pcfingerprint.pcfingerprint.get_attribute_data(compare_pc, attribute)
        for compare_pc in compare_pcs
    ]

    if attribute == "classification":
        bins = np.unique(reference_pc.classification).size + 1

    if overlay:  # overlay histograms with transparency
        fig, ax = plt.subplots(figsize=(8, 6))
        n_colours = len(compare_pcs) + 1
        colors = plt.cm.get_cmap("tab10", n_colours)
        # reference world
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
        if len(compare_pcs) == 1:
            nrows, ncols = 1, 2
        elif len(compare_pcs) <= 3:
            nrows, ncols = 2, 2
        elif len(compare_pcs) <= 5:
            nrows, ncols = 2, 3
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 5))

        # Ensure axs is always 2D for consistent indexing
        if nrows == 1:
            axs = axs.reshape(1, -1)

        # get max frequency to scale y-axis equally
        max_freq = 0
        max_freq = max(max_freq, max(np.histogram(reference_data, bins=bins)[0]))
        for data in compare_data:
            max_freq = max(max_freq, max(np.histogram(data, bins=bins)[0]))
        # ax 0 is always reference pc
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
            row = (i + 1) // ncols
            col = (i + 1) % ncols
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

    _validate_classes(reference_pc)
    for compare_pc in compare_pcs:
        _validate_classes(compare_pc)

    figs = []

    classes = np.unique(reference_pc.classification)
    for compare_pc in compare_pcs:
        compare_classes = np.unique(compare_pc.classification)
        classes = np.union1d(classes, compare_classes)
    for class_value in classes:
        class_filter_reference = reference_pc.classification == class_value
        class_filters_compare = [
            compare_pc.classification == class_value for compare_pc in compare_pcs
        ]
        hist_reference_class = reference_pc.mask(class_filter_reference)
        hist_compare_class = [
            compare_pc.mask(class_filters_compare[i])
            for i, compare_pc in enumerate(compare_pcs)
        ]
        fig = plot_histogram(
            hist_reference_class,
            hist_compare_class,
            attribute,
            bins,
            overlay=True,
            normalize=normalize,
        )
        figs.append(fig)
    return figs, classes
