"""
Module for visualisations on comparisons between point clouds.
"""

import matplotlib.pyplot as plt
import pcfingerprint
from pcfingerprint.utils import subsample_points, _validate_classes
import numpy as np

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
    real_pc,
    sim_pcs: list,
    attribute: str = None,
    max_n_points: int = 500_000,
    overlay: bool = False,
):
    """
    Create a 3D scatter plot of the real and simulated point clouds, colored by the specified attribute if provided.

    :param real_pc: PointCloud object representing the real-world point cloud
    :param sim_pcs: List of PointCloud objects representing the simulated point clouds
    :param attribute: Name of the attribute to color by. If None, points will not be colored by z-coordinate.
    :type attribute: str
    :param max_n_points: Maximum number of points to plot for each point cloud. If a point cloud has more points, it will be randomly subsampled to this number for visualization purposes.
    :type max_n_points: int
    :param overlay: Whether to overlay all point clouds in one plot (True) or plot them side by side (False). Overlaying only makes sense for two point clouds.
    :type overlay: bool

    :return: The matplotlib figure object containing the 3D scatter plot.
    :rtype: matplotlib.figure.Figure
    """
    # subsample points if necessary
    real_pc, subsample_indices = subsample_points(real_pc, max_n_points)
    sim_pc_sub = [
        subsample_points(sim_pc, max_n_points) for i, sim_pc in enumerate(sim_pcs)
    ]
    sim_pcs, sim_subsample_indices = zip(*sim_pc_sub)

    if overlay:
        # overlay all point clouds in one plot - only really makes sense for two point clouds
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
        n_colours = len(sim_pcs) + 1
        colors = plt.cm.get_cmap("tab10", n_colours)
        # real world
        p = ax.scatter(
            real_pc.coords[:, 0],
            real_pc.coords[:, 1],
            real_pc.coords[:, 2],
            c=colors(0),
            s=0.1,
            label="Real World Point Cloud",
        )
        # simulated
        for i, sim_pc in enumerate(sim_pcs):
            color = colors(i + 1)
            ax.scatter(
                sim_pc.coords[:, 0],
                sim_pc.coords[:, 1],
                sim_pc.coords[:, 2],
                c=color,
                s=0.1,
                label=f"Simulated Point Cloud {i+1}",
            )
        ax.set_title("3D Scatter Plot of Point Clouds")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect(
            (
                np.ptp(real_pc.coords[:, 0]),
                np.ptp(real_pc.coords[:, 1]),
                np.ptp(real_pc.coords[:, 2]),
            )
        )
        plt.legend()
    else:
        if attribute is None:
            print(
                "No attribute specified for coloring. Plotting 3D scatter plot colored by z coordinate instead."
            )
            attribute = "z"
        print(f"Generating 3D scatter plot colored by {attribute}...")

        # get attribute data
        real_data = pcfingerprint.pcfingerprint.get_attribute_data(real_pc, attribute)
        sim_data = [
            pcfingerprint.pcfingerprint.get_attribute_data(sim_pc, attribute)
            for sim_pc in sim_pcs
        ]

        # determine plot grid based on inputs
        if len(sim_pcs) == 1:
            nrows, ncols = 1, 2
        elif len(sim_pcs) <= 3:
            nrows, ncols = 2, 2
        elif len(sim_pcs) <= 5:
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
        vmin = min(real_data.min(), min([data.min() for data in sim_data]))
        vmax = max(real_data.max(), max([data.max() for data in sim_data]))

        # ax 0 is always real pc
        p = axs[0, 0].scatter(
            real_pc.coords[:, 0],
            real_pc.coords[:, 1],
            real_pc.coords[:, 2],
            c=real_data,
            cmap="viridis",
            s=0.2,
            vmin=vmin,
            vmax=vmax,
        )
        axs[0, 0].set_title("Real World Point Cloud")
        style_axis(axs[0, 0])
        # equal scaling
        axs[0, 0].set_box_aspect(
            (
                np.ptp(real_pc.coords[:, 0]),
                np.ptp(real_pc.coords[:, 1]),
                np.ptp(real_pc.coords[:, 2]),
            )
        )
        for i, sim_pc in enumerate(sim_pcs):
            row = (i + 1) // ncols
            col = (i + 1) % ncols
            axs[row, col].scatter(
                sim_pc.coords[:, 0],
                sim_pc.coords[:, 1],
                sim_pc.coords[:, 2],
                c=sim_data[i],
                cmap="viridis",
                s=0.2,
                vmin=vmin,
                vmax=vmax,
            )
            axs[row, col].set_title(f"Simulated Point Cloud {i+1}")
            style_axis(axs[row, col])
            # equal scaling
            axs[row, col].set_box_aspect(
                (
                    np.ptp(sim_pc.coords[:, 0]),
                    np.ptp(sim_pc.coords[:, 1]),
                    np.ptp(sim_pc.coords[:, 2]),
                )
            )
        # colorbar to the right side spanning all rows
        fig.colorbar(p, label=attribute, ax=axs[:, :], location="right")
        # hide any unused subplots
        if len(sim_pcs) + 1 < nrows * ncols:
            axs[row, -1].axis("off")
    return fig


def plot_histogram(
    real_pc,
    sim_pcs: list,
    attribute: str,
    bins: int = 100,
    overlay: bool = False,
    normalize: bool = False,
):
    """
    Plot histograms of an attribute for real and simulated point clouds side by side.

    :param real_pc: PointCloud object representing the real-world point cloud
    :param sim_pcs: List of PointCloud objects representing the simulated point clouds
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
    real_data = pcfingerprint.pcfingerprint.get_attribute_data(real_pc, attribute)
    sim_data = [
        pcfingerprint.pcfingerprint.get_attribute_data(sim_pc, attribute)
        for sim_pc in sim_pcs
    ]

    if overlay:  # overlay histograms with transparency
        fig, ax = plt.subplots(figsize=(8, 6))
        n_colours = len(sim_pcs) + 1
        colors = plt.cm.get_cmap("tab10", n_colours)
        # real world
        ax.hist(
            real_data,
            bins=bins,
            color="blue",
            alpha=0.7,
            label="Real World Point Cloud",
            histtype="step",
            density=normalize,
        )
        # simulated
        for i, data in enumerate(sim_data):
            color = colors(i + 1)
            # 1. draw: inner area with alpha
            ax.hist(
                data,
                bins=bins,
                color=color,
                alpha=0.3,
                label=f"Simulated Point Cloud {i+1}",
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
        if len(sim_pcs) == 1:
            nrows, ncols = 1, 2
        elif len(sim_pcs) <= 3:
            nrows, ncols = 2, 2
        elif len(sim_pcs) <= 5:
            nrows, ncols = 2, 3
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 5))

        # Ensure axs is always 2D for consistent indexing
        if nrows == 1:
            axs = axs.reshape(1, -1)

        # get max frequency to scale y-axis equally
        max_freq = 0
        max_freq = max(max_freq, max(np.histogram(real_data, bins=bins)[0]))
        for data in sim_data:
            max_freq = max(max_freq, max(np.histogram(data, bins=bins)[0]))
        # ax 0 is always real pc
        axs[0, 0].hist(real_data, bins=bins, color="blue", alpha=0.7, density=normalize)
        axs[0, 0].set_title("Real World Point Cloud")
        axs[0, 0].set_xlabel(attribute)
        axs[0, 0].set_ylabel(ylabel)
        # add minor ticks
        axs[0, 0].minorticks_on()
        if not normalize:
            axs[0, 0].set_ylim(0, max_freq * 1.1)
        axs[0, 0].set_axisbelow(True)
        for i, data in enumerate(sim_data):
            row = (i + 1) // ncols
            col = (i + 1) % ncols
            axs[row, col].hist(
                data, bins=bins, color="orange", alpha=0.7, density=normalize
            )
            axs[row, col].set_title(f"Simulated Point Cloud {i+1}")
            axs[row, col].set_ylabel(ylabel)
            # add minor ticks
            axs[row, col].minorticks_on()
            if not normalize:
                axs[row, col].set_ylim(0, max_freq * 1.1)
            axs[row, col].set_axisbelow(True)
        # hide any unused subplots
        if len(sim_pcs) + 1 < nrows * ncols:
            axs[row, -1].axis("off")
        plt.tight_layout()

        return fig


def plot_histogram_per_class(
    real_pc, sim_pcs: list, attribute: str, bins: int = 100, normalize: bool = False
):
    """
    Plot histograms of an attribute for real and simulated point clouds side by side, separated by class.
    If a class is present in the real point cloud but not in any of the simulated point clouds, or vice versa,
    there will be one empty histogram.

    :param real_pc: PointCloud object representing the real-world point cloud
    :param sim_pcs: List of PointCloud objects representing the simulated point clouds
    :param attribute: Name of the attribute to plot
    :type attribute: str
    :param bins: Number of bins for the histogram
    :type bins: int
    :param normalize: Whether to normalize the histogram values to show probability density instead of counts.
    :type normalize: bool
    """

    _validate_classes(real_pc)
    for sim_pc in sim_pcs:
        _validate_classes(sim_pc)

    figs = []

    classes = np.unique(real_pc.classification)
    for sim_pc in sim_pcs:
        sim_classes = np.unique(sim_pc.classification)
        classes = np.union1d(classes, sim_classes)
    for class_value in classes:
        class_filter_real = real_pc.classification == class_value
        class_filters_sim = [sim_pc.classification == class_value for sim_pc in sim_pcs]
        hist_real_world_class = real_pc.mask(class_filter_real)
        hist_simulated_class = [
            sim_pc.mask(class_filters_sim[i]) for i, sim_pc in enumerate(sim_pcs)
        ]
        fig = plot_histogram(
            hist_real_world_class,
            hist_simulated_class,
            attribute,
            bins,
            overlay=True,
            normalize=normalize,
        )
        figs.append(fig)
    return figs, classes
