"""
Module for visualisations on comparisons between point clouds.
"""

import matplotlib.pyplot as plt
import pcfingerprint
import numpy as np

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.grid'] = True
# grid for minor ticks
plt.rcParams['axes.grid.which'] = 'both'
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"


def plot_3d_scatter(real_pc, sim_pc, attribute: str = None):
    pass

def plot_histogram(real_pc, sim_pcs: list, attribute: str, bins: int = 100, overlay: bool = False):
    """
    Plot histograms of an attribute for real and simulated point clouds side by side.
    
    :param real_pc: PointCloud object representing the real-world point cloud
    :param sim_pc: PointCloud object representing the simulated point cloud
    :param attribute: Name of the attribute to plot
    :type attribute: str
    :param bins: Number of bins for the histogram
    :type bins: int
    """
    # get attribute data
    real_data = pcfingerprint.pcfingerprint.get_attribute_data(real_pc, attribute)
    sim_data = [pcfingerprint.pcfingerprint.get_attribute_data(sim_pc, attribute) for sim_pc in sim_pcs]

    if overlay:  # overlay histograms with transparency
        fig, ax = plt.subplots(figsize=(8, 6))
        n_colours = len(sim_pcs) + 1
        colors = plt.cm.get_cmap('tab10', n_colours)
        # real world
        ax.hist(real_data, bins=bins, color='blue', alpha=0.7, label='Real World Point Cloud', histtype='step')
        # simulated
        for i, data in enumerate(sim_data):
            color = colors(i + 1)
            # 1. draw: inner area with alpha
            ax.hist(data, bins=bins, color=color, alpha=0.3, label=f'Simulated Point Cloud {i+1}')
            # 2. draw: only edges with full color
            ax.hist(data, bins=bins, color=color, edgecolor=color, fc="None", lw=1, histtype='step')
        ax.set_axisbelow(True)
        plt.title(f'Histogram of {attribute}')
        plt.xlabel(attribute)
        plt.ylabel('Frequency')
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
        # get max frequency to scale y-axis equally
        max_freq = 0
        max_freq = max(max_freq, max(np.histogram(real_data, bins=bins)[0]))
        for data in sim_data:
            max_freq = max(max_freq, max(np.histogram(data, bins=bins)[0]))
        # ax 0 is always real pc
        axs[0, 0].hist(real_data, bins=bins, color='blue', alpha=0.7)
        axs[0, 0].set_title('Real World Point Cloud')
        axs[0, 0].set_xlabel(attribute)
        axs[0, 0].set_ylabel('Frequency')
        # add minor ticks
        axs[0, 0].minorticks_on()
        axs[0, 0].set_ylim(0, max_freq * 1.1)
        axs[0, 0].set_axisbelow(True)
        for i, data in enumerate(sim_data):
            row = (i + 1) // ncols
            col = (i + 1) % ncols
            axs[row, col].hist(data, bins=bins, color='orange', alpha=0.7)
            axs[row, col].set_title(f'Simulated Point Cloud {i+1}')
            axs[row, col].set_xlabel(attribute)
            axs[row, col].set_ylabel('Frequency')
            # add minor ticks
            axs[row, col].minorticks_on()
            axs[row, col].set_ylim(0, max_freq * 1.1)
            axs[row, col].set_axisbelow(True)
        # hide any unused subplots
        if len(sim_pcs) + 1 < nrows * ncols:
            axs[row, -1].axis('off')
        plt.tight_layout()

        return fig