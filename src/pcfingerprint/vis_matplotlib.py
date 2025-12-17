"""
Module for visualisations on comparisons between point clouds.
"""

import matplotlib.pyplot as plt
import pcfingerprint


def plot_3d_scatter(real_pc, sim_pc, attribute: str = None):
    pass

def plot_histogram(real_pc, sim_pcs: list, attribute: str, bins: int = 100):
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

    # 1) plot histograms side by side
    # TODO: b) overlapping histogram with transparency, maybe also compute the IOU between the two histograms?
    
    # determine plot grid based on inputs
    if len(sim_pcs) == 1:
        nrows, ncols = 1, 2
    elif len(sim_pcs) <= 3:
        nrows, ncols = 2, 2
    elif len(sim_pcs) <= 5:
        nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 5))
    # ax 0 is alrways real pc
    axs[0, 0].hist(real_data, bins=bins, color='blue', alpha=0.7)
    axs[0, 0].set_title('Real World Point Cloud')
    axs[0, 0].set_xlabel(attribute)
    axs[0, 0].set_ylabel('Frequency')
    for i, data in enumerate(sim_data):
        row = (i + 1) // ncols
        col = (i + 1) % ncols
        axs[row, col].hist(data, bins=bins, color='orange', alpha=0.7)
        axs[row, col].set_title(f'Simulated Point Cloud {i+1}')
        axs[row, col].set_xlabel(attribute)
        axs[row, col].set_ylabel('Frequency')
    # hide any unused subplots
    if len(sim_pcs) + 1 < nrows * ncols:
        axs[row, -1].axis('off')
    plt.tight_layout()

    return fig