"""
Module for visualisations on comparisons between point clouds.
"""

import matplotlib.pyplot as plt
import pcfingerprint


def plot_3d_scatter(real_pc, sim_pc, attribute: str = None):
    pass

def plot_histogram(real_pc, sim_pc, attribute: str, bins: int = 100):
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
    sim_data = pcfingerprint.pcfingerprint.get_attribute_data(sim_pc, attribute)

    # plot histograms side by side
    # TODO: Rather make an overlapping histogram with transparency, maybe also compute the IOU between the two histograms?
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(real_data, bins=bins, color='blue', alpha=0.7)
    axs[0].set_title('Real World Point Cloud')
    axs[0].set_xlabel(attribute)
    axs[0].set_ylabel('Frequency')
    axs[1].hist(sim_data, bins=bins, color='orange', alpha=0.7)
    axs[1].set_title('Simulated Point Cloud')
    axs[1].set_xlabel(attribute)
    axs[1].set_ylabel('Frequency')
    plt.tight_layout()

    return fig