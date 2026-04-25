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


def plot_3d_scatter(real_pc, sim_pc, color_by=None):
    pass


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
        fig = go.Figure()
        n_colours = len(sim_pcs) + 1
        colors = (
            px.colors.qualitative.Plotly
            if n_colours <= len(px.colors.qualitative.Plotly)
            else px.colors.qualitative.Dark24
        )
        # real world
        fig.add_trace(
            go.Histogram(
                x=real_data,
                histnorm="density" if normalize else None,
                name="Real World Point Cloud",
                marker_color="blue",
                opacity=0.7,
            )
        )
        # simulated
        for i, data in enumerate(sim_data):
            color = colors[i]
            fig.add_trace(
                go.Histogram(
                    x=data,
                    histnorm="density" if normalize else None,
                    name=f"Simulated Point Cloud {i+1}",
                    marker_color=color,
                    opacity=0.3,
                )
            )
        fig.update_layout(barmode="overlay", xaxis_title=attribute, yaxis_title=ylabel)
        fig.update_traces(opacity=0.7)

        return fig
    else:  # plot histograms side by side
        # determine plot grid based on inputs
        total_plots = len(sim_pcs) + 1
        ncols = 2 if total_plots <= 4 else 3
        nrows = math.ceil(total_plots / ncols)
        subplot_titles = [""] * (nrows * ncols)
        subplot_titles[0] = "Real World Point Cloud"
        for i in range(len(sim_pcs)):
            subplot_titles[i + 1] = f"Simulated Point Cloud {i + 1}"

        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles)

        # get max frequency to scale y-axis equally
        max_freq = 0
        max_freq = max(max_freq, max(np.histogram(real_data, bins=bins)[0]))
        for data in sim_data:
            max_freq = max(max_freq, max(np.histogram(data, bins=bins)[0]))
        histnorm = "probability density" if normalize else None

        # ax 0 is always real pc
        fig.add_trace(
            go.Histogram(
                x=real_data,
                nbinsx=bins,
                histnorm=histnorm,
                name="Real World Point Cloud",
                marker_color="blue",
                opacity=0.7,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_yaxes(title_text=ylabel, row=1, col=1)
        if not normalize:
            fig.update_yaxes(range=[0, max_freq * 1.1], row=1, col=1)
        for i, data in enumerate(sim_data):
            subplot_index = i + 1
            row = subplot_index // ncols + 1
            col = subplot_index % ncols + 1
            fig.add_trace(
                go.Histogram(
                    x=data,
                    nbinsx=bins,
                    histnorm=histnorm,
                    name=f"Simulated Point Cloud {i + 1}",
                    marker_color=px.colors.qualitative.Plotly[
                        i % len(px.colors.qualitative.Plotly)
                    ],
                    opacity=0.7,
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
