"""
Module for generating comparison reports between point clouds.
"""

from typing import Union, Optional, List
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numerous.html_report_generator import (
    Report,
    Section,
    Subsection,
    Div,
    Tabs,
    Card,
    DataFrameTable,
)
from numerous.html_report_generator.components.figure import Figure
from numerous.html_report_generator.abstract import ReportInterface
import pcfingerprint
import webbrowser
import http.server
import socketserver


def wrap_div(html, class_=None):
    """Wrap HTML content in a ``div`` element.

    :param html: HTML string to wrap.
    :type html: str
    :param class_: Optional CSS class to add to the wrapper.
    :type class_: str | None
    :return: A ``div``-wrapped HTML string.
    :rtype: str
    """
    if class_:
        return f"<div class='{class_}'>\n{html}\n</div>\n"
    else:
        return f"<div>\n{html}\n</div>\n"


def serve_report_folder(output_folder: str, open_browser: bool = True):
    """Serve a generated report folder over HTTP.

    :param output_folder: Folder containing the generated report files.
    :type output_folder: str
    :param open_browser: Whether to open the report URL in a web browser.
    :type open_browser: bool
    :return: None
    :rtype: None
    """
    folder = Path(output_folder).resolve()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(folder))

    with socketserver.TCPServer(("", 0), handler) as httpd:
        port = httpd.server_address[1]
        url = f"http://localhost:{port}/index.html"
        print(f"Starting server on {url}")

        if open_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


class MatplotlibFigure(Figure):
    """
    Class representing a matplotlib figure in the report
    """

    def __init__(
        self,
        image_name: Path = None,
        caption: str = "",
        notes: Optional[list[str]] = None,
    ):
        """Initialize a matplotlib-backed report figure.

        :param image_name: Path to the matplotlib figure image.
        :type image_name: pathlib.Path | None
        :param caption: Caption text for the figure.
        :type caption: str
        :param notes: Optional notes to be added to the caption.
        :type notes: list[str] | None
        """
        super(MatplotlibFigure, self).__init__(caption, notes)
        self.image_name = image_name

    def _as_html_figure_content(self, report: ReportInterface):
        """Render the matplotlib figure as HTML content.

        :param report: Report instance used for figure numbering.
        :type report: ReportInterface
        :return: HTML fragment for embedding the image.
        :rtype: str
        """

        html = f"""
                <div style='display: inline-flex;'>
                    <img src="{self.image_name.name}" alt=Figure {report.figure_number} style="max-width: 100%"/>
                </div>
                """
        return html


class MatplotlibFigureSelection(Figure):
    """
    Class representing a set of matplotlib figures in the report with a dropdown menu to switch between
    """

    def __init__(
        self,
        image_names: List[Path],
        image_labels: List[str],
        caption: str = "",
        notes: Optional[list[str]] = None,
    ):
        """Initialize a dropdown-controlled figure selection.

        :param image_names: Paths to images available in the selector.
        :type image_names: list[pathlib.Path]
        :param image_labels: Labels shown for each selectable image.
        :type image_labels: list[str]
        :param caption: Caption text for the figure block.
        :type caption: str
        :param notes: Optional notes to include with the caption.
        :type notes: list[str] | None
        :raises ValueError: If image and label counts differ or no images are provided.
        """
        super().__init__(caption, notes)

        if len(image_names) != len(image_labels):
            raise ValueError("image_names and image_labels must have the same length")
        if len(image_names) == 0:
            raise ValueError("At least one image must be provided")

        self.image_names = image_names
        self.image_labels = image_labels

    def _as_html_figure_content(self, report: ReportInterface) -> str:
        """Render figure-selection HTML with dropdown and switchable images.

        :param report: Report instance used to provide unique figure context.
        :type report: ReportInterface
        :return: HTML fragment with selector, image elements, and JavaScript.
        :rtype: str
        """
        element_id = f"fig_sel_{self.id.replace('-', '_')}"

        # Create the dropdown options
        options_html = ""
        for i, label in enumerate(self.image_labels):
            selected = "selected" if i == 0 else ""
            options_html += f'<option value="{i}" {selected}>{label}</option>\n'

        # Create the image tags - FIRST IMAGE BLOCK, OTHERS NONE
        images_html = ""
        for i, name in enumerate(self.image_names):
            display_style = "block" if i == 0 else "none"  # <-- This is the key fix
            images_html += f"""
                <img id="{element_id}_img_{i}"
                    src="{name.name}"
                    alt="{self.image_labels[i]}"
                    style="max-width: 100%; display: {display_style};" />
            """

        html = f"""
        <br><br>
        <div style="margin-bottom: 20px;">
            <div style="margin-bottom: 10px;">
                <label for="{element_id}_dropdown" style="font-weight: bold; margin-right: 10px;"></label>
                <select id="{element_id}_dropdown"
                        onchange="switchImage_{element_id}(this.value)"
                        style="padding: 8px; border-radius: 4px; border: 1px solid #ccc; min-width: 200px;">
                    {options_html}
                </select>
            </div>
            <div style="text-align: center; background-color: white; padding: 10px; border-radius: 4px;">
                {images_html}
            </div>
        </div>

        <script>
        function switchImage_{element_id}(index) {{
            for (let i = 0; i < {len(self.image_names)}; i++) {{
                let img = document.getElementById('{element_id}_img_' + i);
                if (img) img.style.display = i == parseInt(index) ? 'block' : 'none';
            }}
        }}
        // Ensure first image is visible when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            switchImage_{element_id}(0);
        }});
        </script>
        """

        return html


def generate_report(
    real_pc,
    sim_pcs: list,
    output_folder: str,
    attribute_list: list[str],
    summary_statistics: bool = True,
    return_number_metrics: bool = False,
    visualizations: list[str] = ["histogram"],
    vis_backend: str = "matplotlib",
    preview: bool = True,
):
    """Generate an HTML report comparing real and simulated point clouds.

    :param real_pc: Reference point cloud object.
    :param sim_pcs: Simulated point cloud objects to compare against ``real_pc``.
    :type sim_pcs: list
    :param output_folder: Output folder for the generated HTML report.
    :type output_folder: str
    :param attribute_list: Point attributes to analyze and visualize.
    :type attribute_list: list[str]
    :param summary_statistics: Whether to include summary statistics tables.
    :type summary_statistics: bool
    :param return_number_metrics: Whether to include return-number metrics.
    :type return_number_metrics: bool
    :param visualizations: Visualization types to include.
    :type visualizations: list[str]
    :param vis_backend: Visualization backend to use.
    :type vis_backend: str
    :param preview: Whether to start a local web server and open the report in a browser.
    :type preview: bool
    :raises NotImplementedError: If an unsupported visualization backend is requested.
    :return: None
    :rtype: None
    """

    # Create report
    filename = "index.html"
    report = Report(
        target_folder=output_folder,
        filename=filename,
        template="src/pcfingerprint/templates/report_template_fingerprint.html",
    )

    # Add info for the header and title page
    report.add_header_info(
        header="Point Cloud Fingerprint",
        title="Point Cloud Fingerprint Report",
        sub_title=f"Comparison between {real_pc.source} and {len(sim_pcs)} simulated point clouds",
        sub_sub_title=f'Attributes: {", ".join(attribute_list)}',
    )

    block_content = {}

    section = Section(section_title=f"Overview of Point Cloud Files")
    # start with overview of files, as table

    # create a pandas dataframe from this
    df = pd.DataFrame(columns=["Dataset", "Filename"])
    df["Dataset"] = ["Real"] + [f"Simulated {i+1}" for i in range(len(sim_pcs))]
    filenames = []
    try:
        filenames.append(Path(real_pc.source).as_posix())
    except TypeError:
        filenames.append(None)
    for sim_pc in sim_pcs:
        try:
            filenames.append(Path(sim_pc.source).as_posix())
        except TypeError:
            filenames.append(None)
    df["Filename"] = filenames
    table = DataFrameTable(df, caption="Point Cloud Files Used in Comparison")
    section.add_content({"file_table": table})
    block_content["section_overview"] = section

    # section for 3D scatter plot
    if "3d_scatter" in visualizations:
        section = Section(section_title=f"3D Scatter Plot Comparison")
        # generate 3d scatter plot
        if vis_backend == "matplotlib":
            # if using matplotlib, we generate a static PNG image file and reference it from the HTML report
            from pcfingerprint.vis_matplotlib import plot_3d_scatter

            if len(attribute_list) == 0:
                plot_attribute_list = ["z"]
            else:
                plot_attribute_list = attribute_list
            imgs = {}
            for attribute in plot_attribute_list:
                fig = plot_3d_scatter(real_pc, sim_pcs, attribute=attribute)
                # convert fig to base64
                img_file = report.target_folder / Path(f"3d_scatter_{attribute}.png")
                fig.savefig(img_file, format="png")
                plt.close(fig)
                imgs[attribute] = img_file
            figure_selection = MatplotlibFigureSelection(
                image_names=list(imgs.values()),
                image_labels=list(imgs.keys()),
                caption="3D Scatter Plot Comparison",
            )
            section.add_content({"3d_scatter_plot": figure_selection})
        else:
            # other backends not supported yet
            raise NotImplementedError(
                f"Visualization backend {vis_backend} not supported for 3D scatter plot."
            )
        block_content["section_3d_scatter"] = section

    # Add a section for each attribute
    for attribute in attribute_list:
        section = Section(section_title=f"Comparing {attribute}")

        # extract data for attribute
        real_data = pcfingerprint.pcfingerprint.get_attribute_data(real_pc, attribute)
        sim_data = []
        for sim_pc in sim_pcs:
            sim_data.append(
                pcfingerprint.pcfingerprint.get_attribute_data(sim_pc, attribute)
            )

        section_content = {}

        if summary_statistics:
            from pcfingerprint import stats

            # compute summary statistics
            stats_real = stats.compute_statistics(real_data)
            stats_sim = [stats.compute_statistics(data) for data in sim_data]

            # remove stats where all values are NaN across real and simulated (e.g. mode if not int)
            keys_to_remove = []
            for k in stats_real.keys():
                try:
                    if np.isnan(stats_real[k]) and all(
                        [np.isnan(s_stats[k]) for s_stats in stats_sim]
                    ):
                        keys_to_remove.append(k)
                except TypeError:
                    if pd.isnull(stats_real[k]) and all(
                        [pd.isnull(s_stats[k]) for s_stats in stats_sim]
                    ):
                        keys_to_remove.append(k)
            for k in keys_to_remove:
                del stats_real[k]
                for s_stats in stats_sim:
                    del s_stats[k]

            # create a pandas dataframe from this
            keys = stats_real.keys()
            df = pd.DataFrame(
                {
                    "Statistics": keys,
                    f"Real": [stats_real[k] for k in keys],
                }
            )
            df["Real"] = df["Real"].astype(object)
            # add a column for each simulated pc
            for i, s_stats in enumerate(stats_sim):
                df[f"Simulated {i+1}"] = [s_stats[k] for k in keys]
                df[f"Simulated {i+1}"] = df[f"Simulated {i+1}"].astype(object)

            # row formatting
            for k in keys:
                if not "mode" in k:
                    # limit to four decimal places
                    df.loc[df["Statistics"] == k, df.columns != "Statistics"] = df.loc[
                        df["Statistics"] == k, df.columns != "Statistics"
                    ].map(lambda x: f"{x:.4f}")

            table = DataFrameTable(
                df, caption="Summary statistics for attribute: " + attribute
            )

            section_content["summary_stat_table"] = table

        if "histogram" in visualizations or "histogram_overlay" in visualizations:
            if "histogram_overlay" in visualizations:
                overlay = True
            else:
                overlay = False
            # generate histogram plot
            if vis_backend == "matplotlib":
                # if using matplotlib, we generate a static PNG image file and reference it from the HTML report
                from pcfingerprint.vis_matplotlib import plot_histogram

                imgs = []
                for norm, norm_str in ((False, "frequency"), (True, "density")):
                    fig = plot_histogram(
                        real_pc, sim_pcs, attribute, overlay=overlay, normalize=norm
                    )
                    img_file = report.target_folder / Path(
                        f"{attribute}_histogram_{norm_str}.png"
                    )
                    fig.savefig(img_file, format="png")
                    plt.close(fig)
                    imgs.append((norm_str, img_file))
                image_labels = [label for label, _ in imgs]
                image_names = [path for _, path in imgs]
                figure_selection = MatplotlibFigureSelection(
                    image_names=image_names,
                    image_labels=image_labels,
                    caption=f"Histogram of {attribute}",
                )
                section_content["histogram"] = figure_selection
            else:
                # other backends not supported yet
                raise NotImplementedError(
                    f"Visualization backend {vis_backend} not supported for histogram plot."
                )

        if "histogram_per_class" in visualizations:
            # generate histogram plot
            if vis_backend == "matplotlib":
                # if using matplotlib, we generate a static PNG image file and reference it from the HTML report
                from pcfingerprint.vis_matplotlib import plot_histogram_per_class

                imgs = []
                for norm, norm_str in ((False, "frequency"), (True, "density")):
                    figs, classes = plot_histogram_per_class(
                        real_pc, sim_pcs, attribute, normalize=norm
                    )
                    for fig, class_val in zip(figs, classes):
                        img_file = report.target_folder / Path(
                            f"{attribute}_histogram_class{class_val}_{norm_str}.png"
                        )
                        fig.savefig(img_file, format="png")
                        plt.close(fig)
                        imgs.append((f"class_{class_val}_{norm_str}", img_file))
                if len(imgs) > 0:
                    image_labels = [label for label, _ in imgs]
                    image_names = [path for _, path in imgs]
                    figure_selection = MatplotlibFigureSelection(
                        image_names=image_names,
                        image_labels=image_labels,
                        caption=f"Histogram of {attribute} per class",
                    )
                    section_content["histogram per class"] = figure_selection
            else:
                # other backends not supported yet
                raise NotImplementedError(
                    f"Visualization backend {vis_backend} not supported for histogram plot."
                )

        # Add the figure and table to the section
        section.add_content(section_content)

        block_content[f"section_{attribute}"] = section

    # section for return number metrics
    if return_number_metrics:
        from pcfingerprint import stats

        section = Section(section_title=f"Return Number Metrics")

        # compute return number statistics
        metrics_real = stats.compute_return_number_metrics(
            real_pc.return_number, real_pc.number_of_returns
        )
        metrics_sim = [
            stats.compute_return_number_metrics(
                sim_pc.return_number, sim_pc.number_of_returns
            )
            for sim_pc in sim_pcs
        ]

        # create a pandas dataframe from this
        keys = metrics_real.keys()
        df = pd.DataFrame(
            {
                "Statistics": keys,
                f"Real": [metrics_real[k] for k in keys],
            }
        )
        df["Real"] = df["Real"].astype(object)
        # add a column for each simulated pc
        for i, s_metrics in enumerate(metrics_sim):
            df[f"Simulated {i+1}"] = [s_metrics[k] for k in keys]
            df[f"Simulated {i+1}"] = df[f"Simulated {i+1}"].astype(object)
        # row formatting
        for k in keys:
            if "fraction" in k:
                df.loc[df["Statistics"] == k, df.columns != "Statistics"] = df.loc[
                    df["Statistics"] == k, df.columns != "Statistics"
                ].map(lambda x: f"{x:.3%}")
            elif "number" in k:
                # as integer with comma as thousand separator
                df.loc[df["Statistics"] == k, df.columns != "Statistics"] = df.loc[
                    df["Statistics"] == k, df.columns != "Statistics"
                ].map(lambda x: f"{int(x):,}")
        table = DataFrameTable(df, caption="Return Number Metrics")

        section.add_content({"return_number_table": table})
        block_content["section_return_number_metrics"] = section

    # Add the sections to the report
    report.add_blocks(block_content)

    # Save the report - creates the html output file
    report.save()

    if preview:
        serve_report_folder(output_folder=output_folder, open_browser=True)
