"""
Module for generating comparison reports between point clouds.
"""

from __future__ import annotations

import base64
import http.server
import logging
import socketserver
import webbrowser
from dataclasses import dataclass, field
from functools import partial
from importlib.resources import files
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numerous.html_report_generator import DataFrameTable, Report, Section, Tabs
from numerous.html_report_generator.abstract import ReportInterface
from numerous.html_report_generator.components.figure import Figure

import pcfingerprint

logger = logging.getLogger("pcfingerprint")

IMG_FOLDER = "images"


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


def serve_report_folder(output_folder: str | Path, open_browser: bool = True):
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
        logger.info(f"Starting server on {url}")

        if open_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("\nServer stopped.")


def inline_report_header_logo(report_file: Path) -> None:
    """Replace the header logo reference with a self-contained data URI."""

    # Try development path first (most reliable for active development)
    logo_path = Path(__file__).resolve().parents[2] / "pcfingerprint_logo.png"

    # If not found in development location, try installed package location
    if not logo_path.exists():
        try:
            # Traversable has no ``parent``, so go through a concrete Path
            package_dir = Path(str(files("pcfingerprint")))
            logo_path = package_dir.parent / "pcfingerprint_logo.png"
        except (TypeError, AttributeError, ModuleNotFoundError):
            pass

    if not report_file.exists() or not logo_path.exists():
        return

    old_logo_reference = 'const logoUrl = "../pcfingerprint_logo.png";'
    report_content = report_file.read_text(encoding="utf-8")
    if old_logo_reference not in report_content:
        return

    logo_base64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    logo_data_uri = f"data:image/png;base64,{logo_base64}"
    report_content = report_content.replace(
        old_logo_reference, f'const logoUrl = "{logo_data_uri}";'
    )
    report_file.write_text(report_content, encoding="utf-8")


class MatplotlibFigure(Figure):
    """
    Class representing a matplotlib figure in the report
    """

    def __init__(
        self,
        image_path: Path | str,
        caption: str = "",
        notes: list[str] | None = None,
    ):
        """Initialize a matplotlib-backed report figure.

        :param image_name: Path to the matplotlib figure image.
        :type image_name: pathlib.Path | str
        :param caption: Caption text for the figure.
        :type caption: str
        :param notes: Optional notes to be added to the caption.
        :type notes: list[str] | None
        """
        super().__init__(caption, notes)
        self.image_path = image_path

    def _as_html_figure_content(self, report: ReportInterface):
        """Render the matplotlib figure as HTML content.

        :param report: Report instance used for figure numbering.
        :type report: ReportInterface
        :return: HTML fragment for embedding the image.
        :rtype: str
        """

        html = f"""
                <div style='display: inline-flex;'>
                    <img src="{self.image_path}" alt=Figure {report.figure_number} style="max-width: 100%"/>
                </div>
                """
        return html


class IFrameFigure(Figure):
    """Class representing an HTML iframe-based figure in the report."""

    def __init__(
        self,
        html_path: Path | str,
        caption: str = "",
        notes: list[str] | None = None,
    ):
        """Initialize an iframe-backed report figure.

        :param html_path: Path to the HTML file rendered in the iframe.
        :type html_path: pathlib.Path | str
        :param caption: Caption text for the figure.
        :type caption: str
        :param notes: Optional notes to include with the caption.
        :type notes: list[str] | None
        """
        super().__init__(caption, notes)
        self.html_path = html_path

    def _as_html_figure_content(self, report: ReportInterface):
        """Render the iframe figure as HTML content.

        :param report: Report instance used for figure numbering.
        :type report: ReportInterface
        :return: HTML fragment for embedding the iframe.
        :rtype: str
        """
        html = f"""
                <div style='display: inline-flex; width: 100%;'>
                    <iframe data-src="{self.html_path}" src="about:blank" title="Figure {report.figure_number}" width="100%" height="600px" style="border:none;"></iframe>
                </div>
                """
        return html


class FigureSelection(Figure):
    """
    Class representing a set of figures in the report with a dropdown menu to switch between.
    Supports either image files (matplotlib) or HTML iframes (plotly).
    """

    def __init__(
        self,
        figure_paths: list[Path],
        figure_labels: list[str],
        render_mode: str = "image",
        caption: str = "",
        notes: list[str] | None = None,
    ):
        """Initialize a dropdown-controlled figure selection.

        :param figure_paths: Paths to files available in the selector.
        :type figure_paths: list[pathlib.Path]
        :param figure_labels: Labels shown for each selectable file.
        :type figure_labels: list[str]
        :param render_mode: Rendering mode, either ``image`` or ``iframe``.
        :type render_mode: str
        :param caption: Caption text for the figure block.
        :type caption: str
        :param notes: Optional notes to include with the caption.
        :type notes: list[str] | None
        :raises ValueError: If file and label counts differ, no files are provided, or mode is invalid.
        """
        super().__init__(caption, notes)

        if len(figure_paths) != len(figure_labels):
            raise ValueError("figure_paths and figure_labels must have the same length")
        if len(figure_paths) == 0:
            raise ValueError("At least one figure must be provided")
        if render_mode not in ("image", "iframe"):
            raise ValueError("render_mode must be either 'image' or 'iframe'")

        self.figure_paths = figure_paths
        self.figure_labels = figure_labels
        self.render_mode = render_mode

    def _as_html_figure_content(self, report: ReportInterface) -> str:
        """Render figure-selection HTML with dropdown and switchable images.

        :param report: Report instance used to provide unique figure context.
        :type report: ReportInterface
        :return: HTML fragment with selector, image elements, and JavaScript.
        :rtype: str
        """
        figure_id = (self.id or f"{id(self):x}").replace("-", "_")
        element_id = f"fig_sel_{figure_id}"

        # Create the dropdown options
        options_html = ""
        for i, label in enumerate(self.figure_labels):
            selected = "selected" if i == 0 else ""
            options_html += f'<option value="{i}" {selected}>{label}</option>\n'

        # Create the content tags
        images_html = ""
        for i, figure_path in enumerate(self.figure_paths):
            display_style = "block" if i == 0 else "none"
            if self.render_mode == "image":
                images_html += f"""
                <img id="{element_id}_img_{i}"
                    src="{figure_path}"
                    alt="{self.figure_labels[i]}"
                    style="max-width: 100%; display: {display_style};" />
                """
            else:
                images_html += f"""
                <iframe id="{element_id}_img_{i}"
                    data-src="{figure_path}"
                    src="about:blank"
                    width="100%"
                    height="600px"
                    style="border:none; display: {display_style};"></iframe>
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
        function loadIframe_{element_id}(iframe) {{
            if (!iframe || iframe.dataset.loaded === 'true') {{
                return;
            }}

            const iframeSrc = iframe.dataset.src;
            if (!iframeSrc) {{
                return;
            }}

            iframe.addEventListener('load', function () {{
                try {{
                    iframe.contentWindow.dispatchEvent(new iframe.contentWindow.Event('resize'));
                }} catch (error) {{
                    // Ignore timing issues while the iframe finishes loading.
                }}
            }}, {{ once: true }});

            iframe.dataset.loaded = 'true';
            iframe.src = iframeSrc;
        }}

        function switchImage_{element_id}(index) {{
            for (let i = 0; i < {len(self.figure_paths)}; i++) {{
                let img = document.getElementById('{element_id}_img_' + i);
                if (!img) continue;

                const isActive = i == parseInt(index);
                img.style.display = isActive ? 'block' : 'none';
                if (isActive && '{self.render_mode}' === 'iframe') {{
                    loadIframe_{element_id}(img);
                }}
            }}
        }}
        // Ensure first image is visible when page loads
        document.addEventListener('DOMContentLoaded', function() {{
              const container = document.getElementById('{element_id}');
              if (container && container.getClientRects().length > 0) {{
                 switchImage_{element_id}(0);
              }}
        }});
        </script>
        """

        return html


@dataclass
class Metric:
    """A single renderable metric: one set of table/histogram/histogram-per-class panels.

    ``reference`` and each entry in ``compares`` are ``(PointCloud, attribute)`` pairs.
    The attribute name may differ per entry (and ``reference`` may be ``None``), which is
    what lets one ``Metric`` represent direction-asymmetric distances (e.g. c2c forward/
    backward) using the same rendering path as plain, symmetric attributes.
    """

    key: str
    display_name: str
    reference: tuple | None = None
    compares: list[tuple] = field(default_factory=list)
    skip_summary: bool = False
    skip_per_class: bool = False


def _attribute_metrics(
    reference_pc,
    compare_pcs: list,
    attribute_list: list,
    class_distribution: bool,
    return_number_metrics: bool,
) -> list[Metric]:
    """Build one Metric per attribute in ``attribute_list``."""
    metrics = []
    for attribute in attribute_list:
        skip_summary = (class_distribution and attribute == "classification") or (
            return_number_metrics
            and attribute in ["return_number", "number_of_returns"]
        )
        metrics.append(
            Metric(
                key=attribute,
                display_name=attribute,
                reference=(reference_pc, attribute),
                compares=[(pc, attribute) for pc in compare_pcs],
                skip_summary=skip_summary,
                skip_per_class=(attribute == "classification"),
            )
        )
    return metrics


def _distance_metrics(
    reference_pc,
    compare_pcs: list,
    distance_list: list,
) -> list[Metric]:
    """Build forward/backward Metrics per distance category (e.g. "c2c") in ``distance_list``.

    Distances have no true reference-side counterpart: forward distances
    (reference to compare) all live on ``reference_pc`` under one key per compare
    cloud, and backward distances (compare to reference) live on each compare
    cloud under one shared key. Both are modelled as ``reference=None`` metrics
    whose ``compares`` list carries the asymmetry via differing ``(pc, attribute)``
    pairs.
    """
    if not distance_list:
        return []
    categories = sorted({d.split("_")[0] for d in distance_list})
    metrics = []
    for category in categories:
        metrics.append(
            Metric(
                key=f"{category}_to-compare",
                display_name=f"{category} (reference to compare)",
                compares=[
                    (reference_pc, f"{category}_to-compare-{i}")
                    for i in range(len(compare_pcs))
                ],
            )
        )
        metrics.append(
            Metric(
                key=f"{category}_to-reference",
                display_name=f"{category} (compare to reference)",
                compares=[
                    (compare_pc, f"{category}_to-reference")
                    for compare_pc in compare_pcs
                ],
            )
        )
    return metrics


def _summary_stats_table(metric: Metric):
    """Build a summary-statistics table for a metric's reference (if any) and compare series."""
    from pcfingerprint import stats

    stats_reference = None
    if metric.reference is not None:
        ref_pc, ref_attr = metric.reference
        stats_reference = stats.compute_statistics(
            pcfingerprint.pcfingerprint.get_attribute_data(ref_pc, ref_attr)
        )
    stats_compares = [
        stats.compute_statistics(
            pcfingerprint.pcfingerprint.get_attribute_data(pc, attr)
        )
        for pc, attr in metric.compares
    ]

    all_stats = (
        [stats_reference] if stats_reference is not None else []
    ) + stats_compares
    keys = list(all_stats[0].keys())

    def _is_null(value):
        try:
            return np.isnan(value)
        except TypeError:
            return pd.isnull(value)

    keys_to_remove = [k for k in keys if all(_is_null(s[k]) for s in all_stats)]
    keys = [k for k in keys if k not in keys_to_remove]

    df = pd.DataFrame({"Statistics": keys})
    if stats_reference is not None:
        df["Reference"] = [stats_reference[k] for k in keys]
        df["Reference"] = df["Reference"].astype(object)
    for i, s_stats in enumerate(stats_compares):
        col = f"Comparison {i + 1}"
        df[col] = [s_stats[k] for k in keys]
        df[col] = df[col].astype(object)

    # row formatting: limit to four decimal places (mode stays as-is)
    for k in keys:
        if "mode" not in k:
            df.loc[df["Statistics"] == k, df.columns != "Statistics"] = df.loc[
                df["Statistics"] == k, df.columns != "Statistics"
            ].map(lambda x: f"{x:.4f}")

    return DataFrameTable(df, caption=f"Summary statistics for {metric.display_name}")


def _histogram_content(report, metric: Metric, overlay: bool, vis_backend: str):
    """Build a FigureSelection with frequency/density histogram panels for a metric."""
    if vis_backend == "matplotlib":
        from pcfingerprint.vis_matplotlib import plot_histogram

        ext = "png"
    elif vis_backend == "plotly":
        from pcfingerprint.vis_plotly import plot_histogram

        ext = "html"
    else:
        raise NotImplementedError(
            f"Visualization backend {vis_backend} not supported for histogram plot."
        )

    figs = []
    for norm, norm_str in ((False, "frequency"), (True, "density")):
        fig = plot_histogram(
            series=metric.compares,
            reference=metric.reference,
            xlabel=metric.display_name,
            overlay=overlay,
            normalize=norm,
        )
        fig_file = (
            report.target_folder
            / IMG_FOLDER
            / f"{metric.key}_histogram_{norm_str}.{ext}"
        )
        if vis_backend == "matplotlib":
            fig.savefig(fig_file, format="png")
            plt.close(fig)
        else:
            fig.write_html(fig_file, config={"responsive": True})
        figs.append((norm_str, fig_file))

    return FigureSelection(
        figure_paths=[p.relative_to(report.target_folder) for _, p in figs],
        figure_labels=[label for label, _ in figs],
        render_mode="image" if vis_backend == "matplotlib" else "iframe",
        caption=f"Histogram of {metric.display_name}",
    )


def _histogram_per_class_content(
    report, metric: Metric, vis_backend: str, class_mapping: dict | None
):
    """Build a FigureSelection with one panel per class value for a metric (both norm variants)."""
    if vis_backend == "matplotlib":
        from pcfingerprint.vis_matplotlib import plot_histogram_per_class

        ext = "png"
    elif vis_backend == "plotly":
        from pcfingerprint.vis_plotly import plot_histogram_per_class

        ext = "html"
    else:
        raise NotImplementedError(
            f"Visualization backend {vis_backend} not supported for histogram plot."
        )

    figure_paths = []
    figure_labels = []
    for norm, norm_str in ((False, "frequency"), (True, "density")):
        figs, classes = plot_histogram_per_class(
            series=metric.compares,
            reference=metric.reference,
            xlabel=metric.display_name,
            normalize=norm,
        )
        for fig, class_val in zip(figs, classes):
            if class_mapping is not None and class_val in class_mapping:
                class_label = class_mapping[class_val]
            else:
                class_label = str(class_val)
            fig_file = (
                report.target_folder
                / IMG_FOLDER
                / f"{metric.key}_histogram_class{class_label}_{norm_str}.{ext}"
            )
            if vis_backend == "matplotlib":
                fig.savefig(fig_file, format="png")
                plt.close(fig)
            else:
                fig.write_html(fig_file, config={"responsive": True})
            figure_paths.append(fig_file.relative_to(report.target_folder))
            figure_labels.append(f"class_{class_label}_{norm_str}")

    if len(figure_paths) == 0:
        return None

    return FigureSelection(
        figure_paths=figure_paths,
        figure_labels=figure_labels,
        render_mode="image" if vis_backend == "matplotlib" else "iframe",
        caption=f"Histogram of {metric.display_name} per class",
    )


def _render_metric(
    report,
    metric: Metric,
    *,
    summary_statistics: bool,
    visualizations: list,
    vis_backend: str,
    class_mapping: dict | None,
) -> dict:
    """Render one metric's requested content pieces (table / histogram / histogram-per-class)."""
    content = {}

    if summary_statistics and not metric.skip_summary:
        content["table"] = _summary_stats_table(metric)

    if "histogram" in visualizations or "histogram_overlay" in visualizations:
        overlay = "histogram_overlay" in visualizations
        content["hist"] = _histogram_content(report, metric, overlay, vis_backend)

    if "histogram_per_class" in visualizations and not metric.skip_per_class:
        hist_per_class = _histogram_per_class_content(
            report, metric, vis_backend, class_mapping
        )
        if hist_per_class is not None:
            content["hist_per_class"] = hist_per_class

    return content


def _add_metrics_to_section(
    report,
    section,
    metrics: list[Metric],
    *,
    summary_statistics: bool,
    visualizations: list,
    vis_backend: str,
    class_mapping: dict | None,
) -> bool:
    """Render ``metrics`` and attach non-empty table/histogram/per-class tabs to ``section``.

    :return: Whether any content was added to the section.
    """
    tab_tables = Tabs()
    tab_hists = Tabs()
    tab_hists_per_class = Tabs()
    content_tables = {}
    content_hists = {}
    content_hists_per_class = {}

    for metric in metrics:
        content = _render_metric(
            report,
            metric,
            summary_statistics=summary_statistics,
            visualizations=visualizations,
            vis_backend=vis_backend,
            class_mapping=class_mapping,
        )
        if "table" in content:
            content_tables[metric.key] = content["table"]
        if "hist" in content:
            content_hists[metric.key] = content["hist"]
        if "hist_per_class" in content:
            content_hists_per_class[metric.key] = content["hist_per_class"]

    if len(content_tables) > 0:
        tab_tables.add_tabs(content_tables)
        section.add_content({"tables": tab_tables})
    if len(content_hists) > 0:
        tab_hists.add_tabs(content_hists)
        section.add_content({"histograms": tab_hists})
    if len(content_hists_per_class) > 0:
        tab_hists_per_class.add_tabs(content_hists_per_class)
        section.add_content({"histograms_per_class": tab_hists_per_class})

    return bool(content_tables or content_hists or content_hists_per_class)


def generate_report(
    reference_pc,
    compare_pcs: list,
    output_folder: str | Path,
    attribute_list: list[str],
    distance_list: list[str] | None = None,
    summary_statistics: bool = True,
    outlier_statistics: bool = False,
    return_number_metrics: bool = False,
    class_distribution: bool = False,
    class_mapping: dict[int, str] | None = None,
    visualizations: list[str] | None = None,
    vis_backend: str = "matplotlib",
    preview: bool = True,
):
    """Generate an HTML report comparing reference and comparison point clouds.

    :param reference_pc: Reference point cloud object.
    :type reference_pc: PointCloud
    :param compare_pcs: Comparison point cloud objects to compare against ``reference_pc``.
    :type compare_pcs: list
    :param output_folder: Output folder for the generated HTML report.
    :type output_folder: str
    :param attribute_list: Point attributes to analyze and visualize.
    :type attribute_list: list[str]
    :param distance_list: Distance attributes to analyze and visualize, e.g. "c2c"
    :type distance_list: list[str]
    :param summary_statistics: Whether to include summary statistics tables.
    :type summary_statistics: bool
    :param return_number_metrics: Whether to include return-number metrics.
    :type return_number_metrics: bool
    :param class_distribution: Whether to include class distribution metrics.
    :type class_distribution: bool
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
    if visualizations is None:
        visualizations = ["histogram"]

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if summary_statistics or return_number_metrics:
        from pcfingerprint import stats

    # Create report
    filename = "index.html"
    report = Report(
        target_folder=output_path,
        filename=filename,
        template="src/pcfingerprint/templates/report_template_fingerprint.html",
    )
    (report.target_folder / IMG_FOLDER).mkdir(parents=True, exist_ok=True)

    # Add info for the header and title page
    report.add_header_info(
        header="Point Cloud Fingerprint",
        title="Report",
        sub_title=f"Comparison between {reference_pc.source} and {len(compare_pcs)} comparison point clouds",
        sub_sub_title=f"Attributes: {', '.join(attribute_list)}",
    )

    block_content = {}

    section = Section(section_title="Overview of Point Cloud Files")
    # start with overview of files, as table

    # create a pandas dataframe from this
    df = pd.DataFrame(columns=["Dataset", "Filename"])
    df["Dataset"] = ["Reference"] + [
        f"Comparison {i + 1}" for i in range(len(compare_pcs))
    ]
    filenames: list[str | None] = []
    try:
        filenames.append(Path(reference_pc.source).as_posix())
    except TypeError:
        filenames.append(None)
    for compare_pc in compare_pcs:
        try:
            filenames.append(Path(compare_pc.source).as_posix())
        except TypeError:
            filenames.append(None)
    df["Filename"] = filenames
    table = DataFrameTable(df, caption="Point Cloud Files Used in Comparison")
    section.add_content({"file_table": table})
    block_content["section_overview"] = section

    # section for 3D scatter plot
    if "3d_scatter" in visualizations and len(attribute_list) > 0:
        section = Section(section_title="3D Scatter Plot Comparison")
        scatter_tabs = Tabs()
        if len(attribute_list) == 0:
            plot_attribute_list = ["z"]
        else:
            plot_attribute_list = attribute_list
        tab_content_scatter = {}

        # generate 3d scatter plot
        if vis_backend == "matplotlib":
            # if using matplotlib, we generate a static PNG image file and reference it from the HTML report
            from pcfingerprint.vis_matplotlib import (
                plot_3d_scatter as plot_3d_scatter_mpl,
            )

            for attribute in plot_attribute_list:
                fig = plot_3d_scatter_mpl(
                    reference_pc, compare_pcs, attribute=attribute
                )
                # save the figure to a file in the report folder and add it to the report
                img_file = (
                    report.target_folder
                    / IMG_FOLDER
                    / Path(f"3d_scatter_{attribute}.png")
                )
                fig.savefig(img_file, format="png")
                plt.close(fig)
                tab_content_scatter[attribute] = MatplotlibFigure(
                    image_path=img_file.relative_to(report.target_folder),
                    caption=f"3D scatter plot colored by {attribute}",
                )
        elif vis_backend == "plotly":
            from pcfingerprint.vis_plotly import (
                plot_3d_scatter as plot_3d_scatter_plotly,
            )
            from pcfingerprint.vis_plotly import (
                write_html_with_linked_3d_camera,
            )

            for attribute in plot_attribute_list:
                fig = plot_3d_scatter_plotly(
                    reference_pc, compare_pcs, attribute=attribute
                )
                fig_file = (
                    report.target_folder
                    / IMG_FOLDER
                    / Path(f"3d_scatter_{attribute}.html")
                )
                write_html_with_linked_3d_camera(fig, fig_file)
                tab_content_scatter[attribute] = IFrameFigure(
                    html_path=fig_file.relative_to(report.target_folder),
                    caption=f"3D scatter plot colored by {attribute}",
                )
        else:
            # other backends not supported yet
            raise NotImplementedError(
                f"Visualization backend {vis_backend} not supported for 3D scatter plot."
            )

        if len(tab_content_scatter) > 0:
            scatter_tabs.add_tabs(tab_content_scatter)
            section.add_content({"3d_scatter_tabs": scatter_tabs})
        block_content["section_3d_scatter"] = section

    # Section for attribute comparisons
    section = Section(section_title="Attribute Comparisons")
    attribute_metrics = _attribute_metrics(
        reference_pc,
        compare_pcs,
        attribute_list,
        class_distribution,
        return_number_metrics,
    )
    has_content = _add_metrics_to_section(
        report,
        section,
        attribute_metrics,
        summary_statistics=summary_statistics,
        visualizations=visualizations,
        vis_backend=vis_backend,
        class_mapping=class_mapping,
    )
    if has_content:
        block_content["section_attribute_comparisons"] = section

    if outlier_statistics:
        section = Section(section_title="Outlier Statistics")
        logger.info("Computing outlier statistics...")
        # compute outlier statistics
        if len(reference_pc.sor.keys()) == 0 or any(
            len(compare_pc.sor.keys()) == 0 for compare_pc in compare_pcs
        ):
            logger.warning(
                "Outlier statistics requested but SOR outlier flags not computed for all point clouds. Skipping outlier statistics."
            )
        else:
            if len(reference_pc.sor.keys()) > 1:
                logger.warning("Multiple SOR outlier flags found. Using the first one.")
            outliers_reference = reference_pc.sor[
                next(iter(reference_pc.sor.keys()))
            ].astype(bool)
            outliers_compare = [
                compare_pc.sor[next(iter(compare_pc.sor.keys()))].astype(bool)
                for compare_pc in compare_pcs
            ]
            num_outliers_reference = np.sum(outliers_reference)
            num_outliers_compare = [np.sum(outliers) for outliers in outliers_compare]

            # create a pandas dataframe which contains number of inliers and outliers and percentage of outliers in each point cloud
            outlier_df = pd.DataFrame(
                {
                    "Point Cloud": ["Reference"]
                    + [f"Comparison {i + 1}" for i in range(len(compare_pcs))],
                    "Number of Inliers": [np.sum(~outliers_reference)]
                    + [np.sum(~outliers) for outliers in outliers_compare],
                    "Number of Outliers": [num_outliers_reference]
                    + [*num_outliers_compare],
                    "Percentage of Outliers": [
                        num_outliers_reference / reference_pc.coords.shape[0]
                    ]
                    + [
                        num_outliers / compare_pc.coords.shape[0]
                        for num_outliers, compare_pc in zip(
                            num_outliers_compare, compare_pcs
                        )
                    ],
                }
            )
            # add a thousand separator to the number of inliers and outliers columns
            outlier_df["Number of Inliers"] = outlier_df["Number of Inliers"].map(
                lambda x: f"{int(x):,}"
            )
            outlier_df["Number of Outliers"] = outlier_df["Number of Outliers"].map(
                lambda x: f"{int(x):,}"
            )
            # format the percentage of outliers as a percentage with 2 decimal places
            outlier_df["Percentage of Outliers"] = outlier_df[
                "Percentage of Outliers"
            ].map(lambda x: f"{x:.2%}")
            table = DataFrameTable(
                outlier_df,
                caption="Statistical Outlier Removal (SOR) Outlier Statistics",
            )
            section.add_content({"outlier_table": table})
            block_content["section_outlier_statistics"] = section

    if class_distribution:
        section = Section(section_title="Class Distribution Metrics")
        logger.info("Computing class distribution metrics...")
        # determine all unique classes (also non-overlapping)
        unique_classes = np.unique(
            np.concatenate(
                [np.unique(reference_pc.classification)]
                + [np.unique(pc.classification) for pc in compare_pcs]
            )
        )

        # compute class distribution metrics
        metrics_reference = stats.compute_class_distribution_metrics(
            reference_pc.classification,
            target_classes=unique_classes,
            class_mapping=class_mapping,
        )
        metrics_compare = [
            stats.compute_class_distribution_metrics(
                compare_pc.classification,
                target_classes=unique_classes,
                class_mapping=class_mapping,
            )
            for compare_pc in compare_pcs
        ]

        # create a pandas dataframe from this
        keys = metrics_reference.keys()
        df = pd.DataFrame(
            {
                "Metrics": keys,
                "Reference": [metrics_reference[k] for k in keys],
            }
        )
        df["Reference"] = df["Reference"].astype(object)
        # add a column for each comparison pc
        for i, s_metrics in enumerate(metrics_compare):
            df[f"Comparison {i + 1}"] = [s_metrics[k] for k in keys]
            df[f"Comparison {i + 1}"] = df[f"Comparison {i + 1}"].astype(object)
        # row formatting
        for k in keys:
            if "fraction" in k:
                df.loc[df["Metrics"] == k, df.columns != "Metrics"] = df.loc[
                    df["Metrics"] == k, df.columns != "Metrics"
                ].map(lambda x: f"{x:.3%}")
            elif "number" in k or k == "total points":
                # as integer with comma as thousand separator
                df.loc[df["Metrics"] == k, df.columns != "Metrics"] = df.loc[
                    df["Metrics"] == k, df.columns != "Metrics"
                ].map(lambda x: f"{int(x):,}")
        table = DataFrameTable(df, caption="Class Distribution Metrics")

        section.add_content({"class_distribution_table": table})
        block_content["section_class_distribution_metrics"] = section

    # section for distance metrics
    if distance_list is not None and len(distance_list) > 0:
        section = Section(section_title="Distance Metrics")

        tab_categories = {d.split("_")[0] for d in distance_list}
        if "3d_scatter_distances" in visualizations:
            scatter_tabs_ref = Tabs()
            scatter_tabs_comp = Tabs()
            tab_content_scatter_ref = {}
            tab_content_scatter_comp = {}
            if vis_backend == "matplotlib":
                for dist_category in tab_categories:
                    from pcfingerprint.vis_matplotlib import (
                        plot_3d_scatter_dist_comp,
                        plot_3d_scatter_dist_ref,
                    )

                    dist_ref = [
                        d
                        for d in distance_list
                        if dist_category in d and "to-compare" in d
                    ]
                    dist_comp = f"{dist_category}_to-reference"

                    fig_ref = plot_3d_scatter_dist_ref(reference_pc, dist_ref, vmax=0.1)
                    fig_comp = plot_3d_scatter_dist_comp(
                        compare_pcs, dist_comp, vmax=0.1
                    )
                    ref_fig_file = (
                        report.target_folder
                        / IMG_FOLDER
                        / Path(f"3d_scatter_{dist_comp.split('_')[0]}_ref.png")
                    )
                    comp_fig_file = (
                        report.target_folder
                        / IMG_FOLDER
                        / Path(f"3d_scatter_{dist_comp.split('_')[0]}_comp.png")
                    )
                    fig_ref.savefig(ref_fig_file, format="png")
                    fig_comp.savefig(comp_fig_file, format="png")
                    plt.close(fig_ref)
                    plt.close(fig_comp)

                    tab_content_scatter_ref[dist_category] = MatplotlibFigure(
                        image_path=ref_fig_file.relative_to(report.target_folder),
                        caption=f"3D scatter plot of {dist_category} distances from reference to comparison point clouds",
                    )
                    tab_content_scatter_comp[dist_category] = MatplotlibFigure(
                        image_path=comp_fig_file.relative_to(report.target_folder),
                        caption=f"3D scatter plot of {dist_category} distances from comparison to reference point clouds",
                    )
            elif vis_backend == "plotly":
                for dist_category in tab_categories:
                    from pcfingerprint.vis_plotly import (
                        plot_3d_scatter_dist_comp,
                        plot_3d_scatter_dist_ref,
                        write_html_with_linked_3d_camera,
                    )

                    dist_ref = [
                        d
                        for d in distance_list
                        if dist_category in d and "to-compare" in d
                    ]
                    dist_comp = f"{dist_category}_to-reference"

                    fig_ref = plot_3d_scatter_dist_ref(reference_pc, dist_ref, vmax=0.1)
                    fig_comp = plot_3d_scatter_dist_comp(
                        compare_pcs, dist_comp, vmax=0.1
                    )
                    ref_fig_file = (
                        report.target_folder
                        / IMG_FOLDER
                        / Path(f"3d_scatter_{dist_comp.split('_')[0]}_ref.html")
                    )
                    comp_fig_file = (
                        report.target_folder
                        / IMG_FOLDER
                        / Path(f"3d_scatter_{dist_comp.split('_')[0]}_comp.html")
                    )
                    write_html_with_linked_3d_camera(fig_ref, ref_fig_file)
                    write_html_with_linked_3d_camera(fig_comp, comp_fig_file)

                    tab_content_scatter_ref[dist_category] = IFrameFigure(
                        html_path=ref_fig_file.relative_to(report.target_folder),
                        caption=f"3D scatter plot of {dist_category} distances from reference to comparison point clouds",
                    )
                    tab_content_scatter_comp[dist_category] = IFrameFigure(
                        html_path=comp_fig_file.relative_to(report.target_folder),
                        caption=f"3D scatter plot of {dist_category} distances from comparison to reference point clouds",
                    )
            else:
                raise NotImplementedError(
                    f"Visualization backend {vis_backend} not supported for distance metrics."
                )

            if len(tab_content_scatter_ref) > 0:
                scatter_tabs_ref.add_tabs(tab_content_scatter_ref)
                section.add_content({"distance_scatter_tabs_ref": scatter_tabs_ref})
            if len(tab_content_scatter_comp) > 0:
                scatter_tabs_comp.add_tabs(tab_content_scatter_comp)
                section.add_content({"distance_scatter_tabs_comp": scatter_tabs_comp})

        distance_metrics = _distance_metrics(reference_pc, compare_pcs, distance_list)
        _add_metrics_to_section(
            report,
            section,
            distance_metrics,
            summary_statistics=summary_statistics,
            visualizations=visualizations,
            vis_backend=vis_backend,
            class_mapping=class_mapping,
        )

        block_content["section_distance_metrics"] = section

    # section for return number metrics
    if return_number_metrics:
        section = Section(section_title="Return Number Metrics")

        logger.info("Computing return number metrics...")
        # compute return number statistics
        metrics_reference = stats.compute_return_number_metrics(
            reference_pc.return_number, reference_pc.number_of_returns
        )
        metrics_compare = [
            stats.compute_return_number_metrics(
                compare_pc.return_number, compare_pc.number_of_returns
            )
            for compare_pc in compare_pcs
        ]

        # create a pandas dataframe from this
        keys = metrics_reference.keys()
        df = pd.DataFrame(
            {
                "Statistics": keys,
                "Reference": [metrics_reference[k] for k in keys],
            }
        )
        df["Reference"] = df["Reference"].astype(object)
        # add a column for each comparison pc
        for i, s_metrics in enumerate(metrics_compare):
            df[f"Comparison {i + 1}"] = [s_metrics[k] for k in keys]
            df[f"Comparison {i + 1}"] = df[f"Comparison {i + 1}"].astype(object)
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
    inline_report_header_logo(report.target_folder / filename)

    if preview:
        serve_report_folder(output_folder=output_folder, open_browser=True)
