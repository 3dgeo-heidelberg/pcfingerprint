"""
Module for generating comparison reports between point clouds.
"""
from typing import Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numerous.html_report_generator import Report, Section, Subsection, Div, Tabs, Card, DataFrameTable
from numerous.html_report_generator.components.figure import Figure
from numerous.html_report_generator.abstract import ReportInterface
import pcfingerprint


def wrap_div(html, class_=None):
    if class_:
        return f"<div class='{class_}'>\n{html}\n</div>\n"
    else:
        return f'<div>\n{html}\n</div>\n'


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
        """
        The constructor of the MatplotlibFigure class
        Args:
            figure_obj: The matplotlib figure object
            caption(str): The caption text
            notes (List[str]): The notes to be added to the caption
            post_script: javascript code to be executed after the figure is rendered
        """
        super(MatplotlibFigure, self).__init__(caption, notes)
        self.image_name = image_name

    def _as_html_figure_content(self, report: ReportInterface):
        
        html = f'''
                <div style='display: inline-flex;'>
                    <img src="{self.image_name.name}" alt=Figure {report.figure_number} style="max-width: 100%"/>
                </div>
                '''
        return html




def generate_report(real_pc, sim_pc,
                    output_path: str,
                    attribute_list: list[str],
                    summary_statistics: bool = True,
                    visualizations: list[str] = ["histogram"],
                    vis_backend: str = "matplotlib"):
    """Generate an HTML report comparing real-world and simulated point clouds."""

    # Create report
    folder = Path(output_path).parent
    filename = Path(output_path).stem + ".html"
    report = Report(target_folder=folder, filename=filename, template="src/pc_fingerprint/templates/report_template_fingerprint.html")

    # Add info for the header and title page
    report.add_header_info(header='Point Cloud Fingerprint',
                           title='Point Cloud Fingerprint Report',
                           sub_sub_title=f'Comparison between {real_pc.source} and {sim_pc.source}',
                           sub_title=f'Attributes: {", ".join(attribute_list)}',
                           )

    block_content = {}

    # Add a section for each attribute
    for attribute in attribute_list:
        section = Section(section_title=f"Comparing {attribute}")

        # extract data for attribute
        real_data = pcfingerprint.pcfingerprint.get_attribute_data(real_pc, attribute)
        sim_data = pcfingerprint.pcfingerprint.get_attribute_data(sim_pc, attribute)

        section_content = {}

        if summary_statistics:
            # compute summary statistics
            # TODO: Probably put in a separate function eventually
            real_mean = np.nanmean(real_data)
            real_median = np.nanmedian(real_data)
            real_std = np.nanstd(real_data)
            sim_mean = np.nanmean(sim_data)
            sim_median = np.nanmedian(sim_data)
            sim_std = np.nanstd(sim_data)

            # create a table
            table = DataFrameTable(pd.DataFrame({
                'Statistics': ['Mean', 'Median', 'Std Dev'],
                'Real': [real_mean, real_median, real_std],
                'Simulated': [sim_mean, sim_median, sim_std]
            }), caption="Summary statistics")

            section_content['summary_stat_table'] = table
        
        if "histogram" in visualizations:
            # generate histogram plot
            if vis_backend == "matplotlib":
                # if using matplotlib, we generate static image and insert as base64 png
                from pcfingerprint.vis_matplotlib import plot_histogram
                fig = plot_histogram(real_pc, sim_pc, attribute)
                # convert fig to base64
                img_file = report.target_folder / Path(f"{attribute}_histogram.png")
                fig.savefig(img_file, format='png')
                plt.close(fig)
                figure = MatplotlibFigure(image_name=img_file, caption=f"Histogram of {attribute}")
                section_content['histogram'] = figure
            
        #Add the figure and table to the section
        section.add_content(section_content)

        block_content[f'section_{attribute}'] = section
    
    # Add the sections to the report
    report.add_blocks(block_content)

    # Save the report - creates the html output file
    report.save()
