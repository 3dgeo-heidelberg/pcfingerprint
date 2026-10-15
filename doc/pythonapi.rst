User API reference
===================

This is the reference of the Python API for the `pcfingerprint` package.
It focuses on those aspects relevant to end users that are not interested in algorithm development.

PointCloud and PointCloudPair
-----------------------------

.. autoapiclass:: pcfingerprint.PointCloudPair
   :members:

.. autoapiclass:: pcfingerprint.PointCloud
   :members:

.. autoapifunction:: pcfingerprint.pcfingerprint.write_voxelized_point_clouds

Developer API reference
=======================

The following is a more complete documentation of the package for developers.

Geometric features
-------------------

.. autoapifunction:: pcfingerprint.geometric_features.compute_density_jakteristics

.. autoapifunction:: pcfingerprint.geometric_features.compute_density_pgeof

.. autoapifunction:: pcfingerprint.geometric_features.compute_geom_feat_jakteristics

.. autoapifunction:: pcfingerprint.geometric_features.compute_density_vapc

.. autoapifunction:: pcfingerprint.geometric_features.compute_geom_feat_vapc


Stats
-----

.. autoapifunction:: pcfingerprint.stats.compute_statistics

.. autoapifunction:: pcfingerprint.stats.compute_return_number_metrics

.. autoapifunction:: pcfingerprint.stats.compute_class_distribution_metrics

Report
------

.. autoapifunction:: pcfingerprint.report.generate_report

.. autoapiclass:: pcfingerprint.report.MatplotlibFigure
   :members:

.. autoapiclass:: pcfingerprint.report.IFrameFigure
   :members:

.. autoapiclass:: pcfingerprint.report.FigureSelection
   :members:

.. autoapiclass:: pcfingerprint.report.Metric
   :members:

.. autoapifunction:: pcfingerprint.report.serve_report_folder

.. autoapifunction:: pcfingerprint.report._attribute_metrics

.. autoapifunction:: pcfingerprint.report._distance_metrics

.. autoapifunction:: pcfingerprint.report._summary_stats_table

.. autoapifunction:: pcfingerprint.report._histogram_content

.. autoapifunction:: pcfingerprint.report._histogram_per_class_content

.. autoapifunction:: pcfingerprint.report._render_metric

.. autoapifunction:: pcfingerprint.report._add_metrics_to_section


Visualization
-------------

.. autoapifunction:: pcfingerprint.vis_common.prepare_attribute_data

.. autoapifunction:: pcfingerprint.vis_common.prepare_histogram_data_per_class

.. autoapifunction:: pcfingerprint.vis_common.plot_grid_layout

.. autoapifunction:: pcfingerprint.vis_common.prepare_distance_scatter_ref_data

.. autoapifunction:: pcfingerprint.vis_common.prepare_distance_scatter_comp_data

.. autoapifunction:: pcfingerprint.vis_common.histogram_bin_edges

.. autoapifunction:: pcfingerprint.vis_common.classification_color_mapping

.. autoapifunction:: pcfingerprint.vis_matplotlib.plot_3d_scatter

.. autoapifunction:: pcfingerprint.vis_matplotlib.plot_3d_scatter_dist_ref

.. autoapifunction:: pcfingerprint.vis_matplotlib.plot_3d_scatter_dist_comp

.. autoapifunction:: pcfingerprint.vis_matplotlib.plot_histogram

.. autoapifunction:: pcfingerprint.vis_matplotlib.plot_histogram_per_class

.. autoapifunction:: pcfingerprint.vis_matplotlib.style_axis

.. autoapifunction:: pcfingerprint.vis_plotly.plot_3d_scatter

.. autoapifunction:: pcfingerprint.vis_plotly.plot_3d_scatter_dist_ref

.. autoapifunction:: pcfingerprint.vis_plotly.plot_3d_scatter_dist_comp

.. autoapifunction:: pcfingerprint.vis_plotly.plot_histogram

.. autoapifunction:: pcfingerprint.vis_plotly.plot_histogram_per_class

.. autoapifunction:: pcfingerprint.vis_plotly.write_html_with_linked_3d_camera

.. autoapifunction:: pcfingerprint.vis_plotly._histogram_bar_trace


Utils
-----

.. autoapifunction:: pcfingerprint.utils.shift_to_zero

.. autoapifunction:: pcfingerprint.utils.standardize

.. autoapifunction:: pcfingerprint.utils.scale_robust

.. autoapifunction:: pcfingerprint.utils._build_structured_additional_dimensions

.. autoapifunction:: pcfingerprint.utils.match_representative_points_to_indices

.. autoapifunction:: pcfingerprint.utils.point_cloud_to_vapc

.. autoapifunction:: pcfingerprint.utils.subsample_points

.. autoapifunction:: pcfingerprint.utils._validate_classes


Chunking
--------

.. autoapifunction:: pcfingerprint.chunking.generate_chunk_tiles

.. autoapifunction:: pcfingerprint.chunking.run_chunked_computation


Logging
-------

.. autoapifunction:: pcfingerprint.logger.create_default_logger

.. autoapifunction:: pcfingerprint.logger.set_pcfingerprint_logfile
