from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import jakteristics
import laspy
import numpy as np
import py4dgeo
from pgeof import EFeatureID

from pcfingerprint.chunking import run_chunked_computation
from pcfingerprint.geometric_features import (
    compute_density_jakteristics,
    compute_density_pgeof,
    compute_geom_feat_jakteristics,
    compute_geom_feat_pgeof,
    compute_geom_feat_vapc,
)
from pcfingerprint.logger import logger_context
from pcfingerprint.report import generate_report
from pcfingerprint.utils import (
    match_representative_points_to_indices,
    scale_robust,
    shift_to_zero,
    standardize,
)

logger = logging.getLogger("pcfingerprint")


class PointCloud:
    def __init__(
        self,
        coords: np.ndarray | None = None,
        gps_time: np.ndarray | None = None,
        return_number: np.ndarray | None = None,
        number_of_returns: np.ndarray | None = None,
        intensity: np.ndarray | None = None,
        classification: np.ndarray | None = None,
    ):
        self.coords: np.ndarray | None = coords
        """Nx3 array of point coordinates."""
        self.source: Path | str | None = None
        """Path of the file the point cloud was loaded from, if any."""
        self.gps_time: np.ndarray | None = gps_time
        """Float array of GPS timestamps."""
        self.return_number: np.ndarray | None = return_number
        """Int array of return numbers."""
        self.number_of_returns: np.ndarray | None = number_of_returns
        """Int array of the number of returns per pulse."""
        self.intensity: np.ndarray | None = intensity
        """Int array of intensity values."""
        self.classification: np.ndarray | None = classification
        """Int array of point classification codes."""
        self.geometric_features: dict[str, np.ndarray] = {}
        """Feature name to per-point float array."""
        self.voxel_features: dict[str, np.ndarray] = {}
        """Feature name to float array for voxel-based features."""
        self.c2c: dict[str, np.ndarray] = {}
        """Cloud-to-cloud distances, keyed by distance name (e.g. ``"c2c_to-reference"``)."""
        self.sor: dict[str, np.ndarray] = {}
        """Statistical outlier removal results, keyed by feature name."""

    def shift_z_to_zero(self):
        """Shift the z-coordinates so that the minimum z-value is zero."""
        if self.coords is not None:
            self.coords[:, 2] = shift_to_zero(self.coords[:, 2])
        else:
            raise ValueError("PointCloud coordinates are not set.")

    def shift_gpstime_to_zero(self):
        """Shift the gps_time so that the minimum gps_time is zero."""
        if self.gps_time is not None:
            self.gps_time = shift_to_zero(self.gps_time)
        else:
            raise ValueError("PointCloud gps_time is not set.")

    def robust_scale_intensity(self):
        """Perform robust scaling on intensity values."""
        if self.intensity is not None:
            self.intensity = scale_robust(self.intensity)
        else:
            raise ValueError("PointCloud intensity is not set.")

    def standardize_intensity(self):
        """Standardize intensity values (z-score normalization)."""
        if self.intensity is not None:
            self.intensity = standardize(self.intensity)
        else:
            raise ValueError("PointCloud intensity is not set.")

    def _selection_mask_and_index_map(self, indices):
        if isinstance(indices, np.ndarray) and indices.dtype == bool:
            if indices.shape[0] != self.coords.shape[0]:
                raise ValueError("Boolean mask length does not match point cloud size.")
            selected_indices = np.flatnonzero(indices)
        else:
            selected_indices = np.asarray(indices, dtype=int)

        point_mask = np.zeros(self.coords.shape[0], dtype=bool)
        point_mask[selected_indices] = True

        index_map = np.full(self.coords.shape[0], -1, dtype=int)
        index_map[selected_indices] = np.arange(selected_indices.shape[0])
        return point_mask, index_map

    def _mask_voxel_feature(self, feat_data, point_mask, index_map):
        if not isinstance(feat_data, tuple) or len(feat_data) < 2:
            return feat_data

        voxel_values = feat_data[0]
        voxel_coords = feat_data[1]
        if len(feat_data) > 2 and feat_data[2] is not None:
            rep_indices = np.asarray(feat_data[2], dtype=int)
        else:
            rep_indices = match_representative_points_to_indices(
                self.coords, voxel_coords
            )

        voxel_mask = point_mask[rep_indices]
        filtered_values = voxel_values[voxel_mask]
        filtered_coords = voxel_coords[voxel_mask]
        filtered_rep_indices = index_map[rep_indices[voxel_mask]]
        return (filtered_values, filtered_coords, filtered_rep_indices)

    def filter_by_class(self, classes: list[int]):
        """
        Filter the point cloud to specified classes using the "classification" attribute (if present).

        :param classes: List of class values to filter by.
        :type classes: list[int]

        :return: A new PointCloud object containing only points of the specified classes.
        :rtype: PointCloud
        """
        if classes is None or classes == []:
            raise ValueError("Please provide a list of classes to filter by.")
        if hasattr(self, "classification") and self.classification is not None:
            mask = np.isin(self.classification, classes)
            # check if mask is empty
            if not np.any(mask):
                warnings.warn(
                    f"No points found for classes {classes} in this point cloud.",
                    UserWarning,
                )
            return self.mask(mask)
        else:
            raise ValueError(
                "PointCloud does not have a 'classification' attribute to filter by."
            )

    def mask(self, indices):
        """
        Mask the point cloud using the given indices.

        :param indices: Boolean or integer indices for masking.
        :type indices: np.ndarray of bool or int

        :return: A new PointCloud object containing only the masked points.
        :rtype: PointCloud
        """
        if self.coords is not None:
            point_mask, index_map = self._selection_mask_and_index_map(indices)
            new_pc = PointCloud()
            new_pc.coords = self.coords[indices]
            # copy over all other attributes if they exist
            if self.source is not None:
                new_pc.source = self.source
            if self.gps_time is not None:
                new_pc.gps_time = self.gps_time[indices]
            if self.return_number is not None:
                new_pc.return_number = self.return_number[indices]
            if self.number_of_returns is not None:
                new_pc.number_of_returns = self.number_of_returns[indices]
            if self.intensity is not None:
                new_pc.intensity = self.intensity[indices]
            if self.classification is not None:
                new_pc.classification = self.classification[indices]
            # geometric features
            for feat_name, feat_data in self.geometric_features.items():
                warnings.warn(
                    f"Masking geometric feature {feat_name}. If using mask to subset points, consider recomputing geometric features",
                    UserWarning,
                )
                new_pc.geometric_features[feat_name] = feat_data[indices]
            # c2c distances
            for dist_name, dist_data in self.c2c.items():
                new_pc.c2c[dist_name] = dist_data[indices]
            # outlier features
            for feat_name, feat_data in self.sor.items():
                new_pc.sor[feat_name] = feat_data[indices]
            # voxel features
            for feat_name, feat_data in self.voxel_features.items():
                new_pc.voxel_features[feat_name] = self._mask_voxel_feature(
                    feat_data, point_mask, index_map
                )
            return new_pc
        else:
            raise ValueError("PointCloud coordinates are not set.")


class PointCloudPair:
    def __init__(
        self,
        shift_z: bool = False,
        shift_gps_time: bool = False,
        standardize_intensity: bool = False,
        robust_scale_intensity: bool = False,
        class_mapping: dict[int, str] | None = None,
    ):
        self.reference: PointCloud = PointCloud()
        """Reference PointCloud"""
        self.compare: list[PointCloud] = []
        """List of PointClouds (minimum 1) to compare to reference"""
        self.shift_z: bool = shift_z
        """Whether to shift the z-coordinates of all PointClouds so that the minimum z-value is zero"""
        self.shift_gps_time: bool = shift_gps_time
        """Whether to shift the gps_time of all PointClouds so that the minimum gps_time is zero"""
        self.standardize_intensity: bool = standardize_intensity
        """Whether to standardize intensity values of all PointClouds"""
        self.robust_scale_intensity: bool = robust_scale_intensity
        """Whether to perform robust scaling on intensity values of all PointClouds"""
        self.class_mapping: dict[int, str] | None = class_mapping
        """Dictionary to map integer class values to class names"""

    def drop_attributes(self):
        """
        Drop attributes that are not present in all point clouds in the pair to ensure comparability.
        """
        all_pcs = [self.reference] + self.compare
        for attr in [
            "gps_time",
            "return_number",
            "number_of_returns",
            "intensity",
        ]:
            if any(getattr(pc, attr) is None for pc in all_pcs):
                logger.info(
                    f"Dropping attribute '{attr}' as it is not present in all point clouds."
                )
                for pc in all_pcs:
                    setattr(pc, attr, None)

    def load_reference(self, filepath: str | Path):
        """
        Load the reference point cloud from a file. Supported formats are LAS/LAZ and XYZ.

        :param filepath: File path to load. Supported formats are LAS/LAZ and XYZ.
        :type filepath: str
        """
        logger.info(f"Loading reference point cloud from {filepath}...")
        # read based on file extension
        if Path(filepath).suffix in [".las", ".laz"]:
            self.reference = read_las_file(filepath)
        elif Path(filepath).suffix in [".txt", ".xyz", ".csv"]:
            self.reference = read_xyz_file(filepath)
        else:
            raise ValueError(
                f"File format {Path(filepath).suffix} not supported. Please provide a LAS/LAZ or XYZ file."
            )
        self.reference.source = Path(filepath)
        if self.shift_z:
            self.reference.shift_z_to_zero()
        if self.shift_gps_time:
            self.reference.shift_gpstime_to_zero()
        if self.robust_scale_intensity and self.standardize_intensity:
            raise ValueError(
                "Cannot apply both robust scaling and standardization to intensity."
            )
        if self.robust_scale_intensity:
            self.reference.robust_scale_intensity()
        if self.standardize_intensity:
            self.reference.standardize_intensity()
        # if comparison point cloud is already loaded, drop non-overlapping attributes
        if self.compare != []:
            self.drop_attributes()

    def load_compare(self, filepaths: Sequence[str | Path] | str | Path):
        """
        Load comparison point cloud(s) from file(s).
        Users can provide both a single file path (str or Path) and a list of file paths.

        :param filepaths: File path(s) to load. Can be a single file path or a sequence of file paths (str or Path). Supported formats are LAS/LAZ and XYZ.
        :type filepaths: Sequence[str | Path] | str | Path
        """
        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]
        for filepath in filepaths:
            logger.info(f"Loading comparison point cloud from {filepath}...")
            if Path(filepath).suffix in [".las", ".laz"]:
                self.compare.append(read_las_file(filepath))
            elif Path(filepath).suffix in [".txt", ".xyz", ".csv"]:
                self.compare.append(read_xyz_file(filepath))
            else:
                raise ValueError(
                    f"File format {Path(filepath).suffix} not supported. Please provide a LAS/LAZ or XYZ file."
                )
            self.compare[-1].source = Path(filepath)
        if self.shift_z:
            for pc in self.compare:
                pc.shift_z_to_zero()
        if self.shift_gps_time:
            for pc in self.compare:
                pc.shift_gpstime_to_zero()
        if self.robust_scale_intensity and self.standardize_intensity:
            raise ValueError(
                "Cannot apply both robust scaling and standardization to intensity."
            )
        if self.standardize_intensity:
            for pc in self.compare:
                pc.standardize_intensity()
        if self.robust_scale_intensity:
            for pc in self.compare:
                pc.robust_scale_intensity()
        # if reference world point cloud is already loaded, drop non-overlapping attributes
        if self.reference.coords is not None:
            self.drop_attributes()

    def compute_statistical_outlier_removal(
        self, k: int = 8, std_dev_multiplier: float = 1.0
    ):
        """
        Perform statistical outlier removal (SOR) on the reference and all comparison point clouds.
        Uses py4dgeo.sor.statistical_outlier_removal and adds a boolean array named by the parameters to the sor dictionary of each PointCloud.

        :param k: Number of nearest neighbors to consider for each point. Default is 8.
        :type k: int
        :param std_dev_multiplier: Standard deviation multiplier for determining outliers. Default is 1.0.
        :type std_dev_multiplier: float
        """
        if self.reference.coords is None or self.compare == []:
            raise ValueError(
                "Please load both reference and compare point clouds before computing statistical outlier removal."
            )
        # round std_dev_multiplier to 1 decimal place
        std_dev_multiplier = round(std_dev_multiplier, 1)
        # create epoch for reference
        epoch = py4dgeo.Epoch(self.reference.coords)
        _, inlier_outlier = py4dgeo.statistical_outlier_removal(
            epoch, k=k, std_dev_multiplier=std_dev_multiplier
        )
        self.reference.sor[f"sor_k{k}_std{std_dev_multiplier}"] = inlier_outlier
        for pc in self.compare:
            epoch = py4dgeo.Epoch(pc.coords)
            _, inlier_outlier = py4dgeo.statistical_outlier_removal(
                epoch, k=k, std_dev_multiplier=std_dev_multiplier
            )
            pc.sor[f"sor_k{k}_std{std_dev_multiplier}"] = inlier_outlier

    def compute_geometric_features(
        self,
        feature_names: list[str],
        radius: float = 2.0,
        max_knn: int = 50_000,
        backend: str = "pgeof",
        chunking: bool = False,
        max_points_per_chunk: int | None = None,
    ):
        """
        Compute geometric features for all point clouds in the pair

        :param feature_names: List of geometric feature names to compute. Supported features depend on the backend used.
        :type feature_names: list[str]
        :param radius: Search radius for geometric feature computation
        :type radius: float
        :param max_knn: Maximum number of neighbors to consider for geometric feature computation
        :type max_knn: int
        :param backend: Backend to use for geometric feature computation. Options are "jakteristics" and "pgeof". If the specified backend does not support certain features, it will automatically fall back to the other backend for those features.
        :type backend: str
        :param chunking: If True, compute features tile-by-tile using spatial chunking instead of on the whole point cloud at once. Use this for large point clouds where a single-pass computation would exceed available memory. Requires ``max_points_per_chunk`` to be set.
        :type chunking: bool
        :param max_points_per_chunk: Maximum number of interior points per spatial tile when ``chunking`` is enabled. Tiles are buffered by at least ``radius`` so results match an unchunked computation.
        :type max_points_per_chunk: int | None
        """
        # raise error if no point clouds loaded
        if self.reference.coords is None or self.compare == []:
            raise ValueError(
                "Please load both reference and compare point clouds before computing geometric features."
            )
        if chunking and max_points_per_chunk is None:
            raise ValueError(
                "Please provide max_points_per_chunk when chunking is enabled."
            )
        # Compute geometric features for all point clouds
        radius = np.round(radius, 3)
        requested_features = list(feature_names)

        density_requested = "density" in requested_features
        feature_names = [f for f in requested_features if f != "density"]

        def _compute(pc, compute_fn, **kwargs):
            if chunking:
                run_chunked_computation(
                    pc, compute_fn, max_points_per_chunk, buffer=radius, **kwargs
                )
            else:
                compute_fn(pc, **kwargs)

        if density_requested:
            if backend == "jakteristics":
                with logger_context(
                    f"Computing density for reference pc using jakteristics. Search radius: {radius}, max_knn: {max_knn}"
                ):
                    _compute(
                        self.reference,
                        compute_density_jakteristics,
                        radius=radius,
                        max_knn=max_knn,
                    )
                for compare_pc in self.compare:
                    name = (
                        Path(compare_pc.source).name
                        if compare_pc.source is not None
                        else "unknown source"
                    )
                    with logger_context(
                        f"Computing density for {name} using jakteristics. Search radius: {radius}, max_knn: {max_knn}"
                    ):
                        _compute(
                            compare_pc,
                            compute_density_jakteristics,
                            radius=radius,
                            max_knn=max_knn,
                        )
            elif backend == "pgeof":
                with logger_context(
                    f"Computing density for reference pc using pgeof. Search radius: {radius}, max_knn: {max_knn}"
                ):
                    _compute(
                        self.reference,
                        compute_density_pgeof,
                        radius=radius,
                        max_knn=max_knn,
                    )
                for compare_pc in self.compare:
                    name = (
                        Path(compare_pc.source).name
                        if compare_pc.source is not None
                        else "unknown source"
                    )
                    with logger_context(
                        f"Computing density for {name} using pgeof. Search radius: {radius}, max_knn: {max_knn}"
                    ):
                        _compute(
                            compare_pc,
                            compute_density_pgeof,
                            radius=radius,
                            max_knn=max_knn,
                        )
            else:
                raise ValueError(f"Backend {backend} not supported.")

        if feature_names == []:
            return  # all requested features already computed

        # check if all feature names available for the backend, if not, try fallback on other backend before raising error
        if backend == "pgeof":
            pgeof_features_lower = [f.lower() for f in EFeatureID._member_names_]
            pgeof_features = [f for f in feature_names if f in pgeof_features_lower]
            jakteristics_features = [
                f
                for f in feature_names
                if (
                    f in jakteristics.constants.FEATURE_NAMES
                    and f not in pgeof_features
                )
            ]
            if jakteristics_features != []:
                warnings.warn(
                    f"Features {jakteristics_features} not available in pgeof. Computing them using jakteristics as fallback.",
                    UserWarning,
                )
        elif backend == "jakteristics":
            jakteristics_features = [
                f for f in feature_names if f in jakteristics.constants.FEATURE_NAMES
            ]
            pgeof_features = [
                f
                for f in feature_names
                if (f in EFeatureID._member_names_ and f not in jakteristics_features)
            ]
            if pgeof_features != []:
                warnings.warn(
                    f"Features {pgeof_features} not available in jakteristics. Computing them using pgeof as fallback.",
                    UserWarning,
                )
        else:
            raise ValueError(f"Backend {backend} not supported.")
        if len(jakteristics_features) > 0:
            with logger_context(
                f"Computing geometric features for reference pc using jakteristics. {jakteristics_features}, search radius: {radius}, max_knn: {max_knn}"
            ):
                _compute(
                    self.reference,
                    compute_geom_feat_jakteristics,
                    feature_names=jakteristics_features,
                    radius=radius,
                    max_knn=max_knn,
                )
            for compare_pc in self.compare:
                name = (
                    Path(compare_pc.source).name
                    if compare_pc.source is not None
                    else "unknown source"
                )
                with logger_context(
                    f"Computing geometric features for {name} using jakteristics. {jakteristics_features}, search radius: {radius}, max_knn: {max_knn}"
                ):
                    _compute(
                        compare_pc,
                        compute_geom_feat_jakteristics,
                        feature_names=jakteristics_features,
                        radius=radius,
                        max_knn=max_knn,
                    )
        if len(pgeof_features) > 0:
            with logger_context(
                f"Computing geometric features for reference pc using pgeof. {pgeof_features}, search radius: {radius}, max_knn: {max_knn}"
            ):
                _compute(
                    self.reference,
                    compute_geom_feat_pgeof,
                    feature_names=pgeof_features,
                    radius=radius,
                    max_knn=max_knn,
                )
            for compare_pc in self.compare:
                name = (
                    Path(compare_pc.source).name
                    if compare_pc.source is not None
                    else "unknown source"
                )
                with logger_context(
                    f"Computing geometric features for {name} using pgeof. {pgeof_features}, search radius: {radius}, max_knn: {max_knn}"
                ):
                    _compute(
                        compare_pc,
                        compute_geom_feat_pgeof,
                        feature_names=pgeof_features,
                        radius=radius,
                        max_knn=max_knn,
                    )

    def compute_voxel_features(
        self,
        voxel_size: float,
        voxel_origin=None,
        feature_names: list[str] | None = None,
    ):
        """
        Compute voxel-based geometric features for all point clouds in the pair using VAPC.

        :param voxel_size: Voxel edge length used for voxel-based feature computation.
        :type voxel_size: float
        :param voxel_origin: Optional voxel grid origin passed to the VAPC backend. If None, [0.0, 0.0, 0.0] is used.
        :type voxel_origin: array-like, optional
        :param feature_names: List of VAPC features to compute. Supported features (py4dgeo v1.1) are "count", "density", "covariance", "eigenvalues", "linearity", "planarity", "sphericity", "sum_of_eigenvalues","omnivariance", "eigenentropy", "anisotropy", "surface_variation". If empty, all available features will be computed.
        :type feature_names: list[str] | None

        :return: Tuple containing (VAPC for reference point cloud, list of VAPCs for compare point clouds)
        :rtype: tuple[py4dgeo.Vapc, list[py4dgeo.Vapc]]
        """
        # raise error if no point clouds loaded
        if self.reference.coords is None or self.compare == []:
            raise ValueError(
                "Please load both reference and compare point clouds before computing voxel features."
            )
        if feature_names is None:
            feature_names = []
        vapc_list = []
        for pc in [self.reference] + self.compare:
            name = Path(pc.source).name if pc.source is not None else "unknown source"
            with logger_context(
                f"Computing voxel features for {name} using VAPC. {feature_names}, voxel size: {voxel_size}"
            ):
                vapc = compute_geom_feat_vapc(
                    pc, voxel_size, voxel_origin, feature_names
                )
            vapc_list.append(vapc)

        return (
            vapc_list[0],
            vapc_list[1:],
        )  # return reference VAPC and list of compare VAPCs

    def compute_c2c_distances(self, max_distance: float = np.inf):
        """
        Compute cloud-to-cloud distances between the reference and each comparison point cloud (both directions).
        The distances are stored in
        - the c2c attribute of the reference point cloud (dictionary with arrays of distances to each comparison point cloud)
        - the c2c attribute of each comparison point cloud (dictionary with a single array of distances to the reference point cloud)

        :param max_distance: Optional maximum distance threshold. Distances greater than this value will be set to NaN. Default: np.inf, i.e., no threshold is applied.
        :type max_distance: float, optional
        """

        # create epochs for the point clouds
        reference_epoch = py4dgeo.Epoch(cloud=self.reference.coords)

        for i, compare_pc in enumerate(self.compare):
            compare_epoch = py4dgeo.Epoch(cloud=compare_pc.coords)
            # forward distances: reference to compare
            c2c = py4dgeo.C2C(
                epochs=(reference_epoch, compare_epoch), max_distance=max_distance
            )
            self.reference.c2c[f"c2c_to-compare-{i}"] = c2c.run()
            # backward distances: compare to reference
            c2c_rev = py4dgeo.C2C(
                epochs=(compare_epoch, reference_epoch), max_distance=max_distance
            )
            compare_pc.c2c["c2c_to-reference"] = c2c_rev.run()

    def filter_by_class(self, classes: list[int]):
        """
        Filter the point clouds in the PointCloudPair to specified classes using the "classification" attribute (if present).

        :param classes: List of class values to filter by.
        :type classes: list[int]

        :return: A new PointCloudPair object containing only points of the specified classes.
        :rtype: PointCloudPair
        """
        # raise error if no point clouds loaded
        if self.reference.coords is None or self.compare == []:
            raise ValueError(
                "Please load both reference and compare point clouds before filtering by class."
            )
        if self.reference.classification is None or any(
            compare_pc.classification is None for compare_pc in self.compare
        ):
            raise ValueError(
                "Cannot filter by class if one of the point clouds does not have 'classification' attribute."
            )
        logger.info(f"Filtering point clouds by classes {classes}...")
        new_pair = PointCloudPair(
            shift_z=self.shift_z,
            shift_gps_time=self.shift_gps_time,
            standardize_intensity=self.standardize_intensity,
            robust_scale_intensity=self.robust_scale_intensity,
        )
        new_pair.reference = self.reference.filter_by_class(classes)
        new_pair.compare = [
            compare_pc.filter_by_class(classes) for compare_pc in self.compare
        ]

        return new_pair

    def _collect_all_attributes(self, compare) -> list[str]:
        """
        Collect all available attributes from the reference and compare point clouds to use
        for report generation. Returns a deterministic, deduplicated list of attribute names.

        :param compare: list of comparison PointCloud objects
        :type compare: list[PointCloud]
        :return: Ordered list of attribute names available in the point clouds
        :rtype: list[str]
        """
        attributes: list[str] = []

        def add_unique(name: str):
            if name not in attributes:
                attributes.append(name)

        # start with coordinates)
        for coord in ("x", "y", "z"):
            add_unique(coord)

        standard_attributes = [
            "gps_time",
            "return_number",
            "number_of_returns",
            "intensity",
            "classification",
        ]
        for pc in [self.reference] + compare:
            for attr in standard_attributes:
                if getattr(pc, attr, None) is not None:
                    add_unique(attr)
            # sort feature keys to ensure consistent ordering across runs
            for attr in sorted(pc.geometric_features.keys()):
                add_unique(attr)
            for attr in sorted(pc.voxel_features.keys()):
                add_unique(attr)
            for attr in sorted(pc.c2c.keys()):
                add_unique(attr)

        return attributes

    def _expand_attribute_names(self, compare, attribute_list: list[str]) -> list[str]:
        """
        Expand attribute names to match available features. For each attribute in the list:
        - If an exact match exists in geometric_features or standard attributes, keep it as-is
        - If no exact match exists, expand to features whose base name matches exactly, where the
           base name is the part of the feature name before the final underscore

        Example: "linearity" expands to ["linearity_r0.1", "linearity_r0.2", etc.] if those exist.

        :param attribute_list: List of attribute names to expand
        :type attribute_list: list[str]
        :return: Expanded list of attribute names
        :rtype: list[str]
        """
        # Collect all available features from reference and all comparison point clouds
        available_features = self._collect_all_attributes(compare)

        expanded_attributes = []
        seen = set()
        for attr in attribute_list:
            attr_lower = attr.lower()

            # Check for exact match first
            exact_matches = [f for f in available_features if f.lower() == attr_lower]

            if exact_matches:
                # Use the original case from available features
                expanded_attributes.append(exact_matches[0])
                seen.add(exact_matches[0])
            else:
                # Find all features that contain this attribute name
                partial_matches = [
                    f
                    for f in available_features
                    if attr_lower == f.lower().rsplit("_", 1)[0]
                ]
                if partial_matches:
                    # Sort to ensure consistent ordering (e.g., linearity_r0.1, linearity_r0.2)
                    partial_matches_sorted = sorted(partial_matches)
                    for f in partial_matches_sorted:
                        # Avoid duplicates in case multiple attributes match the same feature
                        if f not in seen:
                            expanded_attributes.append(f)
                            seen.add(f)
                    if len(partial_matches_sorted) > 1:
                        logger.info(
                            f"Attribute '{attr}' matched multiple features: {partial_matches_sorted}. Adding all matches to the report."
                        )
                else:
                    # No match found - add as-is and let get_attribute_data raise the error
                    expanded_attributes.append(attr)

        return expanded_attributes

    def generate_report(
        self,
        output_folder: str | Path,
        attribute_list: list[str] | None,
        distance_list: list[str] | None = None,
        summary_statistics: bool = True,
        class_distribution: bool = False,
        outlier_statistics: bool = False,
        return_number_metrics: bool = False,
        visualizations: list[str] | None = None,
        vis_backend: str = "matplotlib",
        preview: bool = True,
    ):
        if visualizations is None:
            visualizations = ["histogram_overlay"]

        # raise error if no point clouds loaded
        if self.reference.coords is None or self.compare == []:
            raise ValueError(
                "Please load both reference and compare point clouds before generating report."
            )

        # generate html report with histograms for each attribute in attribute_list
        if len(self.compare) > 5:
            warnings.warn(
                "More than 5 comparison point clouds provided. Report will only consider the first five."
            )
            compare = self.compare[:5]
        else:
            compare = self.compare

        # if attribute list is None, use all attributes available in the point clouds (except for x and y coordinates)
        if attribute_list is None:
            logger.info(
                "No attribute list provided. Using all available attributes in the point clouds for the report (except for x and y coordinates)."
            )
            expanded_attribute_list = self._collect_all_attributes(compare)
            expanded_attribute_list.remove("x")
            expanded_attribute_list.remove("y")
        else:
            # Expand attribute names to match all available features for partial names
            expanded_attribute_list = self._expand_attribute_names(
                compare, attribute_list
            )

        if distance_list is not None:
            # Expand distance names to match all available features for partial names
            expanded_distance_list = self._expand_attribute_names(
                compare, distance_list
            )
        else:
            expanded_distance_list = None

        generate_report(
            self.reference,
            compare,
            output_folder,
            expanded_attribute_list,
            distance_list=expanded_distance_list,
            summary_statistics=summary_statistics,
            outlier_statistics=outlier_statistics,
            return_number_metrics=return_number_metrics,
            class_distribution=class_distribution,
            class_mapping=self.class_mapping,
            visualizations=visualizations,
            vis_backend=vis_backend,
            preview=preview,
        )

    def write_point_clouds(self, output_folder: str | Path) -> None:
        """
        Write the reference and comparison point clouds with all newly computed attributes to the specified output folder as LAS files.
        The output files will have the same names as the input files.

        :param output_folder: The folder to write the output LAS files to. If the folder does not exist, it will be created.
        :type output_folder: str
        """
        logger.info(
            f"Writing reference and comparison point clouds with all attributes to {output_folder}..."
        )
        # write reference and compare point clouds (with new attributes) to output folder
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        def output_name(pc: PointCloud) -> str:
            if pc.source is None:
                raise ValueError(
                    "PointCloud has no source file, cannot derive an output filename."
                )
            return Path(pc.source).name

        # write reference
        write_las(self.reference, output_folder / output_name(self.reference))
        # write compare
        for compare_pc in self.compare:
            write_las(compare_pc, output_folder / output_name(compare_pc))


def write_voxelized_point_clouds(
    vapc, filebase: str | Path, reduce_to_feature="closest_to_centroids"
) -> None:
    """
    Write voxelized point cloud (with voxel-based features) to disk as a LAS file

    :param vapc: VAPC instance containing the voxelized point cloud and features
    :type vapc: py4dgeo.Vapc
    :param filebase: Base file path (without extension) to write the voxelized point cloud to. The final filename will be constructed as {filebase}_voxelized_{reduce_to_feature}_v{voxel_size}.laz
    :type filebase: str
    :param reduce_to_feature: The VAPC feature to reduce the voxelized point cloud to for writing. Supported options are "closest_to_centroids" (default), "centroid", and any other feature available in vapc.out. If "closest_to_centroids", the point closest to the voxel centroid will be written. If "centroid", the voxel centroid will be written. If another feature name, the point with the highest value for that feature in each voxel will be written.
    :type reduce_to_feature: str
    """
    vapc = vapc.reduce_to_feature(feature_name=reduce_to_feature)
    suffix = f"_voxelized_{reduce_to_feature}_v{vapc.voxel_size}.laz"
    outpath = Path(str(filebase) + suffix)
    logger.info(
        f"Writing voxelized point cloud to {outpath}, reduced to {reduce_to_feature}..."
    )
    vapc.save_as_las(outpath)


def read_las_file(filepath):
    """Create a PointCloud from a LAS/LAZ file

    :param filepath:
        The filename to read from. It is expected to be in LAS/LAZ format
        and will be processed using laspy.
    :type filepath: str
    :return:
        A PointCloud object containing the data from the file.
    :rtype: PointCloud
    """
    # Read the lasfile using laspy
    las = laspy.read(filepath)

    # get gpstime, return number, number of returns, intensity if available
    gps_time = las.gps_time if "gps_time" in las.point_format.dimension_names else None
    return_number = (
        las.return_number
        if "return_number" in las.point_format.dimension_names
        else None
    )
    number_of_returns = (
        las.number_of_returns
        if "number_of_returns" in las.point_format.dimension_names
        else None
    )
    intensity = (
        las.intensity if "intensity" in las.point_format.dimension_names else None
    )
    classification = (
        las.classification
        if "classification" in las.point_format.dimension_names
        else None
    )
    # create new PointCloud and assign attributes
    pc = PointCloud()
    pc.coords = np.vstack((las.x, las.y, las.z)).T
    pc.gps_time = gps_time
    pc.return_number = return_number
    pc.number_of_returns = number_of_returns
    pc.intensity = intensity
    pc.classification = classification

    return pc


def read_xyz_file(filepath):
    """Create a PointCloud from a XYZ file

    :param filepath:
        The filename to read from. It is expected to be in XYZ format
        and will be processed using numpy.
    :type filepath: str
    :return:
        A PointCloud object containing the data from the file.
    :rtype: PointCloud
    """
    # find delimiter by checking the first line for common delimiters
    delimiter = None
    with open(filepath) as f:
        first_line = f.readline()
        for d in [",", "\t", " ", ";"]:
            # needs to occur at least twice in the first line to be considered a delimiter
            if first_line.count(d) >= 2:
                delimiter = d
                break

    # check if it has a header
    with open(filepath) as f:
        first_line = f.readline()
        has_header = not all(
            c.isdigit() or c.isspace() or c == "." for c in first_line
        )  # crude check: if first line contains non-numeric characters, assume it's a header
    if has_header:
        data = np.loadtxt(filepath, skiprows=1, delimiter=delimiter)
        fields = first_line.strip().split(delimiter)
        # make lower case
        fields = [f.lower() for f in fields]
        x_index = fields.index("x") if "x" in fields else 0
        y_index = fields.index("y") if "y" in fields else 1
        z_index = fields.index("z") if "z" in fields else 2
    else:
        data = np.loadtxt(filepath, delimiter=delimiter)
        x_index, y_index, z_index = 0, 1, 2

    pc = PointCloud()
    pc.coords = data[:, [x_index, y_index, z_index]]
    # we only read additional attributes if there is a header, otherwise we assume it's just XYZ
    if "intensity" in fields:
        intensity_index = fields.index("intensity")
        pc.intensity = data[:, intensity_index]
    if "gps_time" in fields or "time" in fields:
        gps_time_index = (
            fields.index("gps_time") if "gps_time" in fields else fields.index("time")
        )
        pc.gps_time = data[:, gps_time_index]
    if "return_number" in fields:
        return_number_index = fields.index("return_number")
        pc.return_number = data[:, return_number_index].astype(int)
    if "number_of_returns" in fields:
        number_of_returns_index = fields.index("number_of_returns")
        pc.number_of_returns = data[:, number_of_returns_index].astype(int)
    if "classification" in fields:
        classification_index = fields.index("classification")
        pc.classification = data[:, classification_index].astype(int)

    return pc


def write_las(
    pc: PointCloud,
    outfile: str | Path,
    las_point_format=7,
    las_version="1.4",
    las_scales=None,
    las_offset=None,
):
    """
    Function to write a PointCloud object to a LAS/LAZ file with additional attributes.
    Modified after vapc: https://github.com/3dgeo-heidelberg/vapc/blob/main/src/vapc/datahandler.py#L97-L187

    :param pc: PointCloud object to write
    :type pc: PointCloud
    :param outfile: Output file path
    :type outfile: str or Path
    :param las_point_format: LAS point format to use
    :param las_version: LAS version to use
    :param las_scales: las scales for x, y, z
    :type las_scales: list or np.ndarray
    :param las_offset: las offset for x, y, z
    :type las_offset: list or np.ndarray
    """
    if pc.coords is None:
        raise ValueError("PointCloud coordinates are not set.")
    if las_scales is None:
        las_scales = [0.00025, 0.00025, 0.00025]
    if las_offset is None:
        las_offset = np.min(pc.coords, axis=0)

    new_header = laspy.LasHeader(point_format=las_point_format, version=las_version)
    new_header.offsets = las_offset
    new_header.scales = las_scales

    las = laspy.LasData(new_header)
    las.x = pc.coords[:, 0]
    las.y = pc.coords[:, 1]
    las.z = pc.coords[:, 2]

    # standard las fields: gps_time, return_number, number_of_returns, intensity
    standard_fields = [
        "gps_time",
        "return_number",
        "number_of_returns",
        "intensity",
        "classification",
    ]
    for field in standard_fields:
        data = getattr(pc, field)
        if data is not None:
            las[field] = data
    # geometric features (here the feature name already includes the radius suffix)
    for feat_name, feat_data in pc.geometric_features.items():
        add_dimension_to_laz(las, feat_data, feat_name)
    # outlier flag
    for sor_name, sor_data in pc.sor.items():
        add_dimension_to_laz(las, sor_data.astype(np.uint8), sor_name)

    las.write(outfile)


def add_dimension_to_laz(las_file, array, name):
    """
    Adds a new attribute dimension to the LAZ file.

    Writes a new attribute to the LAZ file by adding an extra dimension with the specified name and data type.

    :param array: The data array to be added as a new attribute.
    :param name: The name of the new attribute.
    :type array: np.ndarray
    :type name: str

    :return: None
    :rtype: None
    """
    las_file.add_extra_dim(
        laspy.ExtraBytesParams(name=name, type=array.dtype, description=name)
    )
    las_file[name] = array


def get_attribute_data(pc: PointCloud, attribute: str):
    """
    Function to get the data for a given attribute from a PointCloud.
    The attribute may be
    - a direct attribute of the PointCloud object,
    - one of the coordinate axes ("x", "y", "z"),
    - a geometric feature (e.g. "linearity_0.2")
    - a voxel feature (e.g. "density_0.5")
    - an outlier flag

    :param pc: PointCloud object
    :type pc: PointCloud
    :param attribute: Name of the attribute to retreive
    :type attribute: str
    """
    # make attribute lower case
    attribute = attribute.lower()

    # search for attribute in all possible locations
    if hasattr(pc, attribute):
        return getattr(pc, attribute)

    if attribute in ["x", "y", "z"]:
        if pc.coords is None:
            raise ValueError("PointCloud coordinates are not set.")
        coord_index = {"x": 0, "y": 1, "z": 2}[attribute]
        return pc.coords[:, coord_index]

    # Exact-match lookup for geometric features
    if attribute in pc.geometric_features:
        return pc.geometric_features[attribute]

    # Exact-match lookup for voxel features; return values array if tuple
    if attribute in pc.voxel_features:
        feat = pc.voxel_features[attribute]
        return feat[0] if isinstance(feat, tuple) and len(feat) > 0 else feat

    # Exact-match lookup for c2c distances
    if attribute in pc.c2c:
        return pc.c2c[attribute]

    # Exact-match lookup for outlier flags
    if attribute in pc.sor:
        return pc.sor[attribute]

    raise ValueError(f"Attribute {attribute} not found in point clouds.")
