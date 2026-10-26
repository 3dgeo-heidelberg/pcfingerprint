from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

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

_LAS_SUFFIXES = (".las", ".laz")
_XYZ_SUFFIXES = (".txt", ".xyz", ".csv")

VoxelFeature = tuple[np.ndarray, np.ndarray, np.ndarray | None]
"""Voxel-based feature: (feature values per voxel, voxel coordinates Vx3, index of the representative point per voxel or None)."""

_SCOR_RESOLUTION_OFFSET = 0.0028
"""Offset [deg] added to the doubled scan resolution passed to py4dgeo's ScOR (as in the py4dgeo outlier removal notebook)."""


class PointCloud:
    _COORDINATES: ClassVar[tuple[str, ...]] = ("x", "y", "z")
    """Names of the coordinate axes, in the column order of ``coords``."""
    _POINT_ARRAYS: ClassVar[tuple[str, ...]] = (
        "gps_time",
        "return_number",
        "number_of_returns",
        "intensity",
        "classification",
    )
    """Names of optional per-point arrays that are subset together with ``coords`` when masking."""
    _POINT_DICTS: ClassVar[tuple[str, ...]] = (
        "geometric_features",
        "c2c",
        "sor",
        "scor",
    )
    """Names of dictionaries holding one per-point array per key that are subset together with ``coords`` when masking.
    ``voxel_features`` is deliberately not listed: it is indexed by voxel, not by point, and is handled separately."""

    def __init__(
        self,
        coords: np.ndarray | None = None,
        gps_time: np.ndarray | None = None,
        return_number: np.ndarray | None = None,
        number_of_returns: np.ndarray | None = None,
        intensity: np.ndarray | None = None,
        classification: np.ndarray | None = None,
    ) -> None:
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
        self.voxel_features: dict[str, VoxelFeature] = {}
        """Feature name to voxel-based feature (values, voxel coordinates, representative point indices)."""
        self.c2c: dict[str, np.ndarray] = {}
        """Cloud-to-cloud distances, keyed by distance name (e.g. ``"c2c_to-reference"``)."""
        self.sor: dict[str, np.ndarray] = {}
        """Statistical outlier removal results, keyed by feature name."""
        self.scor: dict[str, np.ndarray] = {}
        """Scan outlier ratio results, keyed by feature name."""

    @property
    def display_name(self) -> str:
        """File name of the source, or ``"unknown source"``; used in log messages."""
        return Path(self.source).name if self.source is not None else "unknown source"

    def shift_z_to_zero(self) -> None:
        """Shift the z-coordinates so that the minimum z-value is zero."""
        if self.coords is not None:
            self.coords[:, 2] = shift_to_zero(self.coords[:, 2])
        else:
            raise ValueError("PointCloud coordinates are not set.")

    def shift_gpstime_to_zero(self) -> None:
        """Shift the gps_time so that the minimum gps_time is zero."""
        if self.gps_time is not None:
            self.gps_time = shift_to_zero(self.gps_time)
        else:
            raise ValueError("PointCloud gps_time is not set.")

    def robust_scale_intensity(self) -> None:
        """Perform robust scaling on intensity values."""
        if self.intensity is not None:
            self.intensity = scale_robust(self.intensity)
        else:
            raise ValueError("PointCloud intensity is not set.")

    def standardize_intensity(self) -> None:
        """Standardize intensity values (z-score normalization)."""
        if self.intensity is not None:
            self.intensity = standardize(self.intensity)
        else:
            raise ValueError("PointCloud intensity is not set.")

    def _selection_mask_and_index_map(
        self, indices: np.ndarray | Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.coords is None:
            raise ValueError("PointCloud coordinates are not set.")
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

    def _mask_voxel_feature(
        self, feat_data: VoxelFeature, point_mask: np.ndarray, index_map: np.ndarray
    ) -> VoxelFeature:
        if self.coords is None:
            raise ValueError("PointCloud coordinates are not set.")
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

    def filter_by_class(self, classes: list[int]) -> PointCloud:
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

    def mask(self, indices: np.ndarray | Sequence[int]) -> PointCloud:
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
            for attr in self._POINT_ARRAYS:
                data = getattr(self, attr)
                if data is not None:
                    setattr(new_pc, attr, data[indices])
            for feat_name in self.geometric_features:
                warnings.warn(
                    f"Masking geometric feature {feat_name}. If using mask to subset points, consider recomputing geometric features",
                    UserWarning,
                )
            for attr in self._POINT_DICTS:
                masked: dict[str, np.ndarray] = getattr(new_pc, attr)
                for name, data in getattr(self, attr).items():
                    masked[name] = data[indices]
            # voxel features are indexed by voxel, not by point, and need their indices remapped
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
    ) -> None:
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
        self._check_intensity_scaling_options()

    @property
    def _all_pcs(self) -> list[PointCloud]:
        return [self.reference, *self.compare]

    def _check_intensity_scaling_options(self) -> None:
        # checked in __init__ and again before loading, as the flags are public attributes
        if self.robust_scale_intensity and self.standardize_intensity:
            raise ValueError(
                "Cannot apply both robust scaling and standardization to intensity."
            )

    def _preprocess(self, pcs: list[PointCloud]) -> None:
        """Apply the configured shifting and intensity scaling to the given point clouds."""
        for pc in pcs:
            if self.shift_z:
                pc.shift_z_to_zero()
            if self.shift_gps_time:
                pc.shift_gpstime_to_zero()
            if self.robust_scale_intensity:
                pc.robust_scale_intensity()
            if self.standardize_intensity:
                pc.standardize_intensity()

    def _require_loaded(self, action: str) -> None:
        if self.compare == [] or any(pc.coords is None for pc in self._all_pcs):
            raise ValueError(
                f"Please load both reference and compare point clouds before {action}."
            )

    def drop_attributes(self) -> None:
        """
        Drop attributes that are not present in all point clouds in the pair to ensure comparability.
        """
        for attr in PointCloud._POINT_ARRAYS:
            if any(getattr(pc, attr) is None for pc in self._all_pcs):
                logger.info(
                    f"Dropping attribute '{attr}' as it is not present in all point clouds."
                )
                for pc in self._all_pcs:
                    setattr(pc, attr, None)

    def load_reference(self, filepath: str | Path) -> None:
        """
        Load the reference point cloud from a file. Supported formats are LAS/LAZ and XYZ.
        The previously loaded reference is only replaced if loading and preprocessing succeed.

        :param filepath: File path to load. Supported formats are LAS/LAZ and XYZ.
        :type filepath: str
        """
        self._check_intensity_scaling_options()
        logger.info(f"Loading reference point cloud from {filepath}...")
        reference = _read_point_cloud(filepath)
        self._preprocess([reference])
        self.reference = reference
        # if comparison point cloud is already loaded, drop non-overlapping attributes
        if self.compare != []:
            self.drop_attributes()

    def load_compare(self, filepaths: Sequence[str | Path] | str | Path) -> None:
        """
        Load comparison point cloud(s) from file(s).
        Users can provide both a single file path (str or Path) and a list of file paths.
        The comparison point clouds are only added if all files could be loaded and preprocessed;
        otherwise the pair is left unchanged. Point clouds that were loaded earlier are not preprocessed again.

        :param filepaths: File path(s) to load. Can be a single file path or a sequence of file paths (str or Path). Supported formats are LAS/LAZ and XYZ.
        :type filepaths: Sequence[str | Path] | str | Path
        """
        self._check_intensity_scaling_options()
        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]
        new_pcs: list[PointCloud] = []
        for filepath in filepaths:
            logger.info(f"Loading comparison point cloud from {filepath}...")
            new_pcs.append(_read_point_cloud(filepath))
        self._preprocess(new_pcs)
        self.compare.extend(new_pcs)
        # if reference world point cloud is already loaded, drop non-overlapping attributes
        if self.reference.coords is not None:
            self.drop_attributes()

    def compute_statistical_outlier_removal(
        self, k: int = 8, std_dev_multiplier: float = 1.0
    ) -> None:
        """
        Perform statistical outlier removal (SOR) on the reference and all comparison point clouds.
        Uses py4dgeo.statistical_outlier_removal and adds a boolean array named by the parameters to the sor dictionary of each PointCloud.

        :param k: Number of nearest neighbors to consider for each point. Default is 8.
        :type k: int
        :param std_dev_multiplier: Standard deviation multiplier for determining outliers. Default is 1.0.
        :type std_dev_multiplier: float
        """
        self._require_loaded("computing statistical outlier removal")
        # round std_dev_multiplier to 1 decimal place
        std_dev_multiplier = round(std_dev_multiplier, 1)
        for pc in self._all_pcs:
            epoch = py4dgeo.Epoch(pc.coords)
            _, inlier_outlier = py4dgeo.statistical_outlier_removal(
                epoch, k=k, std_dev_multiplier=std_dev_multiplier
            )
            pc.sor[f"sor_k{k}_std{std_dev_multiplier}"] = inlier_outlier

    def compute_scan_outlier_ratio(
        self,
        scan_position: Sequence[float] | np.ndarray,
        scan_resolution: float,
        increment: float,
        only_last_returns: bool = True,
        scan_positions_comp: Sequence[Sequence[float]] | np.ndarray | None = None,
        scan_resolutions_comp: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """
        Compute the scan outlier ratio (ScOR) on the reference and all comparison point clouds.
        Uses py4dgeo.scan_outlier_ratio and stores the resulting per-point ScOR values in the scor dictionary of each PointCloud (key format described below).
        Each point cloud is evaluated against its own (optionally last-return) points as neighborhood candidates (standard ScOR, not multitemporal ScOR).

        Note that this assumes that each point cloud is a single scan captured from a static position and that the
        vertical angular resolution is equal to the horizontal angular resolution.
        Not suitable for moving platforms or multi-scan terrestrial laser scans.

        The scan resolution passed to py4dgeo is adjusted as ``2 * scan_resolution + 0.0028`` (following the py4dgeo outlier removal notebook).
        The result is stored under the key ``scor_res{scan_resolution}_inc{increment}`` using the unadjusted resolution of the respective point cloud.

        .. seealso::
           Tabernig & Höfle (2026): https://doi.org/10.5194/isprs-annals-XI-2-2026-155-2026

        :param scan_position: 3D scan position (x, y, z) of the reference scan in the coordinate system of the point cloud. Also used for all compare point clouds unless ``scan_positions_comp`` is given.
        :type scan_position: Sequence[float] | np.ndarray
        :param scan_resolution: Angular resolution of the scan in degrees. Also used for all compare point clouds unless ``scan_resolutions_comp`` is given.
        :type scan_resolution: float
        :param increment: Size of the angular neighborhood: 0.5 selects the four adjacent grid cells, a positive integer all cells within that many angular bins.
        :type increment: float
        :param only_last_returns: If True, only last returns (``return_number == number_of_returns``) are used as neighborhood candidates. Requires the ``return_number`` and ``number_of_returns`` attributes in all point clouds. Default is True.
        :type only_last_returns: bool
        :param scan_positions_comp: 3D scan positions (one per compare point cloud, in the order of ``compare``) if they differ from ``scan_position``. Must have the same length as the number of compare point clouds. Default is None.
        :type scan_positions_comp: Sequence[Sequence[float]] | np.ndarray | None
        :param scan_resolutions_comp: Angular resolutions in degrees (one per compare point cloud, in the order of ``compare``) if they differ from ``scan_resolution``. Must have the same length as the number of compare point clouds. Default is None.
        :type scan_resolutions_comp: Sequence[float] | np.ndarray | None

        :raises ValueError: If the reference or compare point clouds are not loaded, if ``only_last_returns`` is set but the return attributes are missing, if the number of ``scan_positions_comp`` / ``scan_resolutions_comp`` entries does not match the number of compare point clouds, if a scan resolution is not positive, or if no last returns are found.
        :raises py4dgeo.util.Py4DGeoError: If a scan position or the increment is invalid (validated by py4dgeo).
        """
        self._require_loaded("computing scan outlier ratio")
        if only_last_returns and any(
            pc.return_number is None or pc.number_of_returns is None
            for pc in self._all_pcs
        ):
            raise ValueError(
                "only_last_returns=True requires 'return_number' and 'number_of_returns' "
                "in all point clouds. Set only_last_returns=False to use all points."
            )
        n_comp = len(self.compare)
        positions = [scan_position] + (
            list(scan_positions_comp)
            if scan_positions_comp is not None
            else [scan_position] * n_comp
        )
        resolutions = [scan_resolution] + (
            list(scan_resolutions_comp)
            if scan_resolutions_comp is not None
            else [scan_resolution] * n_comp
        )
        if len(positions) != n_comp + 1 or len(resolutions) != n_comp + 1:
            raise ValueError(
                "scan_positions_comp and scan_resolutions_comp must provide one entry "
                f"per compare point cloud ({n_comp})."
            )
        for pc, pos, res in zip(self._all_pcs, positions, resolutions):
            with logger_context(
                f"Computing scan outlier ratio for {pc.display_name}. Scan resolution: {res}, increment: {increment}"
            ):
                pc.scor[f"scor_res{res}_inc{increment}"] = _scor_for_cloud(
                    pc, pos, res, increment, only_last_returns
                )

    def compute_geometric_features(
        self,
        feature_names: list[str],
        radius: float = 2.0,
        max_knn: int = 50_000,
        backend: str = "pgeof",
        chunking: bool = False,
        max_points_per_chunk: int | None = None,
    ) -> None:
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

        :raises ValueError: If the reference or compare point clouds are not loaded, if ``chunking`` is enabled without ``max_points_per_chunk``, if ``backend`` is not supported, or if any requested feature is available in neither backend. All of this is checked before any feature is computed.
        :warns UserWarning: If a requested feature is not available in ``backend`` and is computed with the other backend as fallback.
        """
        self._require_loaded("computing geometric features")
        if chunking and max_points_per_chunk is None:
            raise ValueError(
                "Please provide max_points_per_chunk when chunking is enabled."
            )
        # Compute geometric features for all point clouds
        radius = np.round(radius, 3)
        requested_features = list(feature_names)

        density_requested = "density" in requested_features
        density_fns = {
            "jakteristics": compute_density_jakteristics,
            "pgeof": compute_density_pgeof,
        }
        geom_fns = {
            "jakteristics": compute_geom_feat_jakteristics,
            "pgeof": compute_geom_feat_pgeof,
        }
        feature_names = [f for f in requested_features if f != "density"]

        # validate backend and feature names before computing anything
        primary, fallback = _split_features_by_backend(feature_names, backend)
        other = "jakteristics" if backend == "pgeof" else "pgeof"

        def _compute(
            pc: PointCloud, compute_fn: Callable[..., None], **kwargs: Any
        ) -> None:
            if chunking:
                run_chunked_computation(
                    pc, compute_fn, max_points_per_chunk, buffer=radius, **kwargs
                )
            else:
                compute_fn(pc, **kwargs)

        if density_requested:
            for pc in self._all_pcs:
                with logger_context(
                    f"Computing density for {pc.display_name} using {backend}. Search radius: {radius}, max_knn: {max_knn}"
                ):
                    _compute(pc, density_fns[backend], radius=radius, max_knn=max_knn)

        for name, feats in ((backend, primary), (other, fallback)):
            if not feats:
                continue
            for pc in self._all_pcs:
                with logger_context(
                    f"Computing geometric features for {pc.display_name} using {name}. {feats}, search radius: {radius}, max_knn: {max_knn}"
                ):
                    _compute(
                        pc,
                        geom_fns[name],
                        feature_names=feats,
                        radius=radius,
                        max_knn=max_knn,
                    )

    def compute_voxel_features(
        self,
        voxel_size: float,
        voxel_origin: list[float] | None = None,
        feature_names: list[str] | None = None,
    ) -> tuple[py4dgeo.Vapc, list[py4dgeo.Vapc]]:
        """
        Compute voxel-based geometric features for all point clouds in the pair using VAPC.

        :param voxel_size: Voxel edge length used for voxel-based feature computation.
        :type voxel_size: float
        :param voxel_origin: Optional voxel grid origin passed to the VAPC backend. If None, [0.0, 0.0, 0.0] is used.
        :type voxel_origin: list[float] | None
        :param feature_names: List of VAPC features to compute. Supported features (py4dgeo v1.1) are "count", "density", "covariance", "eigenvalues", "linearity", "planarity", "sphericity", "sum_of_eigenvalues","omnivariance", "eigenentropy", "anisotropy", "surface_variation". If empty, all available features will be computed.
        :type feature_names: list[str] | None

        :return: Tuple containing (VAPC for reference point cloud, list of VAPCs for compare point clouds)
        :rtype: tuple[py4dgeo.Vapc, list[py4dgeo.Vapc]]
        """
        self._require_loaded("computing voxel features")
        if feature_names is None:
            feature_names = []
        vapc_list: list[py4dgeo.Vapc] = []
        for pc in self._all_pcs:
            with logger_context(
                f"Computing voxel features for {pc.display_name} using VAPC. {feature_names}, voxel size: {voxel_size}"
            ):
                vapc = compute_geom_feat_vapc(
                    pc, voxel_size, voxel_origin, feature_names
                )
            vapc_list.append(vapc)

        return (
            vapc_list[0],
            vapc_list[1:],
        )  # return reference VAPC and list of compare VAPCs

    def compute_c2c_distances(self, max_distance: float = np.inf) -> None:
        """
        Compute cloud-to-cloud distances between the reference and each comparison point cloud (both directions).
        The distances are stored in
        - the c2c attribute of the reference point cloud (dictionary with arrays of distances to each comparison point cloud)
        - the c2c attribute of each comparison point cloud (dictionary with a single array of distances to the reference point cloud)

        :param max_distance: Optional maximum distance threshold. Distances greater than this value will be set to NaN. Default: np.inf, i.e., no threshold is applied.
        :type max_distance: float, optional
        """
        self._require_loaded("computing C2C distances")
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

    def filter_by_class(self, classes: list[int]) -> PointCloudPair:
        """
        Filter the point clouds in the PointCloudPair to specified classes using the "classification" attribute (if present).

        :param classes: List of class values to filter by.
        :type classes: list[int]

        :return: A new PointCloudPair object containing only points of the specified classes.
        :rtype: PointCloudPair
        """
        self._require_loaded("filtering by class")
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

    def _collect_all_attributes(self, compare: list[PointCloud]) -> list[str]:
        """
        Collect all available attributes from the reference and compare point clouds to use
        for report generation. Returns a deterministic, deduplicated list of attribute names.

        :param compare: list of comparison PointCloud objects
        :type compare: list[PointCloud]
        :return: Ordered list of attribute names available in the point clouds
        :rtype: list[str]
        """
        attributes: list[str] = []

        def add_unique(name: str) -> None:
            if name not in attributes:
                attributes.append(name)

        # start with coordinates
        for coord in PointCloud._COORDINATES:
            add_unique(coord)

        # use the (possibly truncated) compare list, not all point clouds of the pair
        for pc in [self.reference, *compare]:
            for attr in PointCloud._POINT_ARRAYS:
                if getattr(pc, attr) is not None:
                    add_unique(attr)
            # sort feature keys to ensure consistent ordering across runs
            for dict_attr in (*PointCloud._POINT_DICTS, "voxel_features"):
                for name in sorted(getattr(pc, dict_attr)):
                    add_unique(name)

        return attributes

    def _expand_attribute_names(
        self, compare: list[PointCloud], attribute_list: list[str]
    ) -> list[str]:
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
    ) -> None:
        """
        Generate an HTML report comparing the reference and the comparison point clouds.
        If more than five comparison point clouds are loaded, only the first five are included.

        :param output_folder: Folder to write the report to. It is created if it does not exist.
        :type output_folder: str | Path
        :param attribute_list: Point attributes (e.g. "intensity", "z") and geometric/voxel feature names to include. Names without a suffix are expanded to all matching features, e.g. "linearity" to "linearity_r0.1", "linearity_r0.2". If None, all available attributes except the x and y coordinates are used.
        :type attribute_list: list[str] | None
        :param distance_list: Distance attributes to include, e.g. "c2c" (expanded like ``attribute_list``). Default is None.
        :type distance_list: list[str] | None
        :param summary_statistics: Whether to include summary statistics tables. Default is True.
        :type summary_statistics: bool
        :param class_distribution: Whether to include class distribution metrics. Requires the classification attribute. Default is False.
        :type class_distribution: bool
        :param outlier_statistics: Whether to include outlier statistics. Requires that statistical outlier removal was computed for all point clouds, see :meth:`compute_statistical_outlier_removal`. Default is False.
        :type outlier_statistics: bool
        :param return_number_metrics: Whether to include return-number metrics. Default is False.
        :type return_number_metrics: bool
        :param visualizations: Visualization types to include: "histogram", "histogram_overlay", "histogram_per_class", "3d_scatter" and "3d_scatter_distances". If None, ["histogram_overlay"] is used.
        :type visualizations: list[str] | None
        :param vis_backend: Visualization backend, "matplotlib" or "plotly". Default is "matplotlib".
        :type vis_backend: str
        :param preview: Whether to start a local web server and open the report in a browser. Default is True.
        :type preview: bool

        :raises ValueError: If the reference or compare point clouds are not loaded.
        :raises NotImplementedError: If an unsupported visualization backend is requested.
        """
        if visualizations is None:
            visualizations = ["histogram_overlay"]

        self._require_loaded("generating report")

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


def _split_features_by_backend(
    feature_names: list[str], backend: str
) -> tuple[list[str], list[str]]:
    """Return (primary, fallback) features for ``backend``.

    ``primary`` is available in ``backend``; ``fallback`` is only available in the other one.
    Raises ValueError for an unsupported backend or for features available in neither.
    """
    available = {
        "pgeof": {f.lower() for f in EFeatureID._member_names_},
        "jakteristics": set(jakteristics.constants.FEATURE_NAMES),
    }
    if backend not in available:
        raise ValueError(f"Backend {backend} not supported.")
    other = "jakteristics" if backend == "pgeof" else "pgeof"

    primary = [f for f in feature_names if f in available[backend]]
    fallback = [f for f in feature_names if f not in primary and f in available[other]]
    unknown = [f for f in feature_names if f not in primary and f not in fallback]
    if unknown:
        raise ValueError(f"Features {unknown} not available in pgeof or jakteristics.")
    if fallback:
        warnings.warn(
            f"Features {fallback} not available in {backend}. Computing them using {other} as fallback.",
            UserWarning,
        )
    return primary, fallback


def _scor_for_cloud(
    pc: PointCloud,
    scan_position: Sequence[float] | np.ndarray,
    scan_resolution: float,
    increment: float,
    only_last_returns: bool,
) -> np.ndarray:
    """
    Compute the standard ScOR of a single point cloud against its own (optionally last-return) points.

    :param pc: Point cloud with coordinates (and return attributes if ``only_last_returns``).
    :type pc: PointCloud
    :param scan_position: 3D scan position (x, y, z).
    :type scan_position: Sequence[float] | np.ndarray
    :param scan_resolution: Angular resolution of the scan in degrees (unadjusted).
    :type scan_resolution: float
    :param increment: Size of the angular neighborhood: 0.5 selects the four adjacent grid cells, a positive integer all cells within that many angular bins.
    :type increment: float
    :param only_last_returns: Whether to use only last returns as neighborhood candidates.
    :type only_last_returns: bool

    :return: ScOR value per point of ``pc``.
    :rtype: np.ndarray

    :raises ValueError: If ``scan_resolution`` is not positive (checked here because the adjustment applied before calling py4dgeo would otherwise mask it) or no last returns are found.
    :raises py4dgeo.util.Py4DGeoError: If ``scan_position`` or ``increment`` are invalid (validated by py4dgeo).
    """
    if pc.coords is None:
        raise ValueError("PointCloud coordinates are not set.")
    if not scan_resolution > 0:
        raise ValueError(
            f"scan_resolution must be a positive number of degrees, got {scan_resolution}."
        )
    if only_last_returns:
        candidates = pc.coords[pc.return_number == pc.number_of_returns]
        if candidates.shape[0] == 0:
            raise ValueError(f"No last returns found in point cloud {pc.display_name}.")
    else:
        candidates = pc.coords
    scor, _, _ = py4dgeo.scan_outlier_ratio(
        py4dgeo.Epoch(pc.coords),
        neighborhood_candidate_epochs=py4dgeo.Epoch(candidates),
        scan_position=scan_position,
        scan_resolution=2 * scan_resolution + _SCOR_RESOLUTION_OFFSET,
        increment=increment,
    )
    return scor


def write_voxelized_point_clouds(
    vapc: py4dgeo.Vapc,
    filebase: str | Path,
    reduce_to_feature: str = "closest_to_centroids",
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


def _read_point_cloud(filepath: str | Path) -> PointCloud:
    """
    Read a point cloud from a LAS/LAZ or XYZ file, chosen by the (case-insensitive) file extension.

    :param filepath: File to read.
    :type filepath: str | Path

    :return: The point cloud, with ``source`` set to ``filepath``.
    :rtype: PointCloud

    :raises ValueError: If the file extension is not supported.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix in _LAS_SUFFIXES:
        pc = read_las_file(path)
    elif suffix in _XYZ_SUFFIXES:
        pc = read_xyz_file(path)
    else:
        raise ValueError(
            f"File format {path.suffix} not supported. Please provide a LAS/LAZ or XYZ file."
        )
    pc.source = path
    return pc


def read_las_file(filepath: str | Path) -> PointCloud:
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


def read_xyz_file(filepath: str | Path) -> PointCloud:
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
    delimiter: str | None = None
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
    las_point_format: int = 7,
    las_version: str = "1.4",
    las_scales: Sequence[float] | np.ndarray | None = None,
    las_offset: Sequence[float] | np.ndarray | None = None,
) -> None:
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


def add_dimension_to_laz(las_file: laspy.LasData, array: np.ndarray, name: str) -> None:
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


def _lookup_case_insensitive(mapping: dict[str, Any], name: str) -> Any:
    """Return the value of the key matching ``name`` ignoring case, or ``None`` if there is none."""
    for key, value in mapping.items():
        if key.lower() == name:
            return value
    return None


def get_attribute_data(pc: PointCloud, attribute: str) -> np.ndarray:
    """
    Function to get the data for a given attribute from a PointCloud.
    The attribute name is matched case-insensitively and may be
    - a per-point array of the PointCloud object (e.g. "intensity", see ``PointCloud._POINT_ARRAYS``),
    - one of the coordinate axes ("x", "y", "z"),
    - a geometric feature (e.g. "linearity_r0.2"),
    - a voxel feature (e.g. "density_v0.5"),
    - a cloud-to-cloud distance (e.g. "c2c_to-reference"),
    - an outlier flag or a scan outlier ratio.

    :param pc: PointCloud object
    :type pc: PointCloud
    :param attribute: Name of the attribute to retrieve
    :type attribute: str

    :raises ValueError: If the attribute is not found, or is a per-point array that is not set.
    """
    name = attribute.lower()

    # only known data attributes are exposed, not arbitrary members of PointCloud (methods, dictionaries, ...)
    if name in PointCloud._POINT_ARRAYS:
        data = getattr(pc, name)
        if data is None:
            raise ValueError(f"Attribute {attribute} is not set in this point cloud.")
        return data

    if name in PointCloud._COORDINATES:
        if pc.coords is None:
            raise ValueError("PointCloud coordinates are not set.")
        return pc.coords[:, PointCloud._COORDINATES.index(name)]

    # per-point feature dictionaries (keys keep the case of the feature name, e.g. "PCA1_r1.0")
    for dict_attr in PointCloud._POINT_DICTS:
        data = _lookup_case_insensitive(getattr(pc, dict_attr), name)
        if data is not None:
            return data

    # voxel features are indexed by voxel; return the values array if stored as a tuple
    feat = _lookup_case_insensitive(pc.voxel_features, name)
    if feat is not None:
        return feat[0] if isinstance(feat, tuple) and len(feat) > 0 else feat

    raise ValueError(f"Attribute {attribute} not found in point clouds.")
