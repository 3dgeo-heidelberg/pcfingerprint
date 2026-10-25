from pathlib import Path

import numpy as np
import pytest
from py4dgeo.util import Py4DGeoError

import pcfingerprint

# create random seed
rng = np.random.default_rng(45621231543)


@pytest.mark.integration
def test_load_point_clouds(pc_pair_uls_voxels):
    assert pc_pair_uls_voxels.reference.coords is not None
    assert all(sim.coords is not None for sim in pc_pair_uls_voxels.compare)


def test_load_point_clouds_xyz(pc_xyz):
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_reference(pc_xyz)

    assert pc_pair.reference.coords is not None
    assert pc_pair.reference.coords.shape[1] == 3
    assert pc_pair.reference.intensity is not None


@pytest.mark.integration
def test_load_point_clouds_multiple(pc_pair_uls_voxels_multiple):
    assert pc_pair_uls_voxels_multiple.reference.coords is not None
    assert all(sim.coords is not None for sim in pc_pair_uls_voxels_multiple.compare)


def test_only_overlapping_attributes(pc_pair_with_coords):
    for attr in [
        "gps_time",
        "return_number",
        "number_of_returns",
        "intensity",
    ]:
        reference_attr = getattr(pc_pair_with_coords.reference, attr)
        compare_attr = getattr(pc_pair_with_coords.compare[0], attr)
        if reference_attr is None or compare_attr is None:
            assert getattr(pc_pair_with_coords.reference, attr) is None
            assert getattr(pc_pair_with_coords.compare[0], attr) is None


def test_create_pointcloud_numpy():
    points = rng.random((100, 3))
    number_of_returns = rng.integers(1, 5, size=100)
    gpstime = np.arange(100).astype(float) / 10.0
    pc = pcfingerprint.PointCloud(coords=points, gps_time=gpstime)
    pc.number_of_returns = number_of_returns
    assert pc.coords.shape == (100, 3)
    assert pc.gps_time.shape == (100,)
    assert pc.number_of_returns.shape == (100,)


def test_geometric_features_backend_fallback_pgeof(pc_pair_with_coords):
    with pytest.warns(UserWarning, match="Features .* not available in"):
        pc_pair_with_coords.compute_geometric_features(
            feature_names=["linearity", "planarity", "eigenvalue1"],
            radius=1.0,
            backend="pgeof",
        )
    assert "linearity_r1.0" in pc_pair_with_coords.reference.geometric_features
    assert "planarity_r1.0" in pc_pair_with_coords.reference.geometric_features
    assert "eigenvalue1_r1.0" in pc_pair_with_coords.reference.geometric_features


def test_geometric_features_backend_fallback_jakteristics(pc_pair_with_coords):
    with pytest.warns(UserWarning, match="Features .* not available in"):
        pc_pair_with_coords.compute_geometric_features(
            feature_names=["linearity", "planarity", "curvature"],
            radius=1.0,
            backend="jakteristics",
        )
    assert "linearity_r1.0" in pc_pair_with_coords.reference.geometric_features
    assert "planarity_r1.0" in pc_pair_with_coords.reference.geometric_features
    assert "curvature_r1.0" in pc_pair_with_coords.reference.geometric_features


def test_geometric_features_unknown_feature_raises(pc_pair_with_coords):
    with pytest.raises(ValueError, match="not available"):
        pc_pair_with_coords.compute_geometric_features(
            feature_names=["isotropy"],
            radius=1.0,
            backend="jakteristics",
        )


def test_geometric_features_unknown_feature_fails_before_computing(
    pc_pair_with_coords,
):
    """An unknown feature must not leave partially computed density/features behind."""
    with pytest.raises(ValueError, match="not available"):
        pc_pair_with_coords.compute_geometric_features(
            feature_names=["density", "linearity", "isotropy"],
            radius=1.0,
        )
    assert pc_pair_with_coords.reference.geometric_features == {}


def test_geometric_features_unsupported_backend_raises(pc_pair_with_coords):
    with pytest.raises(ValueError, match="not supported"):
        pc_pair_with_coords.compute_geometric_features(
            feature_names=["density"], radius=1.0, backend="nope"
        )
    assert pc_pair_with_coords.reference.geometric_features == {}


def test_voxel_features_different_voxel_sizes(pc_pair_with_coords):
    """Test that different voxel sizes produce different feature values."""
    pc_pair_with_coords.compute_voxel_features(
        voxel_size=0.5, feature_names=["density"]
    )

    pc_pair_with_coords.compute_voxel_features(
        voxel_size=1.0, feature_names=["density"]
    )

    # Different voxel sizes should generally produce different feature arrays
    feat_1 = pc_pair_with_coords.reference.voxel_features["density_v0.5"][0]
    feat_2 = pc_pair_with_coords.reference.voxel_features["density_v1.0"][0]
    assert feat_1.shape != feat_2.shape or not np.allclose(feat_1, feat_2)


def test_voxel_features_all_supported(pc_pair_with_coords):
    """Test computing all supported voxel features."""
    all_features = [
        "count",
        "density",
        "linearity",
        "planarity",
        "sphericity",
        "sum_of_eigenvalues",
        "omnivariance",
        "eigenentropy",
        "anisotropy",
        "surface_variation",
    ]

    pc_pair_with_coords.compute_voxel_features(
        voxel_size=0.5, feature_names=all_features
    )

    for feature_name in all_features:
        assert f"{feature_name}_v0.5" in pc_pair_with_coords.reference.voxel_features


def test_voxel_features_empty_list_computes_all(pc_pair_with_coords):
    """Test that empty feature list computes all available features."""
    pc_pair_with_coords.compute_voxel_features(voxel_size=0.5, feature_names=[])

    # Should compute multiple features when list is empty
    assert len(pc_pair_with_coords.reference.voxel_features) > 1


def test_voxel_features_data_structure(pc_pair_with_coords):
    """Test that voxel feature data has correct structure and dimensions."""
    pc_pair_with_coords.compute_voxel_features(voxel_size=0.5, feature_names=["count"])

    feat_data = pc_pair_with_coords.reference.voxel_features["count_v0.5"]
    feat_values, voxel_coords, rep_indices = feat_data

    # All components should have matching first dimension (number of voxels)
    assert feat_values.shape[0] == voxel_coords.shape[0]
    assert feat_values.shape[0] == rep_indices.shape[0]

    # Coordinates should be 3D
    assert voxel_coords.shape[1] == 3

    # Representative indices should map to actual points
    assert np.max(rep_indices) < pc_pair_with_coords.reference.coords.shape[0]
    assert np.min(rep_indices) >= 0


def test_filter_by_class(pc_pair_with_coords_and_attr):
    pc_pair_filtered = pc_pair_with_coords_and_attr.filter_by_class(
        [1]
    )  # filter to only class 1
    assert pc_pair_filtered.reference.coords is not None
    assert all(sim.coords is not None for sim in pc_pair_filtered.compare)
    assert np.all(pc_pair_filtered.reference.classification == 1)
    for sim in pc_pair_filtered.compare:
        assert np.all(sim.classification == 1)


def test_filter_by_class_missing_classification(pc_pair_with_coords):
    pc_pair_with_coords.reference.classification = None
    with pytest.raises(
        ValueError,
        match="Cannot filter by class if one of the point clouds does not have 'classification' attribute.",
    ):
        pc_pair_with_coords.filter_by_class([1])
    for sim in pc_pair_with_coords.compare:
        sim.classification = None
    with pytest.raises(
        ValueError,
        match="Cannot filter by class if one of the point clouds does not have 'classification' attribute.",
    ):
        pc_pair_with_coords.filter_by_class([1])


def test_collect_all_attributes(pc_pair_with_coords_and_attr_and_feats):
    attributes = pc_pair_with_coords_and_attr_and_feats._collect_all_attributes(
        pc_pair_with_coords_and_attr_and_feats.compare
    )
    expected_attributes = [
        "z",
        "gps_time",
        "intensity",
        "classification",
        "linearity_r0.05",
        "linearity_r0.1",
        "linearity_r0.2",
        "planarity_r0.05",
        "planarity_r0.2",
        "density_r0.5",
        "linearity_v0.5",
        "density_v0.5",
        "density_v1.0",
    ]
    for attr in expected_attributes:
        assert attr in attributes


def test_exact_priority_no_false_positive(pc_pair_with_coords_and_attr_and_feats):
    # Add a feature that contains 'x' to test exact match priority
    pc_pair_with_coords_and_attr_and_feats.reference.geometric_features[
        "x_variance_r0.1"
    ] = np.random.rand(100)
    pc_pair_with_coords_and_attr_and_feats.compare[0].geometric_features[
        "x_variance_r0.1"
    ] = np.random.rand(100)
    expanded = pc_pair_with_coords_and_attr_and_feats._expand_attribute_names(
        pc_pair_with_coords_and_attr_and_feats.compare, ["x"]
    )
    assert expanded == ["x"]


def test_no_match_and_empty(pc_pair_with_coords_and_attr_and_feats):
    assert pc_pair_with_coords_and_attr_and_feats._expand_attribute_names(
        pc_pair_with_coords_and_attr_and_feats.compare, ["nonexistent"]
    ) == ["nonexistent"]
    assert (
        pc_pair_with_coords_and_attr_and_feats._expand_attribute_names(
            pc_pair_with_coords_and_attr_and_feats.compare, []
        )
        == []
    )


def test_filter_by_class_missing_class(pc_pair_with_coords_and_attr):
    # check if it raises a warning when no points are found for the specified class
    with pytest.warns(UserWarning):
        pc_pair_with_coords_and_attr.filter_by_class(
            [999]
        )  # assuming class 999 does not exist


def test_filter_by_class_preserves_voxel_features(pc_pair_with_coords_and_attr):
    pc_pair_with_coords_and_attr.compute_voxel_features(
        voxel_size=1.0, feature_names=["count"]
    )

    pc_pair_filtered = pc_pair_with_coords_and_attr.filter_by_class([1])

    for filtered_pc in [pc_pair_filtered.reference, *pc_pair_filtered.compare]:
        for feat_value in filtered_pc.voxel_features.values():
            feat_data, voxel_coords, rep_indices = feat_value
            assert feat_data.shape[0] == voxel_coords.shape[0]
            assert feat_data.shape[0] == rep_indices.shape[0]
            assert np.all(filtered_pc.classification[rep_indices] == 1)
            assert np.allclose(voxel_coords, filtered_pc.coords[rep_indices])


@pytest.mark.parametrize(
    "feature_names,backend,radius,expected_keys",
    [
        (["linearity"], "pgeof", 1.0, ["linearity_r1.0"]),
        (["density"], "jakteristics", 1.0, ["density_r1.0"]),
        (
            ["linearity", "planarity"],
            "pgeof",
            1.0,
            ["linearity_r1.0", "planarity_r1.0"],
        ),
    ],
    ids=[
        "geom_single_linearity_pgeof",
        "geom_density_jakteristics",
        "geom_multiple_linearity_planarity",
    ],
)
def test_geometric_features_parametrized(
    pc_pair_with_coords, feature_names, backend, radius, expected_keys
):
    """Parametrized checks for geometric feature computation across backends and feature sets."""
    pc_pair_with_coords.compute_geometric_features(
        feature_names=feature_names, radius=radius, backend=backend
    )
    for k in expected_keys:
        assert k in pc_pair_with_coords.reference.geometric_features


@pytest.mark.parametrize(
    "voxel_size,feature_names,expected_feature_names",
    [
        (0.5, ["count"], ["count_v0.5"]),
        (
            0.5,
            ["count", "density", "linearity"],
            ["count_v0.5", "density_v0.5", "linearity_v0.5"],
        ),
        (1.0, ["density"], ["density_v1.0"]),
    ],
    ids=["voxel_count_v0.5", "voxel_multiple_v0.5", "voxel_density_v1.0"],
)
def test_voxel_features_parametrized(
    pc_pair_with_coords, voxel_size, feature_names, expected_feature_names
):
    """Parametrized checks for voxel feature computation and presence/structure."""
    pc_pair_with_coords.compute_voxel_features(
        voxel_size=voxel_size, feature_names=feature_names
    )
    for fname in expected_feature_names:
        assert fname in pc_pair_with_coords.reference.voxel_features
        feat = pc_pair_with_coords.reference.voxel_features[fname]
        assert isinstance(feat, tuple) and len(feat) >= 2


def test_outlier_report_warning_if_not_computed(caplog, pc_pair_with_coords):
    """
    Test that a warning is is logged if outlier statistics are requested
    in the report but have not been computed.
    """
    caplog.set_level("WARNING")
    pc_pair_with_coords.generate_report(
        output_folder="output_dummy",
        attribute_list=["z"],
        summary_statistics=False,
        outlier_statistics=True,
        return_number_metrics=False,
        visualizations=[],
        preview=False,
    )
    assert (
        "Outlier statistics requested but SOR outlier flags not computed for all point clouds."
        in caplog.text
    )


@pytest.mark.parametrize(
    "k,std_dev_multiplier,expected_keys",
    [
        (8, 1.0, ["sor_k8_std1.0"]),
        (16, 1.5, ["sor_k16_std1.5"]),
        (16, 1.66, ["sor_k16_std1.7"]),
    ],
    ids=[
        "sor_k8_std1.0",
        "sor_k16_std1.5",
        "sor_k16_std1.7",
    ],
)
def test_sor_parametrized(pc_pair_with_coords, k, std_dev_multiplier, expected_keys):
    """Parametrized checks for statistical outlier removal."""
    pc_pair_with_coords.compute_statistical_outlier_removal(
        k=k, std_dev_multiplier=std_dev_multiplier
    )
    for key in expected_keys:
        assert key in pc_pair_with_coords.reference.sor


@pytest.mark.parametrize(
    "input_names,expected_list,expect_equal",
    [
        (["x"], ["x"], True),
        (["linearity_r0.1"], ["linearity_r0.1"], True),
        (["linearity"], ["linearity_r0.05", "linearity_r0.1", "linearity_r0.2"], False),
        (["density"], ["density_r0.5", "density_v0.5", "density_v1.0"], False),
        (["LINEARITY"], ["linearity_r0.05"], False),
    ],
    ids=[
        "expand_exact_coordinate_x",
        "expand_exact_feature_linearity_r0_1",
        "expand_partial_geometric_linearity",
        "expand_voxel_and_geometric_density",
        "expand_case_insensitive",
    ],
)
def test_expand_attribute_parametrized(
    pc_pair_with_coords_and_attr_and_feats, input_names, expected_list, expect_equal
):
    """Parametrized attribute-expansion tests covering exact, partial, voxel+geom, and case-insensitive behavior."""
    expanded = pc_pair_with_coords_and_attr_and_feats._expand_attribute_names(
        pc_pair_with_coords_and_attr_and_feats.compare, input_names
    )
    if expect_equal:
        assert expanded == expected_list
    else:
        for e in expected_list:
            assert e in expanded


def test_compute_c2c_distances_basic():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.reference = pcfingerprint.PointCloud(
        coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    )
    pc_pair.compare = [
        pcfingerprint.PointCloud(
            coords=np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 2.0], [2.0, 0.0, 0.1]])
        )
    ]

    pc_pair.compute_c2c_distances()

    assert "c2c_to-compare-0" in pc_pair.reference.c2c
    assert "c2c_to-reference" in pc_pair.compare[0].c2c

    forward = pc_pair.reference.c2c["c2c_to-compare-0"]
    backward = pc_pair.compare[0].c2c["c2c_to-reference"]

    assert forward.shape == (3,)
    assert backward.shape == (3,)

    # forward: nearest compare point to each reference point (by index)
    # ref[1]=(1,0,0) is actually closest to compare[2]=(2,0,0.1), not compare[1]
    np.testing.assert_allclose(forward, [0.5, np.sqrt(1.0**2 + 0.1**2), 0.1])
    # backward: nearest reference point to each compare point (by index)
    # compare[1]=(1,0,2.0) is closest to reference[1]=(1,0,0), at distance 2.0
    np.testing.assert_allclose(backward, [0.5, 2.0, 0.1])


def test_compute_c2c_distances_multiple_compare():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.reference = pcfingerprint.PointCloud(
        coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    )
    pc_pair.compare = [
        pcfingerprint.PointCloud(
            coords=np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [2.0, 0.0, 0.5]])
        ),
        pcfingerprint.PointCloud(
            coords=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
        ),
    ]

    pc_pair.compute_c2c_distances()

    assert set(pc_pair.reference.c2c.keys()) == {
        "c2c_to-compare-0",
        "c2c_to-compare-1",
    }
    np.testing.assert_allclose(
        pc_pair.reference.c2c["c2c_to-compare-0"], [0.5, 0.5, 0.5]
    )
    np.testing.assert_allclose(
        pc_pair.reference.c2c["c2c_to-compare-1"], [1.0, 1.0, 1.0]
    )
    assert "c2c_to-reference" in pc_pair.compare[0].c2c
    assert "c2c_to-reference" in pc_pair.compare[1].c2c
    np.testing.assert_allclose(
        pc_pair.compare[0].c2c["c2c_to-reference"], [0.5, 0.5, 0.5]
    )
    np.testing.assert_allclose(
        pc_pair.compare[1].c2c["c2c_to-reference"], [1.0, 1.0, 1.0]
    )


def test_compute_c2c_distances_max_distance():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.reference = pcfingerprint.PointCloud(
        coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    )
    pc_pair.compare = [
        pcfingerprint.PointCloud(
            coords=np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 2.0], [2.0, 0.0, 0.1]])
        )
    ]

    # without max_distance, the second point has a distance of ~2.24
    pc_pair.compute_c2c_distances(max_distance=1.0)

    forward = pc_pair.reference.c2c["c2c_to-compare-0"]
    backward = pc_pair.compare[0].c2c["c2c_to-reference"]

    # distances within the threshold are unaffected
    assert forward[0] == pytest.approx(0.5)
    assert forward[2] == pytest.approx(0.1)
    assert backward[0] == pytest.approx(0.5)
    assert backward[2] == pytest.approx(0.1)

    # distances beyond max_distance are set to NaN
    assert np.isnan(forward[1])
    assert np.isnan(backward[1])


def test_histogram_per_class_matplotlib(pc_pair_with_coords_and_attr):
    from pcfingerprint.vis_matplotlib import plot_histogram_per_class

    # check if histogram per class can be generated without errors
    fig_list, classes = plot_histogram_per_class(
        series=[(pc, "intensity") for pc in pc_pair_with_coords_and_attr.compare],
        reference=(pc_pair_with_coords_and_attr.reference, "intensity"),
        xlabel="intensity",
    )
    num_classes_expected = 6
    assert fig_list is not None
    assert len(fig_list) == num_classes_expected
    assert [0, 1, 2, 3, 4, 5] in classes


SCAN_RESOLUTION = 0.1  # [deg]
SCAN_POSITION = (0.0, 0.0, 0.0)
N_ANGLES = 40
WALL_DISTANCE = 10.0
OUTLIER_INDICES = np.array([N_ANGLES * 20 + 20, N_ANGLES * 10 + 30])


def make_wall_scan() -> pcfingerprint.PointCloud:
    """Synthetic scan from the origin of a wall at x=10 m, sampled on the angular scan grid.

    Two points are pulled halfway towards the scanner (``OUTLIER_INDICES``), so they are detached
    from their angular neighbors. All points are last returns.
    """
    angles = np.radians(SCAN_RESOLUTION) * np.arange(N_ANGLES)
    phi, theta = np.meshgrid(angles, angles, indexing="ij")
    coords = np.column_stack(
        [
            np.full(phi.size, WALL_DISTANCE),
            WALL_DISTANCE * np.tan(phi.ravel()),
            WALL_DISTANCE * np.tan(theta.ravel()),
        ]
    )
    coords[OUTLIER_INDICES] *= 0.5
    n = coords.shape[0]
    return pcfingerprint.PointCloud(
        coords=coords,
        return_number=np.ones(n, dtype=int),
        number_of_returns=np.ones(n, dtype=int),
    )


@pytest.fixture()
def scan_pair():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.reference = make_wall_scan()
    pc_pair.compare = [make_wall_scan()]
    return pc_pair


SCOR_KEY = f"scor_res{SCAN_RESOLUTION}_inc1"


def test_scor_stores_result_for_all_point_clouds(scan_pair):
    scan_pair.compute_scan_outlier_ratio(SCAN_POSITION, SCAN_RESOLUTION, 1)
    for pc in [scan_pair.reference, *scan_pair.compare]:
        scor = pc.scor[SCOR_KEY]
        assert scor.shape == (pc.coords.shape[0],)
        assert np.all((scor >= 0) & (scor <= 1))


def test_scor_detached_points_have_low_values(scan_pair):
    scan_pair.compute_scan_outlier_ratio(SCAN_POSITION, SCAN_RESOLUTION, 1)
    scor = scan_pair.reference.scor[SCOR_KEY]
    inliers = np.setdiff1d(np.arange(scor.size), OUTLIER_INDICES)
    assert scor[OUTLIER_INDICES].max() < np.median(scor[inliers])


def test_scor_half_increment(scan_pair):
    scan_pair.compute_scan_outlier_ratio(SCAN_POSITION, SCAN_RESOLUTION, 0.5)
    assert f"scor_res{SCAN_RESOLUTION}_inc0.5" in scan_pair.reference.scor


def test_scor_only_last_returns_changes_candidates(scan_pair):
    # mark a block of points as non-last returns; they must not serve as neighborhood candidates
    for pc in [scan_pair.reference, *scan_pair.compare]:
        pc.return_number[N_ANGLES * 5 : N_ANGLES * 15] = 1
        pc.number_of_returns[N_ANGLES * 5 : N_ANGLES * 15] = 2
    scan_pair.compute_scan_outlier_ratio(SCAN_POSITION, SCAN_RESOLUTION, 1)
    last_only = scan_pair.reference.scor[SCOR_KEY].copy()
    scan_pair.compute_scan_outlier_ratio(
        SCAN_POSITION, SCAN_RESOLUTION, 1, only_last_returns=False
    )
    all_returns = scan_pair.reference.scor[SCOR_KEY]
    assert not np.allclose(last_only, all_returns)


def test_scor_all_returns_without_return_attributes(scan_pair):
    for pc in [scan_pair.reference, *scan_pair.compare]:
        pc.return_number = None
        pc.number_of_returns = None
    scan_pair.compute_scan_outlier_ratio(
        SCAN_POSITION, SCAN_RESOLUTION, 1, only_last_returns=False
    )
    assert SCOR_KEY in scan_pair.reference.scor


def test_scor_per_compare_scan_parameters(scan_pair):
    scan_pair.compute_scan_outlier_ratio(
        SCAN_POSITION,
        SCAN_RESOLUTION,
        1,
        scan_positions_comp=np.array([SCAN_POSITION]),
        scan_resolutions_comp=[0.5],
    )
    assert SCOR_KEY in scan_pair.reference.scor
    assert "scor_res0.5_inc1" in scan_pair.compare[0].scor
    # different results (despite identical points) due to compare-specific resolution
    assert not np.allclose(
        scan_pair.reference.scor[SCOR_KEY],
        scan_pair.compare[0].scor["scor_res0.5_inc1"],
    )


def test_scor_not_loaded_raises():
    with pytest.raises(ValueError, match="load both reference and compare"):
        pcfingerprint.PointCloudPair().compute_scan_outlier_ratio(
            SCAN_POSITION, SCAN_RESOLUTION, 1
        )


def test_scor_missing_return_attributes_raises(scan_pair):
    scan_pair.compare[0].number_of_returns = None
    with pytest.raises(ValueError, match="return_number"):
        scan_pair.compute_scan_outlier_ratio(SCAN_POSITION, SCAN_RESOLUTION, 1)


def test_scor_no_last_returns_raises(scan_pair):
    scan_pair.reference.number_of_returns[:] = 2
    with pytest.raises(ValueError, match="No last returns"):
        scan_pair.compute_scan_outlier_ratio(SCAN_POSITION, SCAN_RESOLUTION, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scan_positions_comp": [SCAN_POSITION, SCAN_POSITION]},
        {"scan_resolutions_comp": [0.1, 0.2]},
    ],
    ids=["positions", "resolutions"],
)
def test_scor_comp_length_mismatch_raises(scan_pair, kwargs):
    with pytest.raises(ValueError, match="one entry per compare"):
        scan_pair.compute_scan_outlier_ratio(
            SCAN_POSITION, SCAN_RESOLUTION, 1, **kwargs
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scan_position": (0.0, 0.0)},
        {"scan_position": (0.0, 0.0, np.nan)},
        {"increment": 0},
        {"increment": 1.3},
    ],
    ids=["position-length", "position-nan", "increment-zero", "increment-fractional"],
)
def test_scor_invalid_arguments_raise_py4dgeo_error(scan_pair, kwargs):
    args = {
        "scan_position": SCAN_POSITION,
        "scan_resolution": SCAN_RESOLUTION,
        "increment": 1,
    }
    args.update(kwargs)
    with pytest.raises(Py4DGeoError):
        scan_pair.compute_scan_outlier_ratio(**args)


@pytest.mark.parametrize("resolution", [0, -0.1, np.nan])
def test_scor_invalid_resolution_raises(scan_pair, resolution):
    with pytest.raises(ValueError, match="scan_resolution"):
        scan_pair.compute_scan_outlier_ratio(SCAN_POSITION, resolution, 1)


def test_mask_preserves_scor(scan_pair):
    scan_pair.compute_scan_outlier_ratio(SCAN_POSITION, SCAN_RESOLUTION, 1)
    masked = scan_pair.reference.mask(np.arange(10))
    np.testing.assert_array_equal(
        masked.scor[SCOR_KEY], scan_pair.reference.scor[SCOR_KEY][:10]
    )


def test_require_loaded_checks_compare_coords(pc_pair_with_coords):
    pc_pair_with_coords.compare.append(pcfingerprint.PointCloud())
    with pytest.raises(ValueError, match="load both reference and compare"):
        pc_pair_with_coords.compute_statistical_outlier_removal()


def test_both_intensity_scalings_raise_on_init():
    with pytest.raises(ValueError, match="Cannot apply both"):
        pcfingerprint.PointCloudPair(
            standardize_intensity=True, robust_scale_intensity=True
        )


def test_both_intensity_scalings_raise_before_loading(pc_xyz):
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.standardize_intensity = True
    pc_pair.robust_scale_intensity = True
    with pytest.raises(ValueError, match="Cannot apply both"):
        pc_pair.load_compare(pc_xyz)
    assert pc_pair.compare == []
    with pytest.raises(ValueError, match="Cannot apply both"):
        pc_pair.load_reference(pc_xyz)
    assert pc_pair.reference.coords is None


def test_load_compare_failure_leaves_pair_unchanged(pc_xyz, tmp_path):
    bad = tmp_path / "points.unsupported"
    bad.write_text("")
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_compare(pc_xyz)
    with pytest.raises(ValueError, match="not supported"):
        pc_pair.load_compare([pc_xyz, bad])
    assert len(pc_pair.compare) == 1


def test_load_reference_failure_keeps_previous_reference(pc_xyz, tmp_path):
    bad = tmp_path / "points.unsupported"
    bad.write_text("")
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_reference(pc_xyz)
    previous = pc_pair.reference
    with pytest.raises(ValueError, match="not supported"):
        pc_pair.load_reference(bad)
    assert pc_pair.reference is previous


def test_load_compare_preprocesses_only_new_clouds(pc_xyz):
    pc_pair = pcfingerprint.PointCloudPair(shift_z=True)
    pc_pair.load_compare(pc_xyz)
    first = pc_pair.compare[0]
    first.coords[:, 2] += 5.0  # would be shifted back to 0 if preprocessed again
    pc_pair.load_compare(pc_xyz)
    assert first.coords[:, 2].min() == pytest.approx(5.0)
    assert pc_pair.compare[1].coords[:, 2].min() == pytest.approx(0.0)


def test_mask_subsets_all_per_point_attributes(pc_pair_with_coords_and_attr_and_feats):
    pc = pc_pair_with_coords_and_attr_and_feats.reference
    n = pc.coords.shape[0]
    pc.return_number = np.ones(n, dtype=int)
    pc.number_of_returns = np.full(n, 2)
    pc.c2c["c2c_to-reference"] = rng.random(n)
    pc.sor["sor_k8_std1.0"] = rng.random(n) > 0.5
    pc.scor["scor_res0.1_inc1"] = rng.random(n)
    pc.source = Path("ref.las")

    indices = np.array([5, 3, 40, 41])
    with pytest.warns(UserWarning, match="Masking geometric feature"):
        masked = pc.mask(indices)

    assert masked.source == pc.source
    np.testing.assert_array_equal(masked.coords, pc.coords[indices])
    for attr in pcfingerprint.PointCloud._POINT_ARRAYS:
        np.testing.assert_array_equal(getattr(masked, attr), getattr(pc, attr)[indices])
    for attr in pcfingerprint.PointCloud._POINT_DICTS:
        original = getattr(pc, attr)
        assert original, f"test data must contain entries for {attr}"
        assert getattr(masked, attr).keys() == original.keys()
        for name, data in original.items():
            np.testing.assert_array_equal(getattr(masked, attr)[name], data[indices])


def test_mask_remaps_voxel_features_separately(pc_pair_with_coords_and_attr_and_feats):
    pc = pc_pair_with_coords_and_attr_and_feats.reference
    indices = np.arange(0, 100, 2)  # keep even points only
    with pytest.warns(UserWarning, match="Masking geometric feature"):
        masked = pc.mask(indices)

    assert "voxel_features" not in pcfingerprint.PointCloud._POINT_DICTS
    for name, (values, voxel_coords, rep_indices) in pc.voxel_features.items():
        m_values, m_coords, m_rep = masked.voxel_features[name]
        keep = np.isin(
            rep_indices, indices
        )  # voxels whose representative point is kept
        np.testing.assert_array_equal(m_values, values[keep])
        np.testing.assert_array_equal(m_coords, voxel_coords[keep])
        # representative indices now refer to positions in the masked point cloud
        np.testing.assert_array_equal(
            masked.coords[m_rep], pc.coords[rep_indices[keep]]
        )


def test_collect_all_attributes_includes_all_attribute_groups(
    pc_pair_with_coords_and_attr_and_feats,
):
    pc_pair = pc_pair_with_coords_and_attr_and_feats
    pc_pair.compute_c2c_distances()
    pc_pair.compute_statistical_outlier_removal()
    pc_pair.reference.scor["scor_res0.1_inc1"] = np.zeros(100)
    attributes = pc_pair._collect_all_attributes(pc_pair.compare)
    assert attributes[:3] == ["x", "y", "z"]
    for expected in [
        "gps_time",
        "intensity",
        "classification",
        "linearity_r0.05",
        "linearity_v0.5",
        "c2c_to-reference",
        "sor_k8_std1.0",
        "scor_res0.1_inc1",
    ]:
        assert expected in attributes
    assert len(attributes) == len(set(attributes))


def test_collect_all_attributes_uses_given_compare_list(
    pc_pair_with_coords_and_attr_and_feats,
):
    """Only the passed compare clouds are considered (the report truncates to five)."""
    pc_pair = pc_pair_with_coords_and_attr_and_feats
    pc_pair.compare[0].geometric_features["only_in_compare_r1.0"] = np.zeros(100)
    assert "only_in_compare_r1.0" in pc_pair._collect_all_attributes(pc_pair.compare)
    assert "only_in_compare_r1.0" not in pc_pair._collect_all_attributes([])


def test_get_attribute_data_scor_and_coordinates(pc_pair_with_coords):
    pc = pc_pair_with_coords.reference
    pc.scor["scor_res0.1_inc1"] = np.arange(100.0)
    np.testing.assert_array_equal(
        pcfingerprint.pcfingerprint.get_attribute_data(pc, "scor_res0.1_inc1"),
        np.arange(100.0),
    )
    np.testing.assert_array_equal(
        pcfingerprint.pcfingerprint.get_attribute_data(pc, "Z"), pc.coords[:, 2]
    )


@pytest.mark.parametrize(
    "attr",
    ["gps_time", "return_number", "number_of_returns", "intensity", "classification"],
)
def test_drop_attributes_drops_attribute_missing_in_one_cloud(
    pc_pair_with_coords, attr
):
    n = 100
    for pc in pc_pair_with_coords._all_pcs:
        setattr(pc, attr, np.zeros(n, dtype=int))
    pc_pair_with_coords.compare[0].__dict__[attr] = None
    pc_pair_with_coords.drop_attributes()
    assert all(getattr(pc, attr) is None for pc in pc_pair_with_coords._all_pcs)
