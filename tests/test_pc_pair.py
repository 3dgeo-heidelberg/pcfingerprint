import pcfingerprint
import numpy as np
import pytest

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


def test_geometric_features_backend_fallback(pc_pair_with_coords):
    with pytest.warns(UserWarning, match="Features .* not available in"):
        pc_pair_with_coords.compute_geometric_features(
            feature_names=["linearity", "planarity", "eigenvalue1"],
            radius=1.0,
            backend="pgeof",
        )
    assert "linearity_r1.0" in pc_pair_with_coords.reference.geometric_features
    assert "planarity_r1.0" in pc_pair_with_coords.reference.geometric_features
    assert "eigenvalue1_r1.0" in pc_pair_with_coords.reference.geometric_features


def test_voxel_features_different_voxel_sizes(pc_pair_with_coords):
    """Test that different voxel sizes produce different feature values."""
    vapc_1, _ = pc_pair_with_coords.compute_voxel_features(
        voxel_size=0.5, feature_names=["density"]
    )

    vapc_2, _ = pc_pair_with_coords.compute_voxel_features(
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

    vapc_reference, vapc_comparisons = pc_pair_with_coords.compute_voxel_features(
        voxel_size=0.5, feature_names=all_features
    )

    for feature_name in all_features:
        assert f"{feature_name}_v0.5" in pc_pair_with_coords.reference.voxel_features


def test_voxel_features_empty_list_computes_all(pc_pair_with_coords):
    """Test that empty feature list computes all available features."""
    vapc_reference, vapc_comparisons = pc_pair_with_coords.compute_voxel_features(
        voxel_size=0.5, feature_names=[]
    )

    # Should compute multiple features when list is empty
    assert len(pc_pair_with_coords.reference.voxel_features) > 1


def test_voxel_features_data_structure(pc_pair_with_coords):
    """Test that voxel feature data has correct structure and dimensions."""
    vapc_reference, vapc_comparisons = pc_pair_with_coords.compute_voxel_features(
        voxel_size=0.5, feature_names=["count"]
    )

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
        for feat_name, feat_value in filtered_pc.voxel_features.items():
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
    vapc_reference, vapc_comparisons = pc_pair_with_coords.compute_voxel_features(
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
    for k in expected_keys:
        assert k in pc_pair_with_coords.reference.sor


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


def test_histogram_per_class_matplotlib(pc_pair_with_coords_and_attr):
    from pcfingerprint.vis_matplotlib import plot_histogram_per_class

    # check if histogram per class can be generated without errors
    fig_list, classes = plot_histogram_per_class(
        pc_pair_with_coords_and_attr.reference,
        pc_pair_with_coords_and_attr.compare,
        attribute="intensity",
    )
    num_classes_expected = 6
    assert fig_list is not None
    assert len(fig_list) == num_classes_expected
    assert [0, 1, 2, 3, 4, 5] in classes
