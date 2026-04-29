from pathlib import Path
import pcfingerprint
import numpy as np
import pytest
from matplotlib.figure import Figure


def test_load_point_clouds(pc_pair_uls_voxels):
    assert pc_pair_uls_voxels.real_world.coords is not None
    assert all(sim.coords is not None for sim in pc_pair_uls_voxels.simulated)


def test_load_point_clouds_xyz():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_real_world("data/livox_avia/avia_office_wall_real-world.csv")

    assert pc_pair.real_world.coords is not None
    assert pc_pair.real_world.coords.shape[1] == 3


def test_load_point_clouds_multiple(pc_pair_uls_voxels_multiple):
    assert pc_pair_uls_voxels_multiple.real_world.coords is not None
    assert all(sim.coords is not None for sim in pc_pair_uls_voxels_multiple.simulated)


def test_only_overlapping_attributes(pc_pair_uls_voxels):
    for attr in [
        "gps_time",
        "return_number",
        "number_of_returns",
        "intensity",
    ]:
        real_attr = getattr(pc_pair_uls_voxels.real_world, attr)
        sim_attr = getattr(pc_pair_uls_voxels.simulated[0], attr)
        if real_attr is None or sim_attr is None:
            assert getattr(pc_pair_uls_voxels.real_world, attr) is None
            assert getattr(pc_pair_uls_voxels.simulated[0], attr) is None


def test_create_pointcloud_numpy():
    points = np.random.rand(100, 3)
    number_of_returns = np.random.randint(1, 5, size=100)
    gpstime = np.arange(100).astype(float) / 10.0
    pc = pcfingerprint.PointCloud(coords=points, gps_time=gpstime)
    pc.number_of_returns = number_of_returns
    assert pc.coords.shape == (100, 3)
    assert pc.gps_time.shape == (100,)
    assert pc.number_of_returns.shape == (100,)


def test_geometric_features_single(pc_pair_uls_voxels):
    pc_pair_uls_voxels.compute_geometric_features(
        feature_names=["linearity"], radius=1.0
    )
    assert "linearity_r1.0" in pc_pair_uls_voxels.real_world.geometric_features


def test_voxel_features_single(pc_pair_uls_voxels):
    """Test computing a single voxel feature."""
    vapc_real, vapc_sims = pc_pair_uls_voxels.compute_voxel_features(
        voxel_size=0.5, feature_names=["count"]
    )

    assert "count_v0.5" in pc_pair_uls_voxels.real_world.voxel_features
    feat_data = pc_pair_uls_voxels.real_world.voxel_features["count_v0.5"]
    assert len(feat_data) == 3  # (values, coords, indices)
    assert feat_data[0].ndim == 1  # feature values are 1D
    assert feat_data[1].ndim == 2 and feat_data[1].shape[1] == 3  # coords are Nx3
    assert feat_data[2].ndim == 1  # indices are 1D


def test_geometric_features_multiple(pc_pair_uls_voxels):
    pc_pair_uls_voxels.compute_geometric_features(
        feature_names=["linearity", "planarity", "scattering"], radius=1.0
    )
    assert "linearity_r1.0" in pc_pair_uls_voxels.real_world.geometric_features
    assert "planarity_r1.0" in pc_pair_uls_voxels.real_world.geometric_features
    assert "scattering_r1.0" in pc_pair_uls_voxels.real_world.geometric_features


def test_voxel_features_multiple(pc_pair_uls_voxels):
    """Test computing multiple voxel features at once."""
    vapc_real, vapc_sims = pc_pair_uls_voxels.compute_voxel_features(
        voxel_size=0.5, feature_names=["count", "density", "linearity"]
    )

    for feature_name in ["count_v0.5", "density_v0.5", "linearity_v0.5"]:
        assert feature_name in pc_pair_uls_voxels.real_world.voxel_features
        assert feature_name in pc_pair_uls_voxels.simulated[0].voxel_features


def test_geometric_features_different_backends(pc_pair_uls_voxels):
    pc_pair_uls_voxels.compute_geometric_features(
        feature_names=["linearity", "curvature"], radius=1.0, backend="pgeof"
    )
    pc_pair_uls_voxels.compute_geometric_features(
        feature_names=["planarity", "eigenvalue1"], radius=1.0, backend="jakteristics"
    )
    assert "linearity_r1.0" in pc_pair_uls_voxels.real_world.geometric_features
    assert "curvature_r1.0" in pc_pair_uls_voxels.real_world.geometric_features
    assert "planarity_r1.0" in pc_pair_uls_voxels.real_world.geometric_features
    assert "eigenvalue1_r1.0" in pc_pair_uls_voxels.real_world.geometric_features


def test_geometric_features_backend_fallback(pc_pair_uls_voxels):
    with pytest.warns(UserWarning, match="Features .* not available in"):
        pc_pair_uls_voxels.compute_geometric_features(
            feature_names=["linearity", "planarity", "eigenvalue1"],
            radius=1.0,
            backend="pgeof",
        )
    assert "linearity_r1.0" in pc_pair_uls_voxels.real_world.geometric_features
    assert "planarity_r1.0" in pc_pair_uls_voxels.real_world.geometric_features
    assert "eigenvalue1_r1.0" in pc_pair_uls_voxels.real_world.geometric_features


def test_voxel_features_different_voxel_sizes(pc_pair_uls_voxels):
    """Test that different voxel sizes produce different feature values."""
    vapc_1, _ = pc_pair_uls_voxels.compute_voxel_features(
        voxel_size=0.5, feature_names=["density"]
    )

    vapc_2, _ = pc_pair_uls_voxels.compute_voxel_features(
        voxel_size=1.0, feature_names=["density"]
    )

    # Different voxel sizes should generally produce different feature arrays
    feat_1 = pc_pair_uls_voxels.real_world.voxel_features["density_v0.5"][0]
    feat_2 = pc_pair_uls_voxels.real_world.voxel_features["density_v1.0"][0]
    assert feat_1.shape != feat_2.shape or not np.allclose(feat_1, feat_2)


def test_voxel_features_all_supported(pc_pair_uls_voxels):
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

    vapc_real, vapc_sims = pc_pair_uls_voxels.compute_voxel_features(
        voxel_size=0.5, feature_names=all_features
    )

    for feature_name in all_features:
        assert f"{feature_name}_v0.5" in pc_pair_uls_voxels.real_world.voxel_features


def test_voxel_features_empty_list_computes_all(pc_pair_uls_voxels):
    """Test that empty feature list computes all available features."""
    vapc_real, vapc_sims = pc_pair_uls_voxels.compute_voxel_features(
        voxel_size=0.5, feature_names=[]
    )

    # Should compute multiple features when list is empty
    assert len(pc_pair_uls_voxels.real_world.voxel_features) > 1


def test_voxel_features_data_structure(pc_pair_uls_voxels):
    """Test that voxel feature data has correct structure and dimensions."""
    vapc_real, vapc_sims = pc_pair_uls_voxels.compute_voxel_features(
        voxel_size=0.5, feature_names=["count"]
    )

    feat_data = pc_pair_uls_voxels.real_world.voxel_features["count_v0.5"]
    feat_values, voxel_coords, rep_indices = feat_data

    # All components should have matching first dimension (number of voxels)
    assert feat_values.shape[0] == voxel_coords.shape[0]
    assert feat_values.shape[0] == rep_indices.shape[0]

    # Coordinates should be 3D
    assert voxel_coords.shape[1] == 3

    # Representative indices should map to actual points
    assert np.max(rep_indices) < pc_pair_uls_voxels.real_world.coords.shape[0]
    assert np.min(rep_indices) >= 0


def test_filter_by_class(pc_pair_classes):
    pc_pair_filtered = pc_pair_classes.filter_by_class([1])  # filter to only class 1
    assert pc_pair_filtered.real_world.coords is not None
    assert all(sim.coords is not None for sim in pc_pair_filtered.simulated)
    assert np.all(pc_pair_filtered.real_world.classification == 1)
    for sim in pc_pair_filtered.simulated:
        assert np.all(sim.classification == 1)


def test_filter_by_class_missing_classification(pc_pair_uls_voxels):
    pc_pair_uls_voxels.real_world.classification = None
    with pytest.raises(
        ValueError,
        match="Cannot filter by class if one of the point clouds does not have 'classification' attribute.",
    ):
        pc_pair_uls_voxels.filter_by_class([1])
    for sim in pc_pair_uls_voxels.simulated:
        sim.classification = None
    with pytest.raises(
        ValueError,
        match="Cannot filter by class if one of the point clouds does not have 'classification' attribute.",
    ):
        pc_pair_uls_voxels.filter_by_class([1])


def test_filter_by_class_missing_class(pc_pair_classes):
    # check if it raises a warning when no points are found for the specified class
    with pytest.warns(UserWarning):
        pc_pair_classes.filter_by_class([999])  # assuming class 999 does not exist


def test_filter_by_class_preserves_voxel_features(pc_pair_classes):
    pc_pair_classes.compute_voxel_features(voxel_size=1.0, feature_names=["count"])

    pc_pair_filtered = pc_pair_classes.filter_by_class([1])

    for filtered_pc in [pc_pair_filtered.real_world, *pc_pair_filtered.simulated]:
        for feat_name, feat_value in filtered_pc.voxel_features.items():
            feat_data, voxel_coords, rep_indices = feat_value
            assert feat_data.shape[0] == voxel_coords.shape[0]
            assert feat_data.shape[0] == rep_indices.shape[0]
            assert np.all(filtered_pc.classification[rep_indices] == 1)
            assert np.allclose(voxel_coords, filtered_pc.coords[rep_indices])


def test_histogram_per_class_matplotlib(pc_pair_classes):
    from pcfingerprint.vis_matplotlib import plot_histogram_per_class

    # check if histogram per class can be generated without errors
    fig_list, classes = plot_histogram_per_class(
        pc_pair_classes.real_world,
        pc_pair_classes.simulated,
        attribute="intensity",
    )
    num_classes_expected = 4
    assert fig_list is not None
    assert len(fig_list) == num_classes_expected
    assert [0, 1, 2, 3] in classes
