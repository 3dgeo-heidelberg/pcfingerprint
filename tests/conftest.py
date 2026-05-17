from pathlib import Path
import numpy as np
import pytest
import pcfingerprint
import warnings

# create random seed
rng = np.random.default_rng(45621231543)


# Factory helpers to build deterministic PointCloud objects for tests
def make_point_cloud(rng, n_points=100, with_attributes=False):
    coords = rng.random((n_points, 3)) * 10
    pc = pcfingerprint.PointCloud(coords=coords)
    if with_attributes:
        pc.gps_time = np.arange(n_points).astype(float)
        pc.intensity = rng.integers(0, 255, n_points)
        pc.classification = rng.integers(0, 32, n_points)
    return pc


def add_geometric_features(pc, rng):
    """Attach deterministic geometric features to a PointCloud."""
    n = pc.coords.shape[0]
    pc.geometric_features["linearity_r0.05"] = rng.random(n)
    pc.geometric_features["linearity_r0.1"] = rng.random(n)
    pc.geometric_features["linearity_r0.2"] = rng.random(n)
    pc.geometric_features["planarity_r0.05"] = rng.random(n)
    pc.geometric_features["planarity_r0.2"] = rng.random(n)
    pc.geometric_features["density_r0.5"] = rng.random(n)


def add_voxel_features(pc, rng, n_voxels=50):
    """Attach deterministic voxel features (values, coords, rep_indices) to a PointCloud."""
    voxel_coords = rng.random((n_voxels, 3)) * 10
    voxel_indices = rng.integers(0, pc.coords.shape[0], n_voxels)
    pc.voxel_features["linearity_v0.5"] = (
        rng.random(n_voxels),
        voxel_coords,
        voxel_indices,
    )
    pc.voxel_features["density_v0.5"] = (
        rng.random(n_voxels),
        voxel_coords,
        voxel_indices,
    )
    n2 = max(1, n_voxels // 2)
    pc.voxel_features["density_v1.0"] = (
        rng.random(n2),
        rng.random((n2, 3)) * 10,
        rng.integers(0, pc.coords.shape[0], n2),
    )


@pytest.fixture()
def pc_pair_uls_voxels():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_reference("data/uls_tree_voxels/uls_reference.laz")
    pc_pair.load_compare("data/uls_tree_voxels/uls_detailed_voxel_modes_scaled.laz")

    return pc_pair


@pytest.fixture()
def pc_pair_uls_voxels_multiple():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_reference("data/uls_tree_voxels/uls_reference.laz")
    pc_pair.load_compare(
        list(Path("data/uls_tree_voxels").glob("uls_detailed_voxel*.laz"))
    )

    return pc_pair


@pytest.fixture()
def pc_xyz(tmp_path):
    xyz_path = tmp_path / "test_point_cloud.csv"
    # Create a simple point cloud with known coordinates and attributes
    import pandas as pd

    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "z": [0.0, 1.0, 2.0],
            "intensity": [100, 150, 200],
        }
    )
    df.to_csv(xyz_path, index=False)
    return xyz_path


@pytest.fixture()
def pc_pair_with_coords():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.reference = make_point_cloud(rng, n_points=100, with_attributes=False)
    pc_pair.compare = [make_point_cloud(rng, n_points=100, with_attributes=False)]
    return pc_pair


@pytest.fixture()
def pc_pair_with_coords_and_attr(pc_pair_with_coords):
    pc_pair = pc_pair_with_coords
    # Add deterministic attributes to reference and compare point clouds
    pc_pair.reference.gps_time = np.arange(pc_pair.reference.coords.shape[0]).astype(
        float
    )
    pc_pair.reference.intensity = rng.integers(
        0, 255, pc_pair.reference.coords.shape[0]
    )
    pc_pair.reference.classification = rng.integers(
        0, 6, pc_pair.reference.coords.shape[0]
    )
    pc_pair.compare[0].gps_time = np.arange(pc_pair.compare[0].coords.shape[0]).astype(
        float
    )
    pc_pair.compare[0].intensity = rng.integers(
        0, 255, pc_pair.compare[0].coords.shape[0]
    )
    pc_pair.compare[0].classification = rng.integers(
        0, 6, pc_pair.compare[0].coords.shape[0]
    )
    return pc_pair


@pytest.fixture()
def pc_pair_with_coords_and_attr_and_feats(pc_pair_with_coords_and_attr):
    pc_pair = pc_pair_with_coords_and_attr
    # Add deterministic geometric and voxel features using helpers
    add_geometric_features(pc_pair.reference, rng)
    add_geometric_features(pc_pair.compare[0], rng)
    add_voxel_features(pc_pair.reference, rng)
    add_voxel_features(pc_pair.compare[0], rng)
    return pc_pair
