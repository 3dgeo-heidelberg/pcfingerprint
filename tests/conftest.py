from pathlib import Path
import pytest
import pcfingerprint


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
def pc_pair_classes():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_reference("data/forest/Plot_10_original_test.laz")
    pc_pair.load_compare("data/forest/Plot_10_simulated_test.laz")

    return pc_pair


@pytest.fixture()
def pc_pair_with_features():
    """Create a simple PointCloudPair with known geometric and voxel features for testing attribute expansion."""
    import numpy as np

    # Create reference point cloud with known features
    coords_ref = np.random.rand(100, 3) * 10
    pc_ref = pcfingerprint.PointCloud(coords=coords_ref)
    pc_ref.gps_time = np.arange(100).astype(float)
    pc_ref.intensity = np.random.randint(0, 255, 100)
    pc_ref.classification = np.random.randint(0, 32, 100)

    # Add geometric features with different radii
    pc_ref.geometric_features["linearity_r0.05"] = np.random.rand(100)
    pc_ref.geometric_features["linearity_r0.1"] = np.random.rand(100)
    pc_ref.geometric_features["linearity_r0.2"] = np.random.rand(100)
    pc_ref.geometric_features["planarity_r0.05"] = np.random.rand(100)
    pc_ref.geometric_features["planarity_r0.2"] = np.random.rand(100)
    pc_ref.geometric_features["density_r0.5"] = np.random.rand(100)

    # Add voxel features with different voxel sizes
    voxel_coords = np.random.rand(50, 3) * 10
    voxel_indices = np.random.randint(0, 100, 50)
    pc_ref.voxel_features["linearity_v0.5"] = (
        np.random.rand(50),
        voxel_coords,
        voxel_indices,
    )
    pc_ref.voxel_features["density_v0.5"] = (
        np.random.rand(50),
        voxel_coords,
        voxel_indices,
    )
    pc_ref.voxel_features["density_v1.0"] = (
        np.random.rand(30),
        np.random.rand(30, 3) * 10,
        np.random.randint(0, 100, 30),
    )

    # Create compare point cloud with same features
    coords_cmp = np.random.rand(100, 3) * 10
    pc_cmp = pcfingerprint.PointCloud(coords=coords_cmp)
    pc_cmp.gps_time = np.arange(100).astype(float)
    pc_cmp.intensity = np.random.randint(0, 255, 100)
    pc_cmp.classification = np.random.randint(0, 32, 100)

    pc_cmp.geometric_features["linearity_r0.05"] = np.random.rand(100)
    pc_cmp.geometric_features["linearity_r0.1"] = np.random.rand(100)
    pc_cmp.geometric_features["linearity_r0.2"] = np.random.rand(100)
    pc_cmp.geometric_features["planarity_r0.05"] = np.random.rand(100)
    pc_cmp.geometric_features["planarity_r0.2"] = np.random.rand(100)
    pc_cmp.geometric_features["density_r0.5"] = np.random.rand(100)

    pc_cmp.voxel_features["linearity_v0.5"] = (
        np.random.rand(50),
        voxel_coords,
        voxel_indices,
    )
    pc_cmp.voxel_features["density_v0.5"] = (
        np.random.rand(50),
        voxel_coords,
        voxel_indices,
    )
    pc_cmp.voxel_features["density_v1.0"] = (
        np.random.rand(30),
        np.random.rand(30, 3) * 10,
        np.random.randint(0, 100, 30),
    )

    # Create and return PointCloudPair
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.reference = pc_ref
    pc_pair.compare = [pc_cmp]

    return pc_pair
