from pathlib import Path
import pytest
import pcfingerprint


@pytest.fixture()
def pc_pair_uls_voxels():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_real_world("data/uls_tree_voxels/uls_reference.laz")
    pc_pair.load_simulated("data/uls_tree_voxels/uls_detailed_voxel_modes_scaled.laz")

    return pc_pair


@pytest.fixture()
def pc_pair_uls_voxels_multiple():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_real_world("data/uls_tree_voxels/uls_reference.laz")
    pc_pair.load_simulated(
        list(Path("data/uls_tree_voxels").glob("uls_detailed_voxel*.laz"))
    )

    return pc_pair


@pytest.fixture()
def pc_pair_classes():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_real_world("data/forest/Plot_10_original_test.laz")
    pc_pair.load_simulated("data/forest/Plot_10_simulated_test.laz")

    return pc_pair
