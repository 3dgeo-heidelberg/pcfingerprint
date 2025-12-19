from pathlib import Path
import pcfingerprint
import numpy as np
import pytest


def test_load_point_clouds():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_real_world("data/uls_reference.laz")
    pc_pair.load_simulated("data/uls_detailed_voxel_modes_scaled.laz")
    assert pc_pair.real_world.coords is not None
    assert all(sim.coords is not None for sim in pc_pair.simulated)


def test_load_point_clouds_multiple():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_real_world("data/uls_reference.laz")
    pc_pair.load_simulated(list(Path("data").glob("uls_detailed_voxel*.laz")))
    assert pc_pair.real_world is not None
    assert pc_pair.simulated is not None


def test_only_overlapping_attributes():
    pc_pair = pcfingerprint.PointCloudPair()
    pc_pair.load_real_world("data/uls_reference.laz")
    pc_pair.load_simulated("data/uls_detailed_voxel_modes_scaled.laz")
    
    for attr in [
        "gps_time",
        "return_number",
        "number_of_returns",
        "intensity",
    ]:
        real_attr = getattr(pc_pair.real_world, attr)
        sim_attr = getattr(pc_pair.simulated[0], attr)
        if real_attr is None or sim_attr is None:
            assert getattr(pc_pair.real_world, attr) is None
            assert getattr(pc_pair.simulated[0], attr) is None


def test_create_pointcloud_numpy():
    points = np.random.rand(100, 3)
    number_of_returns = np.random.randint(1, 5, size=100)
    gpstime = np.arange(100).astype(float) / 10.0
    pc = pcfingerprint.PointCloud(coords=points, gps_time=gpstime)
    pc.number_of_returns = number_of_returns
    assert pc.coords.shape == (100, 3)
    assert pc.gps_time.shape == (100,)
    assert pc.number_of_returns.shape == (100,)

