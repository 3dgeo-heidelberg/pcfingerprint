import numpy as np
import pytest

import pcfingerprint
from pcfingerprint.chunking import generate_chunk_tiles, run_chunked_computation

rng = np.random.default_rng(918273465)


def test_generate_chunk_tiles_covers_every_point_exactly_once():
    coords = rng.random((500, 3)) * 10
    tiles = generate_chunk_tiles(coords, max_points_per_chunk=50, buffer=0.5)

    seen = np.zeros(coords.shape[0], dtype=int)
    for tile in tiles:
        interior_indices = tile["buffered_indices"][tile["interior_mask"]]
        seen[interior_indices] += 1
        # every buffered tile must contain its own interior points
        assert set(interior_indices).issubset(set(tile["buffered_indices"]))

    assert np.all(seen == 1)


def test_generate_chunk_tiles_buffer_includes_neighbors_near_edges():
    # points on a grid, so we know exactly which points are within `radius`
    # of any given point
    xs, ys = np.meshgrid(np.arange(20), np.arange(20))
    coords = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1).astype(float)
    radius = 1.5
    tiles = generate_chunk_tiles(coords, max_points_per_chunk=30, buffer=radius)

    for tile in tiles:
        buffered_indices = tile["buffered_indices"]
        interior_indices = buffered_indices[tile["interior_mask"]]
        buffered_set = set(buffered_indices)
        for idx in interior_indices:
            dists = np.linalg.norm(coords - coords[idx], axis=1)
            neighbor_indices = np.flatnonzero(dists <= radius)
            assert buffered_set.issuperset(neighbor_indices)


def test_generate_chunk_tiles_single_chunk_when_below_threshold():
    coords = rng.random((50, 3)) * 10
    tiles = generate_chunk_tiles(coords, max_points_per_chunk=1000, buffer=0.5)
    assert len(tiles) == 1
    assert tiles[0]["interior_mask"].all()
    assert len(tiles[0]["buffered_indices"]) == 50


@pytest.mark.parametrize(
    "feature_names",
    [(["linearity"]), (["linearity", "planarity"])],
    ids=["single_feat", "multiple_feats"],
)
def test_run_chunked_computation_matches_unchunked_pgeof(feature_names):
    from pcfingerprint.geometric_features import compute_geom_feat_pgeof

    coords = rng.random((400, 3)) * 10
    radius = 1.0

    pc_full = pcfingerprint.PointCloud(coords=coords.copy())
    compute_geom_feat_pgeof(pc_full, feature_names, radius)

    pc_chunked = pcfingerprint.PointCloud(coords=coords.copy())
    run_chunked_computation(
        pc_chunked,
        compute_geom_feat_pgeof,
        max_points_per_chunk=80,
        buffer=radius,
        feature_names=feature_names,
        radius=radius,
    )

    for key in pc_full.geometric_features:
        np.testing.assert_allclose(
            pc_full.geometric_features[key],
            pc_chunked.geometric_features[key],
            atol=1e-5,
        )


@pytest.mark.parametrize(
    "feature_names",
    [(["linearity"]), (["linearity", "planarity"])],
    ids=["single_feat", "multiple_feats"],
)
def test_run_chunked_computation_matches_unchunked_jakteristics(feature_names):
    from pcfingerprint.geometric_features import compute_geom_feat_jakteristics

    coords = rng.random((400, 3)) * 10
    radius = 1.0

    pc_full = pcfingerprint.PointCloud(coords=coords.copy())
    compute_geom_feat_jakteristics(pc_full, feature_names, radius)

    pc_chunked = pcfingerprint.PointCloud(coords=coords.copy())
    run_chunked_computation(
        pc_chunked,
        compute_geom_feat_jakteristics,
        max_points_per_chunk=80,
        buffer=radius,
        feature_names=feature_names,
        radius=radius,
    )

    for key in pc_full.geometric_features:
        np.testing.assert_allclose(
            pc_full.geometric_features[key],
            pc_chunked.geometric_features[key],
            atol=1e-5,
        )


def test_compute_geometric_features_chunking_matches_unchunked(pc_pair_with_coords):
    import copy

    pair_unchunked = pc_pair_with_coords
    pair_chunked = copy.deepcopy(pc_pair_with_coords)

    # single feature name, see NOTE in test_run_chunked_computation_matches_unchunked_pgeof
    pair_unchunked.compute_geometric_features(
        feature_names=["linearity"], radius=1.0, backend="pgeof"
    )
    pair_chunked.compute_geometric_features(
        feature_names=["linearity"],
        radius=1.0,
        backend="pgeof",
        chunking=True,
        max_points_per_chunk=20,
    )

    for key in pair_unchunked.reference.geometric_features:
        np.testing.assert_allclose(
            pair_unchunked.reference.geometric_features[key],
            pair_chunked.reference.geometric_features[key],
            atol=1e-5,
        )


def test_compute_geometric_features_chunking_requires_max_points(pc_pair_with_coords):
    with pytest.raises(ValueError, match="max_points_per_chunk"):
        pc_pair_with_coords.compute_geometric_features(
            feature_names=["linearity"], radius=1.0, chunking=True
        )
