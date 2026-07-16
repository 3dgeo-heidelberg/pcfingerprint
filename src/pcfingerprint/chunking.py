"""
Spatial chunking helpers for running the single-cloud geometric feature
functions in geometric_features.py over memory-constrained large point
clouds.

A point cloud is recursively split by x/y (or x/y/z if its vertical extent
is large relative to its horizontal extent) at the median coordinate until
every tile's interior point count is at or below a threshold. Each tile is
buffered along its split axes by at least the search radius, so a spherical
neighborhood search run on the buffered tile sees the same neighbors near
tile edges as it would on the full point cloud. Only the non-overlap
interior points of each tile are kept when merging results back.
"""

from types import SimpleNamespace
import numpy as np
import logging

logger = logging.getLogger("pcfingerprint")


def _decide_split_axes(coords, vertical_extent_ratio=0.5):
    extents = coords.max(axis=0) - coords.min(axis=0)
    horizontal_extent = max(extents[0], extents[1])
    if horizontal_extent > 0 and extents[2] / horizontal_extent > vertical_extent_ratio:
        return [0, 1, 2]
    return [0, 1]


def _recursive_split(indices, coords, axes, axis_pos, max_points_per_chunk):
    if indices.size <= max_points_per_chunk:
        yield indices
        return
    axis = axes[axis_pos % len(axes)]
    vals = coords[indices, axis]
    median = np.median(vals)
    left_mask = vals <= median
    if left_mask.all() or not left_mask.any():
        # degenerate split (e.g. all points share the same coordinate on this axis)
        yield indices
        return
    yield from _recursive_split(
        indices[left_mask], coords, axes, axis_pos + 1, max_points_per_chunk
    )
    yield from _recursive_split(
        indices[~left_mask], coords, axes, axis_pos + 1, max_points_per_chunk
    )


def generate_chunk_tiles(
    coords, max_points_per_chunk, buffer, vertical_extent_ratio=0.5
):
    """
    Recursively partition a point cloud into buffered spatial tiles.

    :param coords: Nx3 array of point coordinates.
    :type coords: np.ndarray
    :param max_points_per_chunk: Maximum number of interior points per tile.
    :type max_points_per_chunk: int
    :param buffer: Buffer distance added to each tile along its split axes.
        Should be at least the search radius used for feature computation.
    :type buffer: float
    :param vertical_extent_ratio: Ratio of z-extent to horizontal extent
        above which tiles are also split along z.
    :type vertical_extent_ratio: float

    :return: List of dicts with keys "buffered_indices" (indices into
        ``coords`` selecting the tile including its buffer) and
        "interior_mask" (boolean mask into "buffered_indices" marking the
        tile's non-overlap interior points).
    :rtype: list[dict]
    """
    n_points = coords.shape[0]
    if n_points == 0:
        return []

    axes = _decide_split_axes(coords, vertical_extent_ratio)
    all_indices = np.arange(n_points)

    tiles = []
    for interior_indices in _recursive_split(
        all_indices, coords, axes, 0, max_points_per_chunk
    ):
        tile_coords = coords[interior_indices]
        buffered_min = tile_coords.min(axis=0)
        buffered_max = tile_coords.max(axis=0)
        for axis in axes:
            buffered_min[axis] -= buffer
            buffered_max[axis] += buffer

        in_buffer = np.all((coords >= buffered_min) & (coords <= buffered_max), axis=1)
        buffered_indices = np.flatnonzero(in_buffer)
        interior_mask = np.isin(buffered_indices, interior_indices)
        tiles.append(
            {"buffered_indices": buffered_indices, "interior_mask": interior_mask}
        )
    return tiles


def run_chunked_computation(
    point_cloud,
    compute_fn,
    max_points_per_chunk,
    buffer,
    vertical_extent_ratio=0.5,
    **compute_fn_kwargs,
):
    """
    Run a single-cloud geometric feature function from geometric_features.py
    over spatial tiles of ``point_cloud`` and merge the per-point results
    back into ``point_cloud.geometric_features`` in original point order.

    :param point_cloud: PointCloud object with ``coords`` set.
    :param compute_fn: Single-cloud feature function (e.g.
        ``compute_geom_feat_pgeof``) called as
        ``compute_fn(tile, **compute_fn_kwargs)`` for each tile, where
        ``tile`` exposes ``coords`` and ``geometric_features``.
    :param max_points_per_chunk: Maximum number of interior points per tile.
    :type max_points_per_chunk: int
    :param buffer: Buffer distance for each tile; should be at least the
        search radius used by ``compute_fn``.
    :type buffer: float
    :param vertical_extent_ratio: see :func:`generate_chunk_tiles`.
    :param compute_fn_kwargs: Keyword arguments forwarded to ``compute_fn``.
    """
    n_points = point_cloud.coords.shape[0]
    tiles = generate_chunk_tiles(
        point_cloud.coords, max_points_per_chunk, buffer, vertical_extent_ratio
    )

    logger.info(
        f"Chunked computation using {len(tiles)} tiles with max. {max_points_per_chunk} points."
    )

    merged_features = {}
    for tile in tiles:
        buffered_indices = tile["buffered_indices"]
        interior_mask = tile["interior_mask"]

        tile_pc = SimpleNamespace(
            coords=point_cloud.coords[buffered_indices], geometric_features={}
        )
        compute_fn(tile_pc, **compute_fn_kwargs)

        interior_indices = buffered_indices[interior_mask]
        for feat_name, feat_data in tile_pc.geometric_features.items():
            if feat_name not in merged_features:
                merged_features[feat_name] = np.empty(n_points, dtype=feat_data.dtype)
            merged_features[feat_name][interior_indices] = feat_data[interior_mask]

    point_cloud.geometric_features.update(merged_features)
