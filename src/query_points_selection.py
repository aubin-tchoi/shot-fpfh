import numpy as np
from sklearn.neighbors import KDTree

# setting a seed
rng = np.random.default_rng(seed=1)


def select_query_points_iteratively(points: np.ndarray, radius: float) -> np.ndarray:
    """
    Selects a subset of the points to create a set of key points on which descriptors will be computed.
    Operates by selecting a point randomly, counting its spherical neighbors as visited,
    removing the visited points from the pool of candidate points and reiterating on it.

    Returns:
        selected: array of the indices of the selected points.
    """
    selected = np.zeros(points.shape[0], dtype=bool)
    visited = np.zeros(points.shape[0], dtype=bool)
    kdtree = KDTree(points)

    while not visited.all():
        point_idx = (~visited).nonzero()[0][0]
        selected[point_idx] = True
        visited[kdtree.query_radius(points[point_idx], radius)] = True

    return selected.nonzero()[0]


def select_query_points_subsampling(
    points: np.ndarray, voxel_size: float
) -> np.ndarray:
    """
    Selects a subset of the points to create a set of key points on which descriptors will be computed.
    Operates by subsampling the point cloud and taking the points closest to the barycenter of each voxel is kept.
    """
    pass


def select_query_points_randomly(
    points: np.ndarray, n_feature_points: int
) -> np.ndarray:
    """
    Selects a random subset of the points to create a set of key points on which descriptors will be computed.
    """
    return rng.choice(points, n_feature_points, replace=False, shuffle=False)


def select_query_indices_randomly(n_points: int, n_feature_points: int) -> np.ndarray:
    """
    Selects a random subset of the points to create a set of key points on which descriptors will be computed.
    """
    return np.random.choice(n_points, n_feature_points, replace=False)
