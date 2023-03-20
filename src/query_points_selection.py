import numpy as np

# setting a seed
rng = np.random.default_rng(seed=1)


def select_query_points(points: np.ndarray) -> np.ndarray:
    """
    Selects a subset of the points to create a set of key points on which descriptors will be computed.
    One possible approach would consist in selecting a point randomly, counting its spherical neighbors as visited,
    removing the visited points from the pool of candidate points and reiterating on it.
    Another possible approach would involve a subsampling of the point cloud where the point closest to the barycenter
    is kept.
    """
    pass


def select_query_points_randomly(
    points: np.ndarray, n_feature_points: int
) -> np.ndarray:
    """
    Selects a random subset of the points to create a set of key points on which descriptors will be computed.
    """
    return rng.choice(points, n_feature_points, replace=False, shuffle=False)


def select_query_indices_randomly(
    n_points: int, n_feature_points: int
) -> np.ndarray:
    """
    Selects a random subset of the points to create a set of key points on which descriptors will be computed.
    """
    return np.random.choice(n_points, n_feature_points, replace=False)

