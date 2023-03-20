import numpy as np


def select_query_points(points: np.ndarray) -> np.ndarray:
    """
    Selects a subset of the points to create a set of key points on which descriptors will be computed.
    One possible approach would consist in selecting a point randomly, counting its spherical neighbors as visited,
    removing the visited points from the pool of candidate points and reiterating on it.
    Another possible approach would involve a subsampling of the point cloud where the point closest to the barycenter
    is kept.
    """
    pass
