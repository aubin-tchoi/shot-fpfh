from typing import Tuple

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import trange

from .perf_monitoring import timeit
from .utils import best_rigid_transform, compute_rigid_transform_error

# setting a seed
rng = np.random.default_rng(seed=1)


@timeit
def basic_matching(descriptors: np.ndarray, ref_descriptors: np.ndarray) -> np.ndarray:
    """
    Matching strategy that matches each descriptor with its nearest neighbor in the feature space.

    Args:
        descriptors: Descriptors computed on the dataset of interest.
        ref_descriptors: Descriptors computed on the reference dataset.

    Returns:
        Indices of the matches established.

    """
    distances = np.zeros((descriptors.shape[0], ref_descriptors.shape[0]))
    for i in trange(descriptors.shape[0], desc="Distances computed", delay=1):
        for j in range(ref_descriptors.shape[0]):
            distances[i, j] = np.linalg.norm(descriptors[i] - ref_descriptors[j])

    return distances.argmin(axis=1)


@timeit
def double_matching_with_rejects(
    descriptors: np.ndarray, ref_descriptors: np.ndarray, threshold: float
) -> np.ndarray:
    """
    Matching strategy that establishes point-to-point correspondences between descriptors and rejects matches where
    the ratio between the distance to the closest neighbor and to the second-closest neighbor is below a threshold.
    The motivation between this strategy is that for false matches there will likely be a number of other false matches
    within similar distances due to the high dimensionality of the feature space.

    Args:
        descriptors: Descriptors computed on the dataset of interest.
        ref_descriptors: Descriptors computed on the reference dataset.
        threshold: Threshold for rejection of incorrect matches.

    Returns:
        Indices of the matches established.
    """
    kdtree = KDTree(ref_descriptors)
    distances, matches = kdtree.query(descriptors, 2)

    return matches[distances[:, 0] / distances[:, 1] >= threshold]


@timeit
def ransac_matching(
    point_cloud: np.ndarray,
    descriptors: np.ndarray,
    ref_descriptors: np.ndarray,
    ref_point_cloud: np.ndarray,
    n_draws: int = 100,
    draw_size: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matching strategy that establishes point-to-point correspondences between descriptors and performs RANSAC-type
    iterations to find the best rigid transformation between the two point clouds based on random picks of the matches.

    Returns:
        Rotation and translation of the rigid transform to perform on the point cloud.
    """
    kdtree = KDTree(ref_descriptors)
    matches = kdtree.query(descriptors, return_distance=False).squeeze()

    best_rms: float | None = None
    best_transform: Tuple[np.ndarray, np.ndarray] | None = None

    for _ in range(n_draws):
        draws = rng.choice(matches, draw_size, replace=False, shuffle=False)
        rotation, translation = best_rigid_transform(
            point_cloud[draws],
            ref_point_cloud[draws],
        )
        rms = compute_rigid_transform_error(
            point_cloud, ref_point_cloud, rotation, translation
        )[0]
        if best_rms is None or rms < best_rms:
            best_rms = rms
            best_transform = (rotation, translation)

    return best_transform
