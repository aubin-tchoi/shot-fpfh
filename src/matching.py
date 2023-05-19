from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.neighbors import KDTree

from .perf_monitoring import timeit
from .rigid_transform import solver_point_to_point, compute_point_to_point_error
from .transformation import Transformation
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
    return KDTree(ref_descriptors).query(descriptors, return_distance=False)


@timeit
def double_matching_with_rejects(
    descriptors: np.ndarray,
    ref_descriptors: np.ndarray,
    threshold: float,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matching strategy that establishes point-to-point correspondences between descriptors and rejects matches where
    the ratio between the distance to the closest neighbor and to the second-closest neighbor is below a threshold.
    The motivation between this strategy is that for false matches there will likely be a number of other false matches
    within similar distances due to the high dimensionality of the feature space.

    Args:
        descriptors: Descriptors computed on the dataset of interest.
        ref_descriptors: Descriptors computed on the reference dataset.
        threshold: Threshold for rejection of incorrect matches.
        verbose: Adds verbosity to the execution of the function.

    Returns:
        matches_indices: Indices of the matches established in the initial array.
        matches_indices_ref: Indices of the matches established in the reference array.
    """
    neighbors_distances, neighbors_indices = KDTree(ref_descriptors).query(
        descriptors, 2
    )
    mask = (
        np.divide(
            neighbors_distances[:, 0],
            neighbors_distances[:, 1],
            out=np.ones(descriptors.shape[0]),
            where=neighbors_distances[:, 1] != 0,
        )
        >= threshold
    )

    if verbose:
        print(f"Kept {mask.sum()} matches out of {descriptors.shape[0]} descriptors.")

    return mask.nonzero()[0], neighbors_indices[mask, 0]


@timeit
def ransac_matching(
    descriptors: np.ndarray,
    point_cloud_whole: np.ndarray,
    point_cloud_subset: np.ndarray,
    ref_descriptors: np.ndarray,
    ref_point_cloud_whole: np.ndarray,
    ref_point_cloud_subset: np.ndarray,
    n_draws: int = 100,
    draw_size: int = 4,
) -> Tuple[float, Transformation]:
    """
    Matching strategy that establishes point-to-point correspondences between descriptors and performs RANSAC-type
    iterations to find the best rigid transformation between the two point clouds based on random picks of the matches.

    Returns:
        Rotation and translation of the rigid transform to perform on the point cloud.
    """
    matches = KDTree(ref_descriptors).query(descriptors, return_distance=False)

    best_rms: float | None = None
    best_transform: Transformation | None = None

    for _ in range(n_draws):
        draws = rng.choice(matches, draw_size, replace=False, shuffle=False)
        transformation = solver_point_to_point(
            point_cloud_subset[draws],
            ref_point_cloud_subset[draws],
        )
        rms = compute_point_to_point_error(
            point_cloud_whole, ref_point_cloud_whole, transformation
        )[0]
        if best_rms is None or rms < best_rms:
            best_rms = rms
            best_transform = transformation

    return best_rms, best_transform
