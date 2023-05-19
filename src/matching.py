from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from .perf_monitoring import timeit
from .rigid_transform import solver_point_to_point
from .transformation import Transformation

# setting a seed
rng = np.random.default_rng(seed=1)


@timeit
def basic_matching(
    scan_descriptors: np.ndarray[np.float64], ref_descriptors: np.ndarray[np.float64]
) -> Tuple[np.ndarray[np.int32], np.ndarray[np.int32]]:
    """
    Matching strategy that matches each descriptor with its nearest neighbor in the feature space.

    Args:
        scan_descriptors: Descriptors computed on the dataset of interest.
        ref_descriptors: Descriptors computed on the reference dataset.

    Returns:
        Indices of the matches established.
    """
    non_empty_descriptors = np.any(scan_descriptors, axis=1).nonzero()[0]
    non_empty_ref_descriptors = np.any(ref_descriptors, axis=1).nonzero()[0]
    distance_matrix = cdist(
        scan_descriptors[non_empty_descriptors],
        ref_descriptors[non_empty_ref_descriptors],
    )
    indices = distance_matrix.argmin(axis=1)
    return (
        non_empty_descriptors.nonzero()[0],
        non_empty_ref_descriptors[indices],
    )


@timeit
def double_matching_with_rejects(
    scan_descriptors: np.ndarray[np.float64],
    ref_descriptors: np.ndarray[np.float64],
    threshold: float,
    verbose: bool = True,
) -> Tuple[np.ndarray[np.int32], np.ndarray[np.int32]]:
    """
    Matching strategy that establishes point-to-point correspondences between descriptors and rejects matches where
    the ratio between the distance to the closest neighbor and to the second-closest neighbor is below a threshold.
    The motivation between this strategy is that for false matches, there will likely be a number of other false matches
    within similar distances due to the high dimensionality of the feature space.
    TODO: try out different distances such as the chi^2 statistic.

    Args:
        scan_descriptors: Descriptors computed on the dataset of interest.
        ref_descriptors: Descriptors computed on the reference dataset.
        threshold: Threshold for rejection of incorrect matches.
        verbose: Adds verbosity to the execution of the function.

    Returns:
        matches_indices: Indices of the matches established in the initial array.
        matches_indices_ref: Indices of the matches established in the reference array.
    """
    non_empty_descriptors = np.any(scan_descriptors, axis=1).nonzero()[0]
    non_empty_ref_descriptors = np.any(ref_descriptors, axis=1).nonzero()[0]
    distance_matrix = cdist(
        scan_descriptors[non_empty_descriptors],
        ref_descriptors[non_empty_ref_descriptors],
    )
    indices = np.argpartition(distance_matrix, kth=2, axis=1)
    neighbor_distances = distance_matrix[np.arange(scan_descriptors.shape[0]), indices]
    mask = (
        np.divide(
            neighbor_distances[:, 0],
            neighbor_distances[:, 1],
            out=np.ones(scan_descriptors.shape[0]),
            where=neighbor_distances[:, 1] != 0,
        )
        >= threshold
    )

    if verbose:
        print(
            f"Kept {mask.sum()} matches out of {scan_descriptors.shape[0]} descriptors."
        )

    return (
        non_empty_descriptors[mask.nonzero()[0]],
        non_empty_ref_descriptors[neighbor_distances[mask, 0]],
    )


@timeit
def ransac_on_matches(
    scan_descriptors_indices: np.ndarray[np.int32],
    ref_descriptors_indices: np.ndarray[np.int32],
    scan_keypoints: np.ndarray[np.float64],
    ref_keypoints: np.ndarray[np.float64],
    n_draws: int = 10000,
    draw_size: int = 4,
    distance_threshold: float = 1,
    verbose: bool = False,
    disable_progress_bar: bool = False,
) -> Tuple[float, Transformation]:
    """
    Matching strategy that establishes point-to-point correspondences between descriptors and performs RANSAC-type
    iterations to find the best rigid transformation between the two point clouds based on random picks of the matches.
    Only works if the biggest cluster of consistent matches (matches that can all be laid on top of one another by a
    common rigid transform) contains good matches.

    Returns:
        Rotation and translation of the rigid transform to perform on the point cloud.
    """
    best_n_inliers: int | None = None
    best_transform: Transformation | None = None

    for _ in (
        pbar := tqdm(
            range(n_draws),
            desc="RANSAC",
            total=n_draws,
            disable=disable_progress_bar,
            delay=0.5,
        )
    ):
        try:
            draw = rng.choice(
                scan_descriptors_indices.shape[0],
                draw_size,
                replace=False,
                shuffle=False,
            )
            transformation = solver_point_to_point(
                scan_keypoints[scan_descriptors_indices[draw]],
                ref_keypoints[ref_descriptors_indices[draw]],
            )
            n_inliers = (
                np.linalg.norm(
                    transformation.transform(scan_keypoints[scan_descriptors_indices])
                    - ref_keypoints[ref_descriptors_indices],
                    axis=1,
                )
                <= distance_threshold
            ).sum()
            if best_n_inliers is None or n_inliers > best_n_inliers:
                if verbose:
                    print(
                        f"Updating best n_inliers from {best_n_inliers} to {n_inliers}"
                    )
                best_n_inliers = n_inliers
                best_transform = transformation
            pbar.set_description(f"RANSAC - current best n_inliers: {best_n_inliers}")
        except KeyboardInterrupt:
            print("RANSAC interrupted by user.")
            break

    best_transform.normalize_rotation()

    return best_n_inliers / scan_descriptors_indices.shape[0], best_transform
