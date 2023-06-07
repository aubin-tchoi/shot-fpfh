from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist


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
