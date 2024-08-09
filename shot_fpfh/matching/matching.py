from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist


def match_descriptors(
    scan_descriptors: npt.NDArray[np.float64],
    ref_descriptors: npt.NDArray[np.float64],
    filter_callback: (
        Callable[[npt.NDArray[np.float64], ...], np.ndarray[bool]] | None
    ) = None,
    filter_nonreciprocal: bool = False,
    verbose: bool = True,
    n_min_matches: int = 100,
    **kwargs: bool | int | float | tuple[float, float],
) -> tuple[np.ndarray[np.int32], np.ndarray[np.int32]]:
    """
    Computes the distance matrix between two sets of descriptors and filters the nearest match found for each descriptor
    of the query set. Can additionally filter out matches that are not reciprocal.
    For infinite norm measure, provide the multiscale descriptors as a (n_scales, n_points, descriptor_length) array.

    Args:
        scan_descriptors: Descriptors computed on the dataset of interest.
        ref_descriptors: Descriptors computed on the reference dataset.
        filter_callback: Filter function to call on the distance vector.
        filter_nonreciprocal: Set to True to filter out non-reciprocal matches.
        verbose: verbosity level.
        n_min_matches: Minimum number of matches below which this function will automatically switch back to keeping
        non-reciprocal matches if enabled.
        kwargs: Arguments to pass to the filter function.

    Returns:
        Indices of the matches established.
    """
    # Euclidian norm
    if len(scan_descriptors.shape) == 2:
        print("\n-- Matching descriptors based on Euclidian-norm proximity --")
        # empty descriptors (ones that did not have a density high enough in the computation of SHOT) are left out
        non_empty_descriptors = np.any(scan_descriptors, axis=1).nonzero()[0]
        non_empty_ref_descriptors = np.any(ref_descriptors, axis=1).nonzero()[0]

        # brute force computation of the distance matrix, faster than a KDTree as the search space has a high dimension
        distance_matrix = cdist(
            scan_descriptors[non_empty_descriptors],
            ref_descriptors[non_empty_ref_descriptors],
        )
        indices = distance_matrix.argmin(axis=1)
        distances = distance_matrix[np.arange(non_empty_descriptors.shape[0]), indices]

        # filter function based on the distance observed, returns a mask
        filtered_indices = (
            filter_callback(distances, **kwargs)
            if filter_callback is not None
            else np.ones(distances.shape[0], dtype=bool)
        )

        # filtering out non-reciprocal matches
        if filter_nonreciprocal:
            reciprocal_matches = distance_matrix.argmin(axis=0)[indices] == np.arange(
                indices.shape[0]
            )
            # applying the mask iff it will lead to more than n_min_matches
            if (
                final_mask := filtered_indices & reciprocal_matches
            ).sum() >= n_min_matches:
                filtered_indices = final_mask
            elif verbose:
                print("Too few reciprocal matches, keeping non-reciprocal matches.")

    # minimum of the Euclidian norm on each scale
    else:
        print("\n-- Matching descriptors based on infinite-norm proximity --")
        max_val = 1000
        n_scales, n_points, descriptor_size = scan_descriptors.shape
        n_points_ref = ref_descriptors.shape[1]

        # we will keep a running minimum distance matrix updated at each scale
        inf_distance_matrix = np.ones((n_points, n_points_ref)) * max_val
        for scale in range(n_scales):
            non_empty_descriptors = np.any(scan_descriptors[scale], axis=1)
            non_empty_ref_descriptors = np.any(ref_descriptors[scale], axis=1)

            # initializing a very high value and replacing only the values for nonempty descriptors
            distance_matrix_scale = np.ones((n_points, n_points_ref)) * max_val
            distance_matrix_on_nonempty = cdist(
                scan_descriptors[scale][non_empty_descriptors],
                ref_descriptors[scale][non_empty_ref_descriptors],
            )
            distance_matrix_scale[
                np.ix_(non_empty_descriptors, non_empty_ref_descriptors)
            ] = distance_matrix_on_nonempty

            # putting back high values on matches that are non-reciprocal
            if filter_nonreciprocal:
                non_reciprocal_matches = distance_matrix_on_nonempty.argmin(axis=0)[
                    distance_matrix_on_nonempty.argmin(axis=1)
                ] != np.arange(non_empty_descriptors.sum())
                # applying the mask
                distance_matrix_scale[non_empty_descriptors][
                    non_reciprocal_matches
                ] = max_val
            # taking the minimum with the current distance matrix
            inf_distance_matrix = np.minimum(distance_matrix_scale, inf_distance_matrix)
        indices = inf_distance_matrix.argmin(axis=1)
        distances = inf_distance_matrix[np.arange(n_points), indices]

        # filtering out values that are equal to max_val, which can come from empty descriptors
        filtered_indices = (
            filter_callback(distances, **kwargs)
            if filter_callback is not None
            else np.ones(distances.shape[0], dtype=bool)
        ) & (distances < max_val)

        if filtered_indices.sum() < n_min_matches and filter_nonreciprocal:
            print("Too few reciprocal matches, keeping non-reciprocal matches.")
            # noinspection PyArgumentEqualDefault
            return match_descriptors(
                scan_descriptors,
                ref_descriptors,
                filter_callback,
                filter_nonreciprocal=False,
                verbose=verbose,
                **kwargs,
            )
        # we do not necessarily need what is below, they are kept to use the same syntax as with the Euclidian norm
        non_empty_descriptors = np.arange(n_points)
        non_empty_ref_descriptors = np.arange(n_points_ref)

    if verbose:
        print(
            f"Kept {filtered_indices.sum()} matches out of {scan_descriptors.shape[-2]} descriptors."
        )

    return (
        non_empty_descriptors[filtered_indices],
        non_empty_ref_descriptors[indices[filtered_indices]],
    )


def basic_matching(
    scan_descriptors: npt.NDArray[np.float64], ref_descriptors: npt.NDArray[np.float64]
) -> tuple[np.ndarray[np.int32], np.ndarray[np.int32]]:
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
    return non_empty_descriptors, non_empty_ref_descriptors[indices]


def double_matching_with_rejects(
    scan_descriptors: npt.NDArray[np.float64],
    ref_descriptors: npt.NDArray[np.float64],
    threshold: float,
    verbose: bool = True,
) -> tuple[np.ndarray[np.int32], np.ndarray[np.int32]]:
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
