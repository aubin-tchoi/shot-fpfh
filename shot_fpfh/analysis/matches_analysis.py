"""
Functions that can help analyze the matches by parsing the files that can help retrieve the exact transformation between
the point clouds to register.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree

from shot_fpfh.base_computation import Transformation


def get_incorrect_matches(
    scan: np.ndarray[np.float64],
    ref: np.ndarray[np.float64],
    exact_transformation: Transformation,
) -> np.ndarray[bool]:
    """
    Finds the incorrect matches between two sets of points.

    Args:
        scan: data to align.
        ref: reference data.
        exact_transformation: transformation to go from data to ref.

    Returns:
        incorrect_matches: incorrect matches as an array of booleans.
    """
    return (np.linalg.norm(exact_transformation[scan] - ref, axis=1) > 1e-2).astype(
        bool
    )


def plot_distance_hists(
    scan: np.ndarray[np.float64],
    ref: np.ndarray[np.float64],
    exact_transformation: Transformation,
    scan_descriptors: np.ndarray[np.float64],
    ref_descriptors: np.ndarray[np.float64],
) -> None:
    """
    Validates the double_matching_with_rejects by plotting histograms of the ratio between the distances to the nearest
    neighbor and to the second-nearest neighbor in the descriptor space.

    Args:
        scan: data to align.
        ref: reference data.
        exact_transformation: transformation to go from data to ref.
        scan_descriptors: descriptors computed on data.
        ref_descriptors: descriptors computed on ref.
    """
    kdtree = KDTree(ref)
    dist_points, indices_points = kdtree.query(exact_transformation[scan])

    kdtree = KDTree(ref_descriptors)
    distances_desc, indices_desc = kdtree.query(scan_descriptors, 2)
    correct_matches = (indices_desc[:, 0] == indices_points.squeeze()) & (
        dist_points.squeeze() < 1e-2
    )

    # noinspection PyArgumentEqualDefault
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.hist(
        np.divide(
            distances_desc[correct_matches, 0],
            distances_desc[correct_matches, 1],
            out=np.ones(correct_matches.sum()),
            where=distances_desc[correct_matches, 1] > 0,
        ),
        bins=50,
        label="Correct matches",
    )
    ax2.hist(
        np.divide(
            distances_desc[~correct_matches, 0],
            distances_desc[~correct_matches, 1],
            out=np.ones(np.sum(~correct_matches)),
            where=distances_desc[~correct_matches, 1] > 0,
        ),
        bins=50,
        label="Incorrect matches",
    )
    ax1.legend()
    ax2.legend()
    ax1.set(title="Ratio between the nearest neighbor and the second nearest one")
    ax2.set(title="Ratio between the nearest neighbor and the second nearest one")
    plt.show()
