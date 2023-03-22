"""
Implementation of the ICP method.
"""

from typing import Tuple

import numpy as np
from sklearn.neighbors import KDTree

from .utils import best_rigid_transform, compute_rigid_transform_error


def icp_point_to_point(
    data: np.ndarray,
    ref: np.ndarray,
    max_iter: int = 1000,
    rms_threshold: float = 0.1,
    sampling_limit: int = 10,
) -> Tuple[np.ndarray, bool]:
    """
    Iterative closest point algorithm with a point to point strategy.
    Each iteration is performed on a subsampled of the point clouds to fasten the computation.

    Args:
        data: (N_data x d) matrix where "N_data" is the number of points and "d" the dimension
        ref: (N_ref x d) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter: stop condition on the number of iterations
        rms_threshold: stop condition on the distance
        sampling_limit: number of points used at each iteration.

    Returns:
        data_aligned: data aligned on reference cloud.
        has_converged: boolean value indicating whether the method has converged or not.
    """
    data_aligned = np.copy(data)
    kdtree = KDTree(ref.T)

    rms = 0.0

    for i in range(max_iter):
        indexes = np.random.choice(data.shape[0], sampling_limit, replace=False)
        data_aligned_subset = data_aligned[indexes]
        neighbors = kdtree.query(data_aligned_subset, return_distance=False).squeeze()
        rms, data_aligned = compute_rigid_transform_error(
            data_aligned_subset,
            ref[neighbors],
            *best_rigid_transform(
                data_aligned_subset,
                ref[neighbors],
            ),
        )

        if rms < rms_threshold:
            print("RMS threshold reached.")
            break
    # for ... else clause executed if we did not break out of the loop
    else:
        print("Max iteration number reached.")

    print(f"Final RMS: {rms:.4f}")

    return data_aligned, rms < rms_threshold


def icp_point_to_plane(
    points: np.ndarray,
    normals: np.ndarray,
    ref: np.ndarray,
    ref_normals: np.ndarray,
    max_iter: int,
    rms_threshold: float,
) -> np.ndarray:
    """
    Point to plane ICP.
    More robust to point clouds of variable densities where the plane estimations by the normals are good.
    """
    pass
