"""
Implementation of the ICP method.
"""

import numpy as np
from sklearn.neighbors import KDTree

from .utils import best_rigid_transform, compute_rigid_transform_error


def icp_point_to_point(
    data: np.ndarray,
    ref: np.ndarray,
    max_iter: int,
    rms_threshold: float,
    sampling_limit: int,
) -> np.ndarray:
    """
    Iterative closest point algorithm with a point to point strategy.
    Each iteration is performed on a subsampled of the point clouds to fasten the computation.

    Args:
        data: (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref: (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter: stop condition on the number of iterations
        rms_threshold: stop condition on the distance
        sampling_limit: number of points used at each iteration.

    Returns:
        data_aligned: data aligned on reference cloud
        R_list: list of the (d x d) rotation matrices found at each iteration
        T_list: list of the (d x 1) translation vectors found at each iteration
        neighbors_list: At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
    """
    data_aligned = np.copy(data)
    kdtree = KDTree(ref.T)

    rms = 0.0

    for i in range(max_iter):
        indexes = np.random.choice(data.shape[1], sampling_limit, replace=False)
        data_aligned_subset = data_aligned[:, indexes]
        neighbors = kdtree.query(data_aligned_subset.T, return_distance=False).squeeze()
        rms, data_aligned = compute_rigid_transform_error(
            data_aligned_subset,
            ref[:, neighbors],
            *best_rigid_transform(
                data_aligned_subset,
                ref[:, neighbors],
            ),
        )

        if rms < rms_threshold:
            print("RMS threshold reached.")
            break
    # for ... else clause executed if we did not break out of the loop
    else:
        print("Max iteration number reached.")

    print(f"Final RMS: {rms:.4f}")

    return data_aligned


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
