"""
Implementation of the ICP method.
"""
from typing import Tuple

import numpy as np
from sklearn.neighbors import KDTree


def best_rigid_transform(
    data: np.ndarray, ref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    """

    data_barycenter = data.mean(axis=1)
    ref_barycenter = ref.mean(axis=1)
    covariance_matrix = (data - data_barycenter[:, np.newaxis]).squeeze() @ (
        ref - ref_barycenter[:, np.newaxis]
    ).squeeze().T
    u, sigma, v = np.linalg.svd(covariance_matrix)
    rotation = v.T @ u.T

    # ensuring that we have a direct rotation (determinant equal to 1 and not -1)
    if np.linalg.det(rotation) < 0:
        u_transpose = u.T
        u_transpose[-1] *= -1
        rotation = v.T @ u_transpose

    translation = ref_barycenter - rotation.dot(data_barycenter)

    return rotation, translation


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
        R, T = best_rigid_transform(
            data_aligned_subset,
            ref[:, neighbors],
        )
        data_aligned = R.dot(data_aligned) + T[:, np.newaxis]
        rms = np.sqrt(
            np.sum(
                (data_aligned_subset - ref[:, neighbors]) ** 2,
                axis=0,
            ).mean()
        )

        if rms < rms_threshold:
            print("RMS threshold reached.")
            break
    # for ... else clause executed if we did not break out of the loop
    else:
        print("Max iteration number reached.")

    print(f"Final RMS: {rms:.4f}")

    return data_aligned
