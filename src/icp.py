"""
Implementation of the ICP method.
"""

from typing import Tuple

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import trange

from .rigid_transform import best_rigid_transform


def icp_point_to_point(
    data: np.ndarray,
    ref: np.ndarray,
    max_iter: int = 10000,
    rms_threshold: float = 0.01,
    sampling_limit: int = 100,
) -> Tuple[np.ndarray, float, bool]:
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
        rms: the RMS error.
        has_converged: boolean value indicating whether the method has converged or not.
    """
    data_aligned = np.copy(data)
    kdtree = KDTree(ref)
    sampling_limit = min(sampling_limit, data.shape[0])

    rms = 0.0

    for _ in trange(max_iter, desc="ICP", delay=1):
        # this loop can be stopped preemptively by a CTRL+C
        try:
            indexes = np.random.choice(data.shape[0], sampling_limit, replace=False)
            data_aligned_subset = data_aligned[indexes]
            neighbors = kdtree.query(data_aligned_subset, return_distance=False).squeeze()
            rotation, translation = best_rigid_transform(
                data_aligned_subset,
                ref[neighbors],
            )
            rms = np.sqrt(
                np.sum(
                    (data_aligned_subset - ref[neighbors]) ** 2,
                    axis=0,
                ).mean()
            )
            data_aligned = data_aligned.dot(rotation.T) + translation
            if rms < rms_threshold:
                break
        except KeyboardInterrupt:
            print("ICP interrupted by user.")
            break

    return data_aligned, rms, rms < rms_threshold


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
