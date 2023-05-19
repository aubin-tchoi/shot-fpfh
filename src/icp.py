"""
Implementation of the ICP method.
"""

from typing import Tuple

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import trange

from .rigid_transform import solver_point_to_point


def icp_point_to_point_with_sampling(
    scan: np.ndarray[np.float64],
    ref: np.ndarray[np.float64],
    d_max: float,
    max_iter: int = 100,
    rms_threshold: float = 1e-2,
    sampling_limit: int = 100,
    disable_progress_bar: bool = False,
) -> Tuple[np.ndarray[np.float64], float, bool]:
    """
    Iterative closest point algorithm with a point to point strategy.
    Each iteration is performed on a subsampling of the point clouds to fasten the computation.

    Args:
        scan: (N_points x d) matrix where "N_points" is the number of points and "d" the dimension
        ref: (N_ref x d) matrix where "N_ref" is the number of points and "d" the dimension
        d_max: Maximum distance between two points to consider the match as an inlier.
        max_iter: stop condition on the number of iterations
        rms_threshold: stop condition on the distance
        sampling_limit: number of points used at each iteration.
        disable_progress_bar: disables the progress bar.

    Returns:
        points_aligned: points aligned on reference cloud.
        rms: the RMS error.
        has_converged: boolean value indicating whether the method has converged or not.
    """
    points_aligned = np.copy(scan)
    kdtree = KDTree(ref)
    sampling_limit = min(sampling_limit, scan.shape[0])

    rms = 0.0

    for _ in (
        pbar := trange(max_iter, desc="ICP", delay=1, disable=disable_progress_bar)
    ):
        # this loop can be stopped preemptively by a CTRL+C
        try:
            indexes = np.random.choice(scan.shape[0], sampling_limit, replace=False)
            points_aligned_subset = points_aligned[indexes]
            distances, neighbors = kdtree.query(points_aligned_subset)
            # keeping the inlier only
            inlier_points = points_aligned_subset[distances.squeeze() <= d_max]
            neighbors = neighbors[distances <= d_max]
            transformation = solver_point_to_point(
                inlier_points,
                ref[neighbors],
            )
            rms = np.sqrt(
                np.sum(
                    np.linalg.norm(inlier_points - ref[neighbors], axis=1) ** 2,
                    axis=0,
                )
            )
            pbar.set_description(f"ICP - current RMS: {rms:.2f}")
            points_aligned = transformation.transform(points_aligned)
            if rms < rms_threshold:
                break
        except KeyboardInterrupt:
            print("ICP interrupted by user.")
            break

    return points_aligned, rms, rms < rms_threshold


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
