"""
Implementation of the ICP method.
"""
import numpy as np
from sklearn.neighbors import KDTree
from tqdm import trange

from base_computation import (
    solver_point_to_point,
    solver_point_to_plane,
    Transformation,
    grid_subsampling,
)


def icp_point_to_point_with_sampling(
    scan: np.ndarray[np.float64],
    ref: np.ndarray[np.float64],
    d_max: float,
    max_iter: int = 100,
    rms_threshold: float = 1e-2,
    sampling_limit: int = 100,
    disable_progress_bar: bool = False,
) -> tuple[np.ndarray[np.float64], float, bool]:
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
            points_aligned = transformation[points_aligned]
            if rms < rms_threshold:
                break
        except KeyboardInterrupt:
            print("ICP interrupted by user.")
            break

    return points_aligned, rms, rms < rms_threshold


def icp_point_to_point(
    scan: np.ndarray[np.float64],
    ref: np.ndarray[np.float64],
    transformation_init: Transformation,
    d_max: float,
    voxel_size: float = 0.2,
    max_iter: int = 100,
    rms_threshold: float = 1e-2,
    disable_progress_bar: bool = False,
) -> tuple[Transformation, float, bool]:
    """
    Iterative closest point algorithm with a point to point strategy.
    Each iteration is performed on a subsampling of the point clouds to fasten the computation.
    """
    kdtree = KDTree(ref)
    subsampled_indices = grid_subsampling(scan, voxel_size)
    transformation_icp = transformation_init
    rms = 0.0

    for _ in (
        progress_bar := trange(
            max_iter, desc="ICP", delay=1, disable=disable_progress_bar
        )
    ):
        # this loop can be stopped preemptively by a CTRL+C
        try:
            points_aligned = transformation_icp[scan[subsampled_indices]]
            distances, neighbors = kdtree.query(points_aligned)

            # keeping the inliers only
            inliers = points_aligned[distances.squeeze() <= d_max]
            inliers_neighbors = neighbors[distances <= d_max]

            # finding the transformation between the inliers and their neighbors
            transformation_aligned_to_ref = solver_point_to_point(
                inliers,
                ref[inliers_neighbors],
            )
            rms = np.sqrt(
                np.sum(
                    np.linalg.norm(inliers - ref[neighbors], axis=1) ** 2,
                    axis=0,
                )
            )
            transformation_icp = transformation_aligned_to_ref @ transformation_icp
            progress_bar.set_description(f"ICP - current RMS: {rms:.2f}")
            if rms < rms_threshold:
                print("RMS threshold reached.")
                break
        except KeyboardInterrupt:
            print("ICP interrupted by user.")
            break

    return transformation_icp, rms, rms < rms_threshold


def icp_point_to_plane(
    scan: np.ndarray[np.float64],
    ref: np.ndarray[np.float64],
    ref_normals: np.ndarray[np.float64],
    transformation_init: Transformation,
    d_max: float,
    voxel_size: float = 0.2,
    max_iter: int = 50,
    rms_threshold: float = 1e-2,
    disable_progress_bar: bool = False,
) -> tuple[Transformation, float, bool]:
    """
    Point to plane ICP.
    More robust to point clouds of variable densities where the plane estimations by the normals are good.
    """
    kdtree = KDTree(ref)
    subsampled_indices = grid_subsampling(scan, voxel_size)
    transformation_icp = transformation_init
    rms = 0.0

    for _ in (
        progress_bar := trange(
            max_iter, desc="ICP", delay=1, disable=disable_progress_bar
        )
    ):
        # this loop can be stopped preemptively by a CTRL+C
        try:
            points_aligned = transformation_icp[scan[subsampled_indices]]
            distances, neighbors = kdtree.query(points_aligned)

            # keeping the inliers only
            inliers = points_aligned[distances.squeeze() <= d_max]
            inliers_neighbors = neighbors[distances <= d_max]

            # finding the transformation between the inliers and their neighbors
            transformation_aligned_to_ref = solver_point_to_plane(
                inliers,
                ref[inliers_neighbors],
                ref_normals[inliers_neighbors],
            )
            transformation_icp = transformation_aligned_to_ref @ transformation_icp
            rms = np.abs(
                np.einsum(
                    "ij, ij->i",
                    inliers - ref[inliers_neighbors],
                    ref_normals[inliers_neighbors],
                )
            ).mean(axis=0)
            progress_bar.set_description(f"ICP - current RMS: {rms:.2f}")
            if rms < rms_threshold:
                print("RMS threshold reached.")
                break
        except KeyboardInterrupt:
            print("ICP interrupted by user.")
            break

    return transformation_icp, rms, rms < rms_threshold
