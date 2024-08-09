"""
RANSAC iterations applied to matches between descriptors to find the transformation that aligns best the keypoints.
"""

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from shot_fpfh.base_computation import Transformation, solver_point_to_point

# setting a seed
rng = np.random.default_rng(seed=72)


def ransac_on_matches(
    scan_descriptors_indices: np.ndarray[np.int32],
    ref_descriptors_indices: np.ndarray[np.int32],
    scan_keypoints: npt.NDArray[np.float64],
    ref_keypoints: npt.NDArray[np.float64],
    n_draws: int = 10000,
    draw_size: int = 4,
    distance_threshold: float = 1,
    verbose: bool = False,
    disable_progress_bar: bool = False,
) -> tuple[float, Transformation]:
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
                    transformation[scan_keypoints[scan_descriptors_indices]]
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
