"""
Generic pipeline with open choices for the algorithms used to select keypoints and filter matches.
"""
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, List, Union

import numpy as np

from base_computation import Transformation
from utils import checkpoint
from .analysis import get_incorrect_matches, plot_distance_hists
from .descriptors import compute_sphericity
from .icp import icp_point_to_point, icp_point_to_plane
from .keypoint_selection import (
    select_query_indices_randomly,
    select_keypoints_iteratively,
    select_keypoints_subsampling,
    select_keypoints_with_density_threshold,
    filter_keypoints,
)
from .matching import (
    basic_matching,
    double_matching_with_rejects,
    match_descriptors,
    threshold_filter,
    ransac_on_matches,
)
from .shot_parallelization import ShotMultiprocessor


@dataclass
class RegistrationPipeline:
    """
    Generic class for descriptor-based registration between local maps.
    Allows for a selection among a variety of algorithms for keypoint selection, matching, and ICP.
    """

    scan: np.ndarray[np.float64]
    scan_normals: np.ndarray[np.float64]
    ref: np.ndarray[np.float64]
    ref_normals: np.ndarray[np.float64]

    scan_keypoints: Optional[np.ndarray[np.int32]] = None
    ref_keypoints: Optional[np.ndarray[np.int32]] = None

    scan_descriptors: Optional[np.ndarray[np.float64]] = None
    ref_descriptors: Optional[np.ndarray[np.float64]] = None

    matches: Optional[Tuple[np.ndarray[np.int32], np.ndarray[np.int32]]] = None

    def select_keypoints(
        self,
        selection_algorithm: Literal[
            "random", "iterative", "subsampling", "subsampling_with_density"
        ],
        *,
        neighborhood_size: Optional[float] = None,
        min_n_neighbors: Optional[int] = None,
        proportion_picked: float = 1.0,
        force_recompute: bool = False,
    ) -> None:
        """
        Selects keypoints within each of the two point clouds.

        Args:
            selection_algorithm: The algorithm to use for the selection.
            neighborhood_size: The size of the spheres to use for the iterative and the subsampling-based methods.
            min_n_neighbors: Minimum number of neighbors in the subsampling-based method with a density threshold.
            proportion_picked: The proportion of points randomly picked with the random selection algorithm.
            force_recompute: Whether the keypoints should be recomputed even if already present.
        """
        if selection_algorithm == "random":
            print("\n-- Selecting keypoints randomly --")
            assert 0 <= proportion_picked <= 1, "Incorrect proportion passed."
            if self.scan_keypoints is None or force_recompute:
                self.scan_keypoints = select_query_indices_randomly(
                    self.scan.shape[0], int(self.scan.shape[0] * proportion_picked)
                )
            if self.ref_keypoints is None or force_recompute:
                self.ref_keypoints = select_query_indices_randomly(
                    self.ref.shape[0], int(self.ref.shape[0] * proportion_picked)
                )
        elif selection_algorithm == "iterative":
            print("\n-- Selecting keypoints iteratively --")
            if self.scan_keypoints is None or force_recompute:
                self.scan_keypoints = select_keypoints_iteratively(
                    self.scan, neighborhood_size
                )
            if self.ref_keypoints is None or force_recompute:
                self.ref_keypoints = select_keypoints_iteratively(
                    self.ref, neighborhood_size
                )
        elif selection_algorithm == "subsampling":
            print(
                "\n-- Selecting keypoints based on a subsampling of the point cloud --"
            )
            if self.scan_keypoints is None or force_recompute:
                self.scan_keypoints = select_keypoints_subsampling(
                    self.scan, neighborhood_size
                )
            if self.ref_keypoints is None or force_recompute:
                self.ref_keypoints = select_keypoints_subsampling(
                    self.ref, neighborhood_size
                )
        elif selection_algorithm == "subsampling_with_density":
            print(
                "\n-- Selecting keypoints based on a subsampling of the point cloud --"
            )
            if self.scan_keypoints is None or force_recompute:
                self.scan_keypoints = select_keypoints_with_density_threshold(
                    self.scan, neighborhood_size, min_n_neighbors
                )
            if self.ref_keypoints is None or force_recompute:
                self.ref_keypoints = select_keypoints_with_density_threshold(
                    self.ref, neighborhood_size, min_n_neighbors
                )
        else:
            raise ValueError("Incorrect keypoint selection algorithm.")

    def apply_filter_on_keypoints(
        self,
        *,
        normals_z_threshold: float = 0.8,
        sphericity_threshold: float = 0.16,
        sphericity_computation_radius: float = 0.3,
        verbose: bool = True,
    ) -> None:
        """
        Filters the keypoints based on a filter on the normals and on the sphericity.

        Args:
            normals_z_threshold: Threshold on the value of the z component of the normals. Values beneath this threshold
            will be kept.
            sphericity_threshold: Threshold on the value of the sphericity. Values beneath this threshold will be kept.
            sphericity_computation_radius: Radius used in the computation of the sphericity.
            verbose: Verbosity level.
        """
        print("\n-- Filtering the keypoints --")
        if self.scan_keypoints is not None and self.scan_keypoints.shape[0] > 0:
            self.scan_keypoints = filter_keypoints(
                self.scan_keypoints,
                self.scan_normals,
                normals_z_threshold,
                compute_sphericity(
                    self.scan[self.scan_keypoints],
                    self.scan,
                    sphericity_computation_radius,
                ),
                sphericity_threshold,
                verbose,
            )
        else:
            print("No scan keypoint to filter.")
        if self.ref_keypoints is not None and self.ref_keypoints.shape[0] > 0:
            self.ref_keypoints = filter_keypoints(
                self.ref_keypoints,
                self.ref_normals,
                normals_z_threshold,
                compute_sphericity(
                    self.ref[self.ref_keypoints],
                    self.ref,
                    sphericity_computation_radius,
                ),
                sphericity_threshold,
                verbose,
            )
        else:
            print("No ref keypoint to filter.")

    def compute_shot_descriptor_single_scale(
        self,
        radius: float,
        subsampling_voxel_size: Optional[float] = None,
        force_recompute: bool = False,
        **shot_multiprocessor_config: Union[bool, int],
    ) -> None:
        """
        Computes the SHOT descriptor on a single scale.
        Normals are expected to be normalized to 1.

        Args:
            radius: Radius used to compute the SHOT descriptors.
            subsampling_voxel_size: Subsampling strength. Leave empty to keep the whole support.
            force_recompute: Whether the descriptors should be recomputed even if already present.
            shot_multiprocessor_config: Additional parameters to pass to ShotMultiprocessor (all have default values).

        Returns:
            The descriptor as a (self.keypoints.shape[0], 352) array.
        """
        print("\n-- Computing single-scale SHOT descriptors --")
        with ShotMultiprocessor(**shot_multiprocessor_config) as shot_multiprocessor:
            if self.scan_descriptors is None or force_recompute:
                self.scan_descriptors = (
                    shot_multiprocessor.compute_descriptor_single_scale(
                        point_cloud=self.scan,
                        keypoints=self.scan[self.scan_keypoints],
                        normals=self.scan_normals,
                        radius=radius,
                        subsampling_voxel_size=subsampling_voxel_size,
                    )
                )
            if self.ref_descriptors is None or force_recompute:
                self.ref_descriptors = (
                    shot_multiprocessor.compute_descriptor_single_scale(
                        point_cloud=self.ref,
                        keypoints=self.ref[self.ref_keypoints],
                        normals=self.ref_normals,
                        radius=radius,
                        subsampling_voxel_size=subsampling_voxel_size,
                    )
                )

    def compute_shot_descriptor_bi_scale(
        self,
        local_rf_radius: float,
        shot_radius: float,
        subsampling_voxel_size: Optional[float] = None,
        force_recompute: bool = False,
        **shot_multiprocessor_config: Union[bool, int],
    ) -> np.ndarray[np.float64]:
        """
        Computes the SHOT descriptor on a point cloud with two distinct radii: one for the computation of the local
        reference frames and the other one for the computation of the descriptor.
        Normals are expected to be normalized to 1.

        Args:
            local_rf_radius: Radius used to compute the local reference frames.
            shot_radius: Radius used to compute the SHOT descriptors.
            subsampling_voxel_size: Subsampling strength. Leave empty to keep the whole support.
            force_recompute: Whether the descriptors should be recomputed even if already present.
            shot_multiprocessor_config: Additional parameters to pass to ShotMultiprocessor (all have default values).

        Returns:
            The descriptor as a (self.keypoints.shape[0], 352) array.
        """
        print("\n-- Computing SHOT descriptors with two scales (local RF and SHOT) --")
        with ShotMultiprocessor(**shot_multiprocessor_config) as shot_multiprocessor:
            if self.scan_descriptors is None or force_recompute:
                self.scan_descriptors = shot_multiprocessor.compute_descriptor_bi_scale(
                    point_cloud=self.scan,
                    keypoints=self.scan[self.scan_keypoints],
                    normals=self.scan_normals,
                    local_rf_radius=local_rf_radius,
                    shot_radius=shot_radius,
                    subsampling_voxel_size=subsampling_voxel_size,
                )
            if self.ref_descriptors is None or force_recompute:
                self.ref_descriptors = shot_multiprocessor.compute_descriptor_bi_scale(
                    point_cloud=self.ref,
                    keypoints=self.ref[self.ref_keypoints],
                    normals=self.ref_normals,
                    local_rf_radius=local_rf_radius,
                    shot_radius=shot_radius,
                    subsampling_voxel_size=subsampling_voxel_size,
                )

    def compute_shot_descriptor_multiscale(
        self,
        radii: Union[List[float], np.ndarray[np.float64]],
        voxel_sizes: Optional[Union[List[float], np.ndarray[np.float64]]] = None,
        weights: Optional[Union[List[float], np.ndarray[np.float64]]] = None,
        force_recompute: bool = False,
        **shot_multiprocessor_config: Union[bool, int],
    ) -> np.ndarray[np.float64]:
        """
        Computes the SHOT descriptor on multiple scales.
        Normals are expected to be normalized to 1.

        Args:
            radii: The radii to compute the descriptors with.
            voxel_sizes: The voxel sizes used to subsample the support. Leave empty to keep the whole support.
            weights: The weights to multiply each scale with. Leave empty to multiply by 1.
            force_recompute: Whether the descriptors should be recomputed even if already present.
            shot_multiprocessor_config: Additional parameters to pass to ShotMultiprocessor (all have default values).

        Returns:
            The descriptor as a (self.keypoints.shape[0], 352 * n_scales) array.
        """
        print("\n-- Computing multi-scale SHOT descriptors --")
        with ShotMultiprocessor(**shot_multiprocessor_config) as shot_multiprocessor:
            if self.scan_descriptors is None or force_recompute:
                self.scan_descriptors = (
                    shot_multiprocessor.compute_descriptor_multiscale(
                        point_cloud=self.scan,
                        keypoints=self.scan[self.scan_keypoints],
                        normals=self.scan_normals,
                        radii=radii,
                        voxel_sizes=voxel_sizes,
                        weights=weights,
                    )
                )
            if self.ref_descriptors is None or force_recompute:
                self.ref_descriptors = (
                    shot_multiprocessor.compute_descriptor_multiscale(
                        point_cloud=self.ref,
                        keypoints=self.ref[self.ref_keypoints],
                        normals=self.ref_normals,
                        radii=radii,
                        voxel_sizes=voxel_sizes,
                        weights=weights,
                    )
                )

    def find_descriptors_matches(
        self,
        matching_algorithm: Literal["simple", "double", "threshold"],
        *,
        reject_threshold: float,
        threshold_multiplier: float,
        debug_mode: bool = False,
        force_recompute: bool = False,
    ) -> None:
        """
        Matches two sets of descriptors for both the point cloud to align and the reference point cloud.

        Args:
            matching_algorithm: Choice of the algorithm employed.
            reject_threshold: Threshold used by the 'double' method.
            threshold_multiplier: Base multiplier used to compute the threshold in the threshold-based method.
            debug_mode: Enables a debug mode that additionally prints the maximum distance between matched descriptors.
            force_recompute: Whether the descriptors should be recomputed even if already present.
        """
        timer = checkpoint()
        if matching_algorithm == "simple":
            print(
                "\n-- Matching descriptors using simple matching to closest neighbor --"
            )
            if self.matches is None or force_recompute:
                self.matches = basic_matching(
                    self.scan_descriptors, self.ref_descriptors
                )
            timer("Time spent finding matches between the SHOT descriptors")

        elif matching_algorithm == "double":
            print("\n-- Matching descriptors using double matching with rejects --")
            if self.matches is None or force_recompute:
                self.matches = double_matching_with_rejects(
                    self.scan_descriptors, self.ref_descriptors, reject_threshold
                )
            timer("Time spent finding matches between the SHOT descriptors")

        elif matching_algorithm == "threshold":
            print("\n-- Matching descriptors using threshold-based matching --")
            if self.matches is None or force_recompute:
                self.matches = match_descriptors(
                    self.scan_descriptors,
                    self.ref_descriptors,
                    threshold_filter,
                    threshold_multiplier=threshold_multiplier,
                )
            timer("Time spent finding matches between the SHOT descriptors")

        else:
            raise ValueError("Incorrect matching algorithm selection.")

        if debug_mode:
            max_dist = np.linalg.norm(
                self.scan_descriptors[self.matches[0]]
                - self.ref_descriptors[self.matches[1]],
                axis=0,
            ).max(initial=0)
            print(
                f"Maximum distance between the descriptors matched: " f"{max_dist:.2f}"
            )

    def analyze_matches(
        self,
        matching_algorithm: Literal["simple", "double", "threshold"],
        exact_transformation: Transformation,
    ) -> None:
        """
        Analyzes the matches based on prior knowledge on the exact transformation between the two point clouds.

        Args:
            matching_algorithm: Algorithm used for the matching. An additional analysis of the Lowe criterion is
            performed with the 'double' algorithm.
            exact_transformation: The exact 4x4 transformation that registers the two point clouds.
        """
        incorrect_matches_shot = get_incorrect_matches(
            self.scan_keypoints[self.matches[0]],
            self.ref_keypoints[self.matches[1]],
            exact_transformation,
        )
        print(
            f"SHOT: {incorrect_matches_shot.sum()} incorrect matches out of {self.matches[0].shape[0]} matches and "
            f"{self.scan_descriptors.shape[0]} descriptors."
        )
        if matching_algorithm == "double":
            plot_distance_hists(
                self.scan_keypoints,
                self.ref_keypoints,
                exact_transformation,
                self.scan_descriptors,
                self.ref_descriptors,
            )

    def run_ransac(
        self,
        *,
        n_draws: int = 10000,
        draw_size: int = 4,
        max_inliers_distance: float = 2,
        transformation: Optional[Transformation] = None,
        disable_progress_bar: bool = False,
    ) -> Tuple[Transformation, float]:
        """
        Performs RANSAC-like draws to match the descriptors.

        Args:
            n_draws: Number of draws.
            draw_size: Number of descriptors picked in each draw.
            max_inliers_distance: Threshold on the distance to consider two points as a pair of inliers.
            transformation: The exact transformation that registers the two point clouds (only used for analysis).
            disable_progress_bar: Whether the progress bar on the iterations of RANSAC should be disabled.

        Returns:
            The resulting transformation, and the ratio of inliers on the keypoints.
        """
        print("\n -- Aligning the point clouds by RANSAC-ing the matches --")
        (
            inliers_ratio_shot,
            transformation_shot,
        ) = ransac_on_matches(
            *self.matches,
            self.scan[self.scan_keypoints],
            self.ref[self.ref_keypoints],
            n_draws=n_draws,
            draw_size=draw_size,
            distance_threshold=max_inliers_distance,
            disable_progress_bar=disable_progress_bar,
        )
        if transformation is not None:
            print(
                f"Norm of the angle between the two rotations: "
                f"{abs(np.arccos((np.trace(transformation_shot.rotation @ transformation.rotation.T) - 1) / 2)):.2f}"
                f"\nNorm of the difference between the two translation: "
                f"{np.linalg.norm(transformation_shot.translation - transformation.translation):.2f}"
            )

        return transformation_shot, inliers_ratio_shot

    def run_icp(
        self,
        icp_type: Literal["point_to_point", "point_to_plane"],
        transformation_init: Transformation,
        *,
        d_max: float,
        voxel_size: float = 0.2,
        max_iter: int = 100,
        rms_threshold: float = 1e-2,
        disable_progress_bar: bool = False,
    ) -> Tuple[Transformation, float, bool]:
        """
        Performs an ICP for fine registration between scan and ref.

        Args:
            icp_type: The type of ICP to perform. Possible choices are point-to-point and point-to-plane.
            transformation_init: The initial transformation to start from.
            d_max: Threshold on the distance to consider two points as a pair of inliers.
            voxel_size: Size of the voxels in the subsampling performed to select a subset of the points for the ICP.
            max_iter: Maximum number of iterations.
            rms_threshold: Threshold that ends the method upon passing.
            disable_progress_bar: Whether the progress bar should be disabled.

        Returns:
            The resulting transformation, the residual error and whether the ICP has converged.
        """
        if icp_type == "point_to_point":
            print("\n-- Running point-to-point ICP --")
            return icp_point_to_point(
                self.scan,
                self.ref,
                transformation_init,
                d_max=d_max,
                voxel_size=voxel_size,
                max_iter=max_iter,
                rms_threshold=rms_threshold,
                disable_progress_bar=disable_progress_bar,
            )
        elif icp_type == "point_to_plane":
            print("\n-- Running point-to-plane ICP --")
            return icp_point_to_plane(
                self.scan,
                self.ref,
                self.ref_normals,
                transformation_init,
                d_max=d_max,
                voxel_size=voxel_size,
                max_iter=max_iter,
                rms_threshold=rms_threshold,
                disable_progress_bar=disable_progress_bar,
            )
        else:
            raise ValueError("Incorrect ICP type selected.")
