from dataclasses import dataclass
from multiprocessing import Pool
from types import TracebackType

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import KDTree
from tqdm import tqdm

from shot_fpfh.base_computation import grid_subsampling

from .shot import compute_single_shot_descriptor, get_local_rf


@dataclass
class ShotMultiprocessor:
    """
    Base class to compute SHOT descriptors in parallel on multiple processes.
    """

    normalize: bool = True
    share_local_rfs: bool = True
    min_neighborhood_size: int = 100

    n_procs: int = 8
    disable_progress_bar: bool = False
    verbose: bool = True

    def __enter__(self):
        self.pool = Pool(processes=self.n_procs)
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.pool.terminate()
        else:
            self.pool.close()
        self.pool.join()

    def compute_local_rf(
        self,
        keypoints: npt.NDArray[np.float64],
        neighborhoods: np.ndarray[np.object_],
        support: npt.NDArray[np.float64],
        radius: float,
    ) -> npt.NDArray[np.float64]:
        """
        Parallelization of the function get_local_rf.

        Args:
            support: The supporting point cloud.
            keypoints: The keypoints to compute local reference frames on.
            radius: The radius used to compute the local reference frames.
            neighborhoods: The neighborhoods associated with each keypoint. neighborhoods[i] should be an array of ints.

        Returns:
            The local reference frames computed on every keypoint.
        """
        return np.array(
            list(
                tqdm(
                    self.pool.imap(
                        get_local_rf,
                        [
                            (
                                keypoint,
                                support[neighborhoods[i]],
                                radius,
                            )
                            for i, keypoint in enumerate(keypoints)
                        ],
                    ),
                    desc=f"Local RFs with radius {radius}",
                    total=keypoints.shape[0],
                    disable=self.disable_progress_bar,
                )
            )
        )

    def compute_descriptor(
        self,
        keypoints: npt.NDArray[np.float64],
        normals: npt.NDArray[np.float64],
        neighborhoods: np.ndarray[np.object_],
        local_rfs: npt.NDArray[np.float64],
        support: npt.NDArray[np.float64],
        radius: float,
    ) -> npt.NDArray[np.float64]:
        """
        Parallelization of the function compute_single_shot_descriptor.

        Args:
            keypoints: The keypoints to compute descriptors on.
            normals: The normals of points in the support.
            neighborhoods: The neighborhoods associated with each keypoint. neighborhoods[i] should be an array of ints.
            local_rfs: The local reference frames associated with each keypoint.
            support: The supporting point cloud.
            radius: The radius used to compute SHOT.

        Returns:
            The descriptor computed on every keypoint.
        """
        return np.array(
            list(
                tqdm(
                    self.pool.imap(
                        compute_single_shot_descriptor,
                        [
                            (
                                keypoint,
                                support[neighborhoods[i]],
                                normals[neighborhoods[i]],
                                radius,
                                local_rfs[i],
                                self.normalize,
                                self.min_neighborhood_size,
                            )
                            for i, keypoint in enumerate(keypoints)
                        ],
                        chunksize=int(np.ceil(keypoints.shape[0] / (2 * self.n_procs))),
                    ),
                    desc=f"SHOT desc with radius {radius}",
                    total=keypoints.shape[0],
                    disable=self.disable_progress_bar,
                )
            )
        )

    def compute_descriptor_single_scale(
        self,
        point_cloud: npt.NDArray[np.float64],
        normals: npt.NDArray[np.float64],
        keypoints: npt.NDArray[np.float64],
        radius: float,
        subsampling_voxel_size: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Computes the SHOT descriptor on a single scale.
        Normals are expected to be normalized to 1.

        Args:
            point_cloud: The entire point cloud.
            normals: The normals computed on the point cloud.
            keypoints: The keypoints to compute descriptors on.
            radius: Radius used to compute the SHOT descriptors.
            subsampling_voxel_size: Subsampling strength. Leave empty to keep the whole support.

        Returns:
            The descriptor as a (self.keypoints.shape[0], 352) array.
        """
        support = (
            grid_subsampling(point_cloud, subsampling_voxel_size)
            if subsampling_voxel_size is not None
            else None
        )
        if self.verbose and support is not None:
            print(
                f"Keeping a support of {support.shape[0]} points out of {point_cloud.shape[0]} "
                f"(voxel size: {subsampling_voxel_size:.2f})"
            )
        neighborhoods = KDTree(
            point_cloud[support] if support is not None else point_cloud
        ).query_radius(keypoints, radius)
        local_rfs = self.compute_local_rf(
            keypoints=keypoints,
            neighborhoods=neighborhoods,
            support=point_cloud[support] if support is not None else point_cloud,
            radius=radius,
        )
        return self.compute_descriptor(
            keypoints=keypoints,
            normals=normals[support] if support is not None else normals,
            neighborhoods=neighborhoods,
            local_rfs=local_rfs,
            radius=radius,
            support=point_cloud[support] if support is not None else point_cloud,
        )

    def compute_descriptor_bi_scale(
        self,
        point_cloud: npt.NDArray[np.float64],
        normals: npt.NDArray[np.float64],
        keypoints: npt.NDArray[np.float64],
        local_rf_radius: float,
        shot_radius: float,
        subsampling_voxel_size: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Computes the SHOT descriptor on a point cloud with two distinct radii: one for the computation of the local
        reference frames and the other one for the computation of the descriptor.
        Normals are expected to be normalized to 1.

        Args:
            point_cloud: The entire point cloud.
            normals: The normals computed on the point cloud.
            keypoints: The keypoints to compute descriptors on.
            local_rf_radius: Radius used to compute the local reference frames.
            shot_radius: Radius used to compute the SHOT descriptors.
            subsampling_voxel_size: Subsampling strength. Leave empty to keep the whole support.

        Returns:
            The descriptor as a (self.keypoints.shape[0], 352) array.
        """
        support = (
            grid_subsampling(point_cloud, subsampling_voxel_size)
            if subsampling_voxel_size is not None
            else None
        )
        if self.verbose and support is not None:
            print(
                f"Keeping a support of {support.shape[0]} points out of {point_cloud.shape[0]} "
                f"(voxel size: {subsampling_voxel_size})"
            )
        neighborhoods = KDTree(
            point_cloud[support] if support is not None else point_cloud
        ).query_radius(keypoints, local_rf_radius)
        local_rfs = self.compute_local_rf(
            keypoints=keypoints,
            neighborhoods=neighborhoods,
            support=point_cloud[support] if support is not None else point_cloud,
            radius=local_rf_radius,
        )
        neighborhoods = KDTree(point_cloud[support]).query_radius(
            keypoints, shot_radius
        )
        return self.compute_descriptor(
            keypoints=keypoints,
            normals=normals[support] if support is not None else normals,
            neighborhoods=neighborhoods,
            local_rfs=local_rfs,
            radius=shot_radius,
            support=point_cloud[support] if support is not None else point_cloud,
        )

    def compute_descriptor_multiscale(
        self,
        point_cloud: npt.NDArray[np.float64],
        normals: npt.NDArray[np.float64],
        keypoints: npt.NDArray[np.float64],
        radii: list[float] | npt.NDArray[np.float64],
        voxel_sizes: list[float] | npt.NDArray[np.float64] | None = None,
        weights: list[float] | npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Computes the SHOT descriptor on multiple scales.
        Normals are expected to be normalized to 1.

        Args:
            point_cloud: The entire point cloud.
            normals: The normals computed on the point cloud.
            keypoints: The keypoints to compute descriptors on.
            radii: The radii to compute the descriptors with.
            voxel_sizes: The voxel sizes used to subsample the support. Leave empty to keep the whole support.
            weights: The weights to multiply each scale with. Leave empty to multiply by 1.

        Returns:
            The descriptor as a (self.keypoints.shape[0], 352 * n_scales) array.
        """
        if weights is None:
            weights = np.ones(len(radii))

        all_descriptors = np.zeros((len(radii), keypoints.shape[0], 352))

        local_rfs = None
        for scale, radius in enumerate(radii):
            # choosing the support point cloud
            support = (
                grid_subsampling(point_cloud, voxel_sizes[scale])
                if voxel_sizes is not None
                else None
            )
            if self.verbose and support is not None:
                print(
                    f"Keeping a support of {support.shape[0]} points out of {point_cloud.shape[0]} "
                    f"(voxel size: {voxel_sizes[scale]})"
                )
            neighborhoods = KDTree(
                point_cloud[support] if support is not None else point_cloud
            ).query_radius(keypoints, radius)

            # if shared, only using the smallest radius to determine the local RF
            if local_rfs is None or not self.share_local_rfs:
                # recomputing the local rfs if not shared
                local_rfs = self.compute_local_rf(
                    keypoints=keypoints,
                    neighborhoods=neighborhoods,
                    support=(
                        point_cloud[support] if support is not None else point_cloud
                    ),
                    radius=radius,
                )
            all_descriptors[scale, :, :] = (
                self.compute_descriptor(
                    keypoints=keypoints,
                    normals=normals[support] if support is not None else normals,
                    neighborhoods=neighborhoods,
                    local_rfs=local_rfs,
                    radius=radius,
                    support=(
                        point_cloud[support] if support is not None else point_cloud
                    ),
                )
                * weights[scale]
            )

        return all_descriptors.reshape(keypoints.shape[0], 352 * len(radii))
