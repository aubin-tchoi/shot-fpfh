import numpy as np


def grid_subsampling(
    points: np.ndarray[np.float64], voxel_size: float
) -> np.ndarray[np.int32]:
    """
    Performs a voxel subsampling on the point cloud.
    Keeps the point closest to the barycenter of the points in each voxel.
    """
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(
        ((points - np.min(points, axis=0)) // voxel_size).astype(int),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    idx_pts_vox_sorted = np.argsort(inverse)
    sub_sampled_points_idx = []

    last_seen = 0
    for idx in range(len(non_empty_voxel_keys)):
        indexes_in_voxel = idx_pts_vox_sorted[
            last_seen : last_seen + nb_pts_per_voxel[idx]
        ]
        # point closest to the barycenter
        sub_sampled_points_idx.append(
            indexes_in_voxel[
                np.linalg.norm(
                    points[indexes_in_voxel] - points[indexes_in_voxel].mean(axis=0),
                    axis=1,
                ).argmin()
            ]
        )

        last_seen += nb_pts_per_voxel[idx]

    return np.array(sub_sampled_points_idx)
