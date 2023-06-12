import numpy as np
from sklearn.neighbors import KDTree

from base_computation import grid_subsampling

# setting a seed
rng = np.random.default_rng(seed=1)


def select_keypoints_iteratively(
    points: np.ndarray[np.float64], radius: float
) -> np.ndarray[np.int32]:
    """
    Selects a subset of the points to create a set of key points on which descriptors will be computed.
    Operates by selecting a point randomly, counting its spherical neighbors as visited,
    removing the visited points from the pool of candidate points and reiterating on it.

    Returns:
        selected: array containing the indices of the selected points.
    """
    selected = np.zeros(points.shape[0], dtype=bool)
    visited = np.zeros(points.shape[0], dtype=bool)
    kdtree = KDTree(points)

    while not visited.all():
        point_idx = (~visited).nonzero()[0][0]
        selected[point_idx] = True
        visited[kdtree.query_radius([points[point_idx]], radius)[0]] = True

    return selected.nonzero()[0]


def select_keypoints_subsampling(
    points: np.ndarray[np.float64], voxel_size: float
) -> np.ndarray[np.int32]:
    """
    Selects a subset of the points to create a set of key points on which descriptors will be computed.
    Operates by subsampling the point cloud and keeping the points closest to the barycenter of each voxel.

    Returns:
        selected key points: array containing the indices of the selected points.
    """
    return grid_subsampling(points, voxel_size)


def select_keypoints_randomly(
    points: np.ndarray[np.float64], n_feature_points: int
) -> np.ndarray[np.int32]:
    """
    Selects a random subset of the points to create a set of key points on which descriptors will be computed.
    """
    return rng.choice(points, n_feature_points, replace=False, shuffle=False)


def select_query_indices_randomly(
    n_points: int, n_feature_points: int
) -> np.ndarray[np.int32]:
    """
    Selects a random subset of the points to create a set of key points on which descriptors will be computed.
    """
    return np.random.choice(n_points, n_feature_points, replace=False)


def select_keypoints_with_density_threshold(
    points: np.ndarray[np.float64],
    voxel_size: float,
    density_threshold_value: int,
    density_threshold_radius: float | None = None,
) -> np.ndarray[np.int32]:
    """
    Selects a subset of the points to create a set of key points on which descriptors will be computed.
    Operates by subsampling the point cloud and keeping the points closest to the barycenter of each voxel whose density
    exceeds a certain value.

    Returns:
        selected keypoints: array containing the indices of the selected points.
    """
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(
        ((points - np.min(points, axis=0)) // voxel_size).astype(int),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if density_threshold_radius is None:
        density_threshold_radius = voxel_size

    idx_pts_vox_sorted = np.argsort(inverse)
    sub_sampled_points_idx = []
    kdtree = None
    if density_threshold_radius != voxel_size:
        kdtree = KDTree(points)

    last_seen = 0
    for idx in range(len(non_empty_voxel_keys)):
        indexes_in_voxel = idx_pts_vox_sorted[
            last_seen : last_seen + nb_pts_per_voxel[idx]
        ]
        # point closest to the barycenter
        point_closest_to_centroid = indexes_in_voxel[
            np.linalg.norm(
                points[indexes_in_voxel] - points[indexes_in_voxel].mean(axis=0),
                axis=1,
            ).argmin()
        ]
        if (
            voxel_size == density_threshold_radius
            and nb_pts_per_voxel[idx] > density_threshold_value
        ) or (
            voxel_size != density_threshold_radius
            and (
                kdtree.query_radius(
                    [points[point_closest_to_centroid]], density_threshold_radius
                )[0].shape[0]
                > density_threshold_value
            )
        ):
            sub_sampled_points_idx.append(point_closest_to_centroid)

        last_seen += nb_pts_per_voxel[idx]

    return np.array(sub_sampled_points_idx)


def filter_keypoints(
    keypoints: np.ndarray[np.int32],
    normals: np.ndarray[np.float64],
    normals_z_threshold: float = 0.8,
    sphericity: np.ndarray[np.float64] | None = None,
    sphericity_threshold: float = 0.16,
    verbose: bool = True,
) -> np.ndarray[np.int32]:
    """
    Filters keypoints found on the ground based on the value of the z-coordinate of their normals.
    """
    if sphericity is not None and sphericity.shape[0] == normals.shape[0]:
        mask = (np.abs(normals[keypoints, 2]) < normals_z_threshold) & (
            sphericity < sphericity_threshold
        )
        if verbose:
            print(f"Filtering keypoints based on sphericity and normals ", end="")
    elif sphericity is not None and sphericity.shape[0] == keypoints.shape[0]:
        mask = (np.abs(normals[keypoints, 2]) < normals_z_threshold) & (
            sphericity < sphericity_threshold
        )
        if verbose:
            print(f"Filtering keypoints based on sphericity and normals ", end="")
    else:
        mask = normals[keypoints, 2] < normals_z_threshold
        if verbose:
            print(f"Filtering keypoints based on normals only ", end="")
    if verbose:
        print(f"({mask.sum()} keypoints out of {keypoints.shape[0]}).")
    return keypoints[mask]
