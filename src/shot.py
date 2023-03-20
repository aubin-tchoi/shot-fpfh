"""
Implementation from scratch of the SHOT descriptor based on a careful reading of:
Samuele Salti, Federico Tombari, Luigi Di Stefano,
SHOT: Unique signatures of histograms for surface and texture description,
Computer Vision and Image Understanding,
"""

from typing import Tuple

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
from .perf_monitoring import timeit


def get_local_rf(point: np.ndarray, neighbors: np.ndarray, radius: float) -> np.ndarray:
    """
    Extracts a local reference frame based on the eigendecomposition of the weighted covariance matrix.
    """
    barycenter = neighbors.mean(axis=0)
    centered_points = neighbors - barycenter

    # EVD of the weighted covariance matrix
    distances = np.linalg.norm(neighbors - point, axis=1)
    weighted_cov_matrix = centered_points.T @ (
        centered_points * (radius - distances[:, None])
    )
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov_matrix)

    # disambiguating the axes
    if (-(neighbors - point) @ eigenvectors[:, 2] > 0).sum() > (
        (neighbors - point) @ eigenvectors[:, 2] >= 0
    ).sum():
        eigenvectors[:, 2] *= -1
    if (-(neighbors - point) @ eigenvectors[:, 0] > 0).sum() > (
        (neighbors - point) @ eigenvectors[:, 0] >= 0
    ).sum():
        eigenvectors[:, 0] *= -1
    eigenvectors[:, 1] = np.cross(eigenvectors[:, 2], eigenvectors[:, 0])

    return eigenvectors


def get_azimuth_idx(x: float | np.ndarray, y: float | np.ndarray) -> int:
    """
    Finds the bin index of the azimuth of a point in a division in 8 bins.
    Bins are indexed clockwise and the first bin is between pi and 3 * pi / 4
    """
    a = (y > 0) | ((y == 0) & (x < 0))
    return (
        4 * a  # top or bottom half
        + 2 * np.logical_xor((x > 0) | ((x == 0) & (y > 0)), a)  # left or right
        # half of each corner
        + np.where(
            (x * y > 0) | (x == 0),
            np.abs(x) < np.abs(y),
            np.abs(x) > np.abs(y),
        )
    )


def interpolate_on_adjacent_husks(
    distance: float | np.ndarray, radius: float
) -> Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Interpolates on the adjacent husks.
    Assumes there are only two husks, centered around radius / 4 and 3 * radius / 4.

    Args:
        distance: distance or array of distances to the center of the sphere.
        radius: radius of the neighbourhood sphere.

    Returns:
        outer_bin: value or array of values to add to the outer bin. Can be equal to 0 if the point is in the outer bin.
        inner_bin: value or array of values to add to the inner bin. Can be equal to 0 if the point is in the inner bin.
        current_bin: value or array of values to add to the current bin.
    """
    radial_bin_size = (
        radius / 2
    )  # normalized distance between two radially neighbor bins
    # external sphere that is closer to radius / 2 than to the border
    outer_bin = (
        ((distance > radius / 2) & (distance < radius * 3 / 4))
        * (radius * 3 / 4 - distance)
        / radial_bin_size
    )
    # internal sphere that is closer to radius / 2 than to a null radius
    inner_bin = (
        ((distance < radius / 2) & (distance > radius / 4))
        * (distance - radius / 4)
        / radial_bin_size
    )
    current_bin = (
        # radius / 4 is the center of the inner bin
        (distance < radius / 2)
        * (1 - np.abs(distance - radius / 4) / radial_bin_size)
    ) + (
        # 3 * radius / 4 is the center of the outer bin
        (distance > radius / 2)
        * (1 - np.abs(distance - radius * 3 / 4) / radial_bin_size)
    )

    return outer_bin, inner_bin, current_bin


def interpolate_vertical_volumes(phi: float | np.ndarray, z: float | np.ndarray):
    """
    Interpolates on the adjacent vertical volumes.
    Assumes there are only two volumes, centered around pi / 4 and 3 * pi / 4.

    Args:
        phi: elevation or array of elevations.
        z: vertical coordinate.

    Returns:
        outer_volume: value or array of values to add to the outer volume.
        inner_volume: value or array of values to add to the inner volume.
        current_volume: value or array of values to add to the current volume.
    """
    phi_bin_size = np.pi / 2
    # phi between pi / 2 and 3 * pi / 4
    upper_volume = (
        (
            ((phi > np.pi / 2) | ((np.abs(phi - np.pi / 2) < 1e-5) & (z < 0)))
            & (phi <= np.pi * 3 / 4)
        )
        * (np.pi * 3 / 4 - phi)
        / phi_bin_size
    )
    # phi between pi / 4 and pi / 2
    lower_volume = (
        (
            ((phi < np.pi / 2) & ((np.abs(phi - np.pi / 2) >= 1e-5) | (z >= 0)))
            & (phi >= np.pi / 4)
        )
        * (phi - np.pi * 4)
        / phi_bin_size
    )
    current_volume = (
        # pi / 4 is the center of the upper bin
        (phi < np.pi / 2)
        * (1 - np.abs(phi - np.pi / 4) / phi_bin_size)
    ) + (
        # 3 * pi / 4 is the center of the lower bin
        (phi >= np.pi / 2)
        * (1 - np.abs(phi - np.pi * 3 / 4) / phi_bin_size)
    )

    return upper_volume, lower_volume, current_volume


@timeit
def compute_shot_descriptor(
    query_points: np.ndarray,
    cloud_points: np.ndarray,
    normals: np.ndarray,
    radius: float,
    n_cosine_bins: int = 11,
    n_azimuth_bins: int = 8,
    n_elevation_bins: int = 2,
    n_radial_bins: int = 2,
) -> np.ndarray:
    """
    Computes the SHOT descriptor on a point cloud.
    Normals are expected to be normalized to 1.
    """
    assert (
        n_azimuth_bins == 8
    ), "Generic function for other than 8 azimuth divisions not implemented"
    assert (
        n_elevation_bins == 2
    ), "Generic function for other than 2 elevation divisions not implemented"
    assert (
        n_radial_bins == 2
    ), "Generic function for other than 2 radial divisions not implemented"

    kdtree = KDTree(cloud_points)
    neighborhoods = kdtree.query_radius(query_points, radius)

    all_descriptors = np.zeros(
        (
            query_points.shape[0],
            n_cosine_bins * n_azimuth_bins * n_elevation_bins * n_radial_bins,
        )
    )

    for i, point in tqdm(
        enumerate(query_points), desc="SHOT", total=len(query_points)
    ):
        descriptor = np.zeros(
            (n_cosine_bins, n_azimuth_bins, n_elevation_bins, n_radial_bins)
        )
        neighbors = cloud_points[neighborhoods[i]]
        eigenvectors = get_local_rf(point, neighbors, radius)
        local_coordinates = (neighbors - point) @ eigenvectors
        cosine = np.clip(normals[neighborhoods[i]] @ eigenvectors[:, 2].T, -1, 1)

        # computing the spherical coordinates in the local coordinate system
        distances = np.linalg.norm(neighbors - point, axis=1)
        theta = np.arctan2(local_coordinates[:, 1], local_coordinates[:, 0])
        phi = np.arccos(np.clip(local_coordinates[:, 2] / distances, -1, 1))

        # computing the indices in the histograms. This part does not generalize to different numbers of bins.
        bin_dist = (cosine + 1.0) * n_cosine_bins / 2.0 - 0.5
        bin_idx = np.rint(bin_dist).astype(int)
        azimuth_idx = get_azimuth_idx(local_coordinates[:, 0], local_coordinates[:, 1])
        # the two arrays below have to be cast as ints, otherwise they will be treated as masks
        elevation_idx = (local_coordinates[:, 2] > 0).astype(int)
        radial_idx = (distances > radius / 2).astype(int)

        # interpolation on the local bins
        bin_dist -= bin_idx  # normalized distance with the neighbor bin
        dist_sign = np.sign(bin_dist)  # left-neighbor or right-neighbor
        abs_bin_dist = dist_sign * bin_dist  # probably faster than np.abs
        descriptor[
            (bin_idx + dist_sign).astype(int) % n_cosine_bins,
            azimuth_idx,
            elevation_idx,
            radial_idx,
        ] += abs_bin_dist * ((bin_idx > -0.5) & (bin_idx < n_cosine_bins - 0.5))
        descriptor[bin_idx, azimuth_idx, elevation_idx, radial_idx] += 1 - abs_bin_dist

        # interpolation on the adjacent husks
        outer_bin, inner_bin, current_bin = interpolate_on_adjacent_husks(
            distances, radius
        )
        descriptor[
            bin_idx, azimuth_idx, elevation_idx, (radial_idx + 1) % n_radial_bins
        ] += outer_bin
        descriptor[
            bin_idx, azimuth_idx, elevation_idx, (radial_idx - 1) % n_radial_bins
        ] += inner_bin
        descriptor[bin_idx, azimuth_idx, elevation_idx, radial_idx] += current_bin

        # interpolation between adjacent vertical volumes
        upper_volume, lower_volume, current_volume = interpolate_vertical_volumes(
            phi, local_coordinates[:, 2]
        )
        descriptor[
            bin_idx, azimuth_idx, (elevation_idx + 1) % n_elevation_bins, radial_idx
        ] += upper_volume
        descriptor[
            bin_idx, azimuth_idx, (elevation_idx - 1) % n_elevation_bins, radial_idx
        ] += lower_volume
        descriptor[bin_idx, azimuth_idx, elevation_idx, radial_idx] += current_volume

        # interpolation between adjacent horizontal volumes
        # local_coordinates[:, 0] * local_coordinates[:, 1] != 0
        azimuth_bin_size = 2 * np.pi / n_azimuth_bins
        azimuth_dist = np.clip(
            (theta - (-np.pi + azimuth_idx * azimuth_bin_size)) / azimuth_bin_size,
            -0.5,
            0.5,
        )
        dist_sign = np.sign(azimuth_dist)  # left-neighbor or right-neighbor
        abs_bin_dist = dist_sign * azimuth_dist
        descriptor[
            bin_idx,
            (azimuth_idx + dist_sign).astype(int) % n_azimuth_bins,
            elevation_idx,
            radial_idx,
        ] += abs_bin_dist
        descriptor[bin_idx, azimuth_idx, elevation_idx, radial_idx] += 1 - abs_bin_dist

        # normalizing the descriptor to Euclidian norm 1
        all_descriptors[i] = (descriptor / np.linalg.norm(descriptor)).flatten()

    return all_descriptors
