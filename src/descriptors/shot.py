"""
Implementation from scratch of the SHOT descriptor based on a careful reading of:
Samuele Salti, Federico Tombari, Luigi Di Stefano,
SHOT: Unique signatures of histograms for surface and texture description,
Computer Vision and Image Understanding,
"""
import warnings

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm


def get_local_rf(
    values: tuple[np.ndarray[np.float64, 3], np.ndarray[np.float64], float],
) -> np.ndarray[np.float64]:
    """
    Extracts a local reference frame based on the eigendecomposition of the weighted covariance matrix.
    Arguments are given in a tuple to allow for multiprocessing using multiprocessing.Pool.
    """
    point, neighbors, radius = values
    if neighbors.shape[0] == 0:
        return np.eye(3)

    centered_points = neighbors - point

    # EVD of the weighted covariance matrix
    radius_minus_distances = radius - np.linalg.norm(centered_points, axis=1)
    weighted_cov_matrix = (
        centered_points.T
        @ (centered_points * radius_minus_distances[:, None])
        / radius_minus_distances.sum()
    )
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov_matrix)

    # disambiguating the axes
    # TODO: deal with the equality case (where the two sums below are equal)
    x_orient = (neighbors - point) @ eigenvectors[:, 2]
    if (x_orient < 0).sum() > (x_orient >= 0).sum():
        eigenvectors[:, 2] *= -1
    z_orient = (neighbors - point) @ eigenvectors[:, 0]
    if (z_orient < 0).sum() > (z_orient >= 0).sum():
        eigenvectors[:, 0] *= -1
    eigenvectors[:, 1] = np.cross(eigenvectors[:, 0], eigenvectors[:, 2])

    return np.flip(eigenvectors, axis=1)


def get_azimuth_idx(
    x: float | np.ndarray[np.float64], y: float | np.ndarray[np.float64]
) -> int | np.ndarray[np.int32]:
    """
    Finds the bin index of the azimuth of a point in a division in 8 bins.
    Bins are indexed clockwise, and the first bin is between pi and 3 * pi / 4.
    To use 4 bins instead of 8, divide each factor by 2 and remove the last one that compares |y| and |x|.
    To use 2 bins instead of 8, repeat the process to only keep the condition (y > 0) | ((y == 0) & (x < 0)).
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
    distance: float | np.ndarray[np.float64], radius: float
) -> tuple[
    float | np.ndarray[np.float64],
    float | np.ndarray[np.float64],
    float | np.ndarray[np.float64],
]:
    """
    Interpolates on the adjacent husks.
    Assumes there are only two husks, centered around radius / 4 and 3 * radius / 4.

    Args:
        distance: distance or array of distances to the center of the sphere.
        radius: radius of the neighborhood sphere.

    Returns:
        outer_bin: value or array of values to add to the outer bin.
        Equal to 0 if the point is in the outer bin.
        inner_bin: value or array of values to add to the inner bin.
        Equal to 0 if the point is in the inner bin.
        current_bin: value or array of values to add to the current bin.
    """
    radial_bin_size = radius / 2  # normalized distance between two radial neighbor bins
    # external sphere that is closer to radius / 2 than to the border
    inner_bin = (
        ((distance > radius / 2) & (distance < radius * 3 / 4))
        * (radius * 3 / 4 - distance)
        / radial_bin_size
    )
    # internal sphere that is closer to radius / 2 than to a null radius
    outer_bin = (
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


def interpolate_vertical_volumes(
    phi: float | np.ndarray[np.float64], z: float | np.ndarray[np.float64]
) -> tuple[
    float | np.ndarray[np.float64],
    float | np.ndarray[np.float64],
    float | np.ndarray[np.float64],
]:
    """
    Interpolates on the adjacent vertical volumes.
    Assumes there are only two volumes, centered around pi / 4 and 3 * pi / 4.
    The upper volume is the one found for z > 0.

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
            ((phi > np.pi / 2) | ((np.abs(phi - np.pi / 2) < 1e-10) & (z <= 0)))
            & (phi <= np.pi * 3 / 4)
        )
        * (np.pi * 3 / 4 - phi)
        / phi_bin_size
    )
    # phi between pi / 4 and pi / 2
    lower_volume = (
        (
            ((phi < np.pi / 2) & ((np.abs(phi - np.pi / 2) >= 1e-10) | (z > 0)))
            & (phi >= np.pi / 4)
        )
        * (phi - np.pi / 4)
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


# noinspection DuplicatedCode
def compute_single_shot_descriptor(
    values: tuple[
        np.ndarray[np.float64, 3],
        np.ndarray[np.float64],
        np.ndarray[np.float64],
        float,
        np.ndarray[np.float64, (3, 3)],
        bool,
        int,
    ]
) -> np.ndarray[np.float64, 352]:
    """
    Computes a single SHOT descriptor.
    Arguments are given in a tuple to allow for multiprocessing using multiprocessing.Pool.

    Args:
        values: (point, neighbors, normals, radius, local_rf, normalize, min_neighborhood_size).

    Returns:
        The SHOT descriptor.
    """
    # the number of bins are hardcoded in this version, passing parameters in a multiprocessed settings gets cumbersome
    n_cosine_bins, n_azimuth_bins, n_elevation_bins, n_radial_bins = 11, 8, 2, 2

    descriptor = np.zeros(
        (n_cosine_bins, n_azimuth_bins, n_elevation_bins, n_radial_bins)
    )
    (
        point,
        neighbors,
        normals,
        radius,
        eigenvectors,
        normalize,
        min_neighborhood_size,
    ) = values
    rho = np.linalg.norm(neighbors - point, axis=1)
    if (rho > 0).sum() > min_neighborhood_size:
        neighbors = neighbors[rho > 0]
        local_coordinates = (neighbors - point) @ eigenvectors
        cosine = np.clip(normals[rho > 0] @ eigenvectors[:, 2].T, -1, 1)
        rho = rho[rho > 0]

        order = np.argsort(rho)
        rho = rho[order]
        local_coordinates = local_coordinates[order]
        cosine = cosine[order]

        # computing the spherical coordinates in the local coordinate system
        theta = np.arctan2(local_coordinates[:, 1], local_coordinates[:, 0])
        phi = np.arccos(np.clip(local_coordinates[:, 2] / rho, -1, 1))

        # computing the indices in the histograms
        cos_bin_pos = (cosine + 1.0) * n_cosine_bins / 2.0 - 0.5
        cos_bin_idx = np.rint(cos_bin_pos).astype(int)
        theta_bin_idx = get_azimuth_idx(
            local_coordinates[:, 0], local_coordinates[:, 1]
        )
        # the two arrays below have to be cast as ints, otherwise they will be treated as masks
        phi_bin_idx = (local_coordinates[:, 2] > 0).astype(int)
        rho_bin_idx = (rho > radius / 2).astype(int)

        # interpolation on the local bins
        delta_cos = (
            cos_bin_pos - cos_bin_idx
        )  # normalized distance with the neighbor bin
        delta_cos_sign = np.sign(delta_cos)  # left-neighbor or right-neighbor
        abs_delta_cos = delta_cos_sign * delta_cos  # probably faster than np.abs
        # noinspection PyRedundantParentheses
        descriptor[
            (cos_bin_idx + delta_cos_sign).astype(int) % n_cosine_bins,
            theta_bin_idx,
            phi_bin_idx,
            rho_bin_idx,
        ] += abs_delta_cos * (
            (cos_bin_idx > -0.5) & (cos_bin_idx < n_cosine_bins - 0.5)
        )
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += (
            1 - abs_delta_cos
        )

        # interpolation on the adjacent husks
        outer_bin, inner_bin, current_bin = interpolate_on_adjacent_husks(rho, radius)
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, 1] += outer_bin * (
            rho_bin_idx == 0
        )
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, 0] += inner_bin * (
            rho_bin_idx == 1
        )
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += current_bin

        # interpolation between adjacent vertical volumes
        upper_volume, lower_volume, current_volume = interpolate_vertical_volumes(
            phi, local_coordinates[:, 2]
        )
        descriptor[cos_bin_idx, theta_bin_idx, 1, rho_bin_idx] += upper_volume * (
            phi_bin_idx == 0
        )
        descriptor[cos_bin_idx, theta_bin_idx, 0, rho_bin_idx] += lower_volume * (
            phi_bin_idx == 1
        )
        descriptor[
            cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx
        ] += current_volume

        # interpolation between adjacent horizontal volumes
        # local_coordinates[:, 0] * local_coordinates[:, 1] != 0
        theta_bin_size = 2 * np.pi / n_azimuth_bins
        delta_theta = np.clip(
            (theta - (-np.pi + theta_bin_idx * theta_bin_size)) / theta_bin_size - 0.5,
            -0.5,
            0.5,
        )
        delta_theta_sign = np.sign(delta_theta)  # left-neighbor or right-neighbor
        abs_delta_theta = delta_theta_sign * delta_theta
        descriptor[
            cos_bin_idx,
            (theta_bin_idx + delta_theta_sign).astype(int) % n_azimuth_bins,
            phi_bin_idx,
            rho_bin_idx,
        ] += abs_delta_theta
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += (
            1 - abs_delta_theta
        )

        # normalizing the descriptor to Euclidian norm 1
        if (descriptor_norm := np.linalg.norm(descriptor)) > 0:
            if normalize:
                return descriptor.ravel() / descriptor_norm
            else:
                return descriptor.ravel()
    return np.zeros(n_cosine_bins * n_azimuth_bins * n_elevation_bins * n_radial_bins)


# noinspection DuplicatedCode
def compute_shot_descriptor(
    keypoints: np.ndarray[np.float64],
    cloud_points: np.ndarray[np.float64],
    normals: np.ndarray[np.float64],
    radius: float,
    min_neighborhood_size: int = 10,
    n_cosine_bins: int = 11,
    n_azimuth_bins: int = 8,
    n_elevation_bins: int = 2,
    n_radial_bins: int = 2,
    debug_mode: bool = False,
) -> np.ndarray[np.float64]:
    """
    Computes the SHOT descriptor on a point cloud. This function should not be used in practice and is only kept for
    debugging purposes. Use the methods from class ShotMultiprocessor instead.
    Normals are expected to be normalized to 1.
    Only the number of cosine bins can be changed as it is.
    See get_azimuth_idx for a description on how to change the number of azimuth bins.
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
    neighborhoods = kdtree.query_radius(keypoints, radius)

    all_descriptors = np.zeros(
        (
            keypoints.shape[0],
            n_cosine_bins * n_azimuth_bins * n_elevation_bins * n_radial_bins,
        )
    )

    for i, point in tqdm(enumerate(keypoints), desc="SHOT", total=len(keypoints)):
        descriptor = np.zeros(
            (n_cosine_bins, n_azimuth_bins, n_elevation_bins, n_radial_bins)
        )
        neighbors = cloud_points[neighborhoods[i]]
        distances = np.linalg.norm(neighbors - point, axis=1)
        if (distances > 0).sum() > min_neighborhood_size:
            neighbors = neighbors[distances > 0]
            eigenvectors = get_local_rf((point, neighbors, radius))
            local_coordinates = (neighbors - point) @ eigenvectors
            cosine = np.clip(
                normals[neighborhoods[i]][distances > 0] @ eigenvectors[:, 2].T, -1, 1
            )
            distances = distances[distances > 0]

            order = np.argsort(distances)
            distances = distances[order]
            local_coordinates = local_coordinates[order]
            cosine = cosine[order]

            if debug_mode:
                assert distances.shape[0] <= neighbors.shape[0]
                assert local_coordinates.shape[0] == neighbors.shape[0]
                assert local_coordinates.shape[1] == 3
                assert cosine.shape[0] == neighbors.shape[0]

            # computing the spherical coordinates in the local coordinate system
            theta = np.arctan2(local_coordinates[:, 1], local_coordinates[:, 0])
            phi = np.arccos(np.clip(local_coordinates[:, 2] / distances, -1, 1))

            # computing the indices in the histograms
            bin_dist = (cosine + 1.0) * n_cosine_bins / 2.0 - 0.5
            bin_idx = np.rint(bin_dist).astype(int)
            azimuth_idx = get_azimuth_idx(
                local_coordinates[:, 0], local_coordinates[:, 1]
            )
            # the two arrays below have to be cast as ints, otherwise they will be treated as masks
            elevation_idx = (local_coordinates[:, 2] > 0).astype(int)
            radial_idx = (distances > radius / 2).astype(int)

            # interpolation on the local bins
            bin_dist -= bin_idx  # normalized distance with the neighbor bin
            dist_sign = np.sign(bin_dist)  # left-neighbor or right-neighbor
            abs_bin_dist = dist_sign * bin_dist  # probably faster than np.abs
            # noinspection PyRedundantParentheses
            descriptor[
                (bin_idx + dist_sign).astype(int) % n_cosine_bins,
                azimuth_idx,
                elevation_idx,
                radial_idx,
            ] += abs_bin_dist * (((bin_idx > -0.5) & (bin_idx < n_cosine_bins - 0.5)))
            descriptor[bin_idx, azimuth_idx, elevation_idx, radial_idx] += (
                1 - abs_bin_dist
            )

            # interpolation on the adjacent husks
            outer_bin, inner_bin, current_bin = interpolate_on_adjacent_husks(
                distances, radius
            )
            if debug_mode:
                if np.any((radial_idx == 1) & (np.abs(outer_bin) > 1e-4)):
                    warnings.warn(
                        "Nonzero value for outer volume although current volume is already the outer one."
                    )
                    warnings.warn(
                        str(outer_bin[radial_idx == 1 & (np.abs(outer_bin) > 1e-4)])
                    )
                if np.any((radial_idx == 0) & (np.abs(inner_bin) > 1e-4)):
                    warnings.warn(
                        "Nonzero value for inner volume although current volume is already the inner one."
                    )
                    warnings.warn(
                        str(inner_bin[radial_idx == 0 & (np.abs(inner_bin) > 1e-4)])
                    )
            descriptor[bin_idx, azimuth_idx, elevation_idx, 1] += outer_bin * (
                radial_idx == 0
            )
            descriptor[bin_idx, azimuth_idx, elevation_idx, 0] += inner_bin * (
                radial_idx == 1
            )
            descriptor[bin_idx, azimuth_idx, elevation_idx, radial_idx] += current_bin

            # interpolation between adjacent vertical volumes
            upper_volume, lower_volume, current_volume = interpolate_vertical_volumes(
                phi, local_coordinates[:, 2]
            )
            if debug_mode:
                if np.any((elevation_idx == 1) & (np.abs(upper_volume) > 1e-4)):
                    warnings.warn(
                        "Nonzero value for upper volume although current volume is already the upper one."
                    )
                    warnings.warn(
                        str(
                            upper_volume[
                                elevation_idx == 1 & (np.abs(upper_volume) > 1e-4)
                            ]
                        )
                    )
                if np.any((elevation_idx == 0) & (np.abs(lower_volume) > 1e-4)):
                    warnings.warn(
                        "Nonzero value for lower volume although current volume is already the lower one."
                    )
                    warnings.warn(
                        str(
                            lower_volume[
                                elevation_idx == 0 & (np.abs(lower_volume) > 1e-4)
                            ]
                        )
                    )
            descriptor[bin_idx, azimuth_idx, 1, radial_idx] += upper_volume * (
                elevation_idx == 0
            )
            descriptor[bin_idx, azimuth_idx, 0, radial_idx] += lower_volume * (
                elevation_idx == 1
            )
            descriptor[
                bin_idx, azimuth_idx, elevation_idx, radial_idx
            ] += current_volume

            # interpolation between adjacent horizontal volumes
            # local_coordinates[:, 0] * local_coordinates[:, 1] != 0
            azimuth_bin_size = 2 * np.pi / n_azimuth_bins
            azimuth_dist = np.clip(
                (theta - (-np.pi + azimuth_idx * azimuth_bin_size)) / azimuth_bin_size
                - 0.5,
                -0.5,
                0.5,
            )
            azimuth_dist_sign = np.sign(azimuth_dist)  # left-neighbor or right-neighbor
            azimuth_abs_dist = azimuth_dist_sign * azimuth_dist
            descriptor[
                bin_idx,
                (azimuth_idx + azimuth_dist_sign).astype(int) % n_azimuth_bins,
                elevation_idx,
                radial_idx,
            ] += azimuth_abs_dist
            descriptor[bin_idx, azimuth_idx, elevation_idx, radial_idx] += (
                1 - azimuth_abs_dist
            )

            # normalizing the descriptor to Euclidian norm 1
            if (descriptor_norm := np.linalg.norm(descriptor)) > 0:
                all_descriptors[i] = descriptor.ravel() / descriptor_norm

    return all_descriptors
