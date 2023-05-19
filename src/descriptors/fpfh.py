"""
Implementation from scratch of the FPFH descriptor based on a careful reading of:
R. B. Rusu, N. Blodow and M. Beetz,
Fast Point Feature Histograms (FPFH) for 3D registration,
2009 IEEE International Conference on Robotics and Automation
"""

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm


def compute_fpfh_descriptor(
    keypoints_indices: np.ndarray[np.int32],
    cloud_points: np.ndarray[np.float64],
    normals: np.ndarray[np.float64],
    radius: float,
    n_bins: int,
    decorrelated: bool = False,
    verbose: bool = True,
) -> np.ndarray[np.float64]:
    kdtree = KDTree(cloud_points)

    neighborhoods, distances = kdtree.query_radius(
        cloud_points, radius, return_distance=True
    )
    spfh = np.zeros(
        (cloud_points.shape[0], n_bins * 3)
        if decorrelated
        else (cloud_points.shape[0], n_bins, n_bins, n_bins)
    )
    neighborhood_size = 0

    for i, point in tqdm(
        enumerate(cloud_points), desc="SPFH", total=cloud_points.shape[0]
    ):
        if neighborhoods[i].shape[0] > 0:
            neighbors = cloud_points[neighborhoods[i]]
            neighbors_normals = normals[neighborhoods[i]]
            centered_neighbors = neighbors - point
            dist = np.linalg.norm(centered_neighbors, axis=1)
            u = normals[i]
            v = np.cross(centered_neighbors[dist > 0], u)
            w = np.cross(u, v)
            alpha = np.einsum("ij,ij->i", v, neighbors_normals[dist > 0])
            phi = centered_neighbors[dist > 0].dot(u) / dist[dist > 0]
            theta = np.arctan2(
                np.einsum("ij,ij->i", neighbors_normals[dist > 0], w),
                neighbors_normals[dist > 0].dot(u),
            )
            if decorrelated:
                spfh[i, :] = (
                    np.vstack(
                        (
                            np.histogram(
                                alpha,
                                bins=n_bins,
                                range=(-1, 1),
                            )[0],
                            np.histogram(
                                phi,
                                bins=n_bins,
                                range=(-1, 1),
                            )[0],
                            np.histogram(
                                theta,
                                bins=n_bins,
                                range=(-np.pi / 2, np.pi / 2),
                            )[0],
                        )
                    ).T
                    / neighborhoods[i].shape[0]
                )
            else:
                spfh[i, :, :, :] = (
                    np.histogramdd(
                        np.vstack((alpha, phi, theta)).T,
                        bins=n_bins,
                        range=[(-1, 1), (-1, 1), (-np.pi / 2, np.pi / 2)],
                    )[0]
                    / neighborhoods[i].shape[0]
                )
            neighborhood_size += neighborhoods[i].shape[0]

    if verbose:
        print(
            f"Mean neighborhood size over the whole point cloud: {neighborhood_size / cloud_points.shape[0]:.2f}"
        )

    spfh = spfh.reshape(cloud_points.shape[0], -1)
    fpfh = np.zeros(
        (keypoints_indices.shape[0], n_bins * 3 if decorrelated else n_bins**3)
    )
    for i, neighborhood in tqdm(
        enumerate(neighborhoods[keypoints_indices]),
        desc="FPFH",
        total=keypoints_indices.shape[0],
        delay=0.5,
    ):
        with np.errstate(invalid="ignore", divide="ignore"):
            fpfh[i] = (
                spfh[keypoints_indices[i]]
                # should be ok to encounter a RuntimeWarning here since we apply a mask after the divide
                + (spfh[neighborhood] / distances[keypoints_indices[i]][:, None])[
                    distances[keypoints_indices[i]] > 0
                ].sum(axis=0)
                / neighborhood.shape[0]
            )
    return fpfh
