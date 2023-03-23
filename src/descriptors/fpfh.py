"""
Implementation from scratch of the FPFH descriptor based on a careful reading of:
R. B. Rusu, N. Blodow and M. Beetz,
Fast Point Feature Histograms (FPFH) for 3D registration,
2009 IEEE International Conference on Robotics and Automation
"""

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm

from src.perf_monitoring import timeit


@timeit
def compute_fpfh_descriptor(
    query_points_indices: np.ndarray,
    cloud_points: np.ndarray,
    normals: np.ndarray,
    radius: float,
    n_bins: int,
    verbose: bool = True,
) -> np.ndarray:
    kdtree = KDTree(cloud_points)

    neighborhoods, distances = kdtree.query_radius(
        cloud_points, radius, return_distance=True
    )
    spfh = np.zeros((cloud_points.shape[0], n_bins, n_bins, n_bins))
    neighborhood_size = 0

    for i, point in tqdm(
        enumerate(cloud_points), desc="SPFH", total=cloud_points.shape[0]
    ):
        neighbors = cloud_points[neighborhoods[i]]
        neighbors_normals = normals[neighborhoods[i]]
        centered_neighbors = neighbors - point
        dist = np.linalg.norm(centered_neighbors, axis=1)
        u = normals[i]
        v = np.cross(centered_neighbors, u)
        w = np.cross(u, v)
        alpha = np.einsum("ij,ij->i", v, neighbors_normals)
        phi = centered_neighbors.dot(u) / dist
        theta = np.arctan2(
            np.einsum("ij,ij->i", neighbors_normals, w), neighbors_normals.dot(u)
        )
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

    fpfh = np.zeros((query_points_indices.shape[0], n_bins, n_bins, n_bins))
    for i, neighborhood in enumerate(neighborhoods[query_points_indices]):
        fpfh[i] = (
            spfh[query_points_indices[i]]
            + (spfh[neighborhood] / distances[query_points_indices[i]][:, None, None, None])[
                distances[query_points_indices[i]] > 0
                ].sum(axis=0)
            / neighborhood.shape[0]
        )
    return fpfh.reshape(query_points_indices.shape[0], -1)
