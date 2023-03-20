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
    k: int,
    n_bins: int,
) -> np.ndarray:
    kdtree = KDTree(cloud_points)

    neighborhoods = kdtree.query_radius(cloud_points, radius)
    spfh = np.zeros((cloud_points.shape[0], n_bins, n_bins, n_bins))

    for i, point in tqdm(
        enumerate(cloud_points), desc="SPFH", total=cloud_points.shape[0]
    ):
        neighbors = cloud_points[neighborhoods[i]]
        neighbors_normals = normals[neighborhoods[i]]
        centered_neighbors = neighbors - point
        distances = np.linalg.norm(centered_neighbors, axis=1)
        u = normals[i]
        v = np.cross(u, centered_neighbors / distances[:, None])
        w = np.cross(u, v)
        alpha = np.einsum("ij,ij->i", v, neighbors_normals)
        phi = (centered_neighbors / distances[:, None]).dot(u)
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

    neighborhoods = kdtree.query(
        cloud_points[query_points_indices], k, return_distance=False
    )
    distances = np.linalg.norm(
        cloud_points[neighborhoods] - cloud_points[query_points_indices][:, None],
        axis=2,
    )
    fpfh = (
        spfh[query_points_indices]
        + (spfh[neighborhoods] / distances[:, :, None, None, None]).sum(axis=1) / k
    )

    return fpfh
