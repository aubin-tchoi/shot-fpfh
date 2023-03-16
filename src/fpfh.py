"""
Implementation from scratch of the FPFH descriptor based on a careful reading of:
R. B. Rusu, N. Blodow and M. Beetz,
Fast Point Feature Histograms (FPFH) for 3D registration,
2009 IEEE International Conference on Robotics and Automation
"""

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm

from .perf_monitoring import timeit


@timeit
def compute_fpfh_descriptor(
    query_points: np.ndarray,
    cloud_points: np.ndarray,
    normals: np.ndarray,
    radius: float,
    k: int,
) -> np.ndarray:
    kdtree = KDTree(cloud_points)

    neighborhoods = kdtree.query_radius(query_points, radius)
    spfh = np.zeros((query_points.shape[0], 3))
    for i, point in tqdm(
        enumerate(query_points), desc="SPFH", total=query_points.shape[0]
    ):
        neighbors = cloud_points[neighborhoods[i]]
        neighbors_normals = normals[neighborhoods[i]]
        centered_neighbors = neighbors - point
        distances = np.linalg.norm(centered_neighbors, axis=1)
        u = normals[i]
        v = np.cross(u, centered_neighbors / distances[:, None])
        w = np.cross(u, v)
        alpha = v @ neighbors_normals
        phi = u @ centered_neighbors / distances[:, None]
        theta = np.arctan2(w @ neighbors_normals, u @ neighbors_normals)
        # TODO: add the bin accumulation part
        spfh[i] = [alpha, phi, theta]

    neighborhoods = kdtree.query(query_points, k, return_distance=False)
    fpfh = np.zeros((query_points.shape[0], 3))
    for i, point in tqdm(
        enumerate(query_points), desc="FPFH", total=query_points.shape[0]
    ):
        distances = np.linalg.norm(cloud_points[neighborhoods[i]] - point, axis=1)
        fpfh[i] = spfh[i] + (spfh[neighborhoods[i]] / distances) / k

    return fpfh
