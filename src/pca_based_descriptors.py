"""
Implementation of local PCA on point clouds for normals computations and feature extraction (PCA-based descriptors).
"""
from typing import Optional, Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree


def pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the eigenvalues and eigenvectors of the covariance matrix of a point cloud.
    """
    barycenter = points.mean(axis=0)
    centered_points = points - barycenter
    cov_matrix = centered_points.T @ centered_points / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    moment = centered_points @ eigenvectors.T
    vert_moment = centered_points[:, 2]

    return (
        eigenvalues,
        eigenvectors,
        np.hstack(
            (
                np.abs(moment.mean(axis=0)),
                (moment**2).mean(axis=0),
                vert_moment.mean(axis=0),
                (vert_moment**2).mean(axis=0),
            ),
        ),
    )


def compute_local_pca(
    query_points: np.ndarray,
    cloud_points: np.ndarray,
    nghbrd_search: str = "spherical",
    radius: Optional[float] = None,
    k: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Computes PCA on the neighborhoods of all query_points in cloud_points.

    Returns:
        all_eigenvalues: (N, 3)-array of the eigenvalues associated with each query point.
        all_eigenvectors: (N, 3, 3)-array of the eigenvectors associated with each query point.
    """

    kdtree = KDTree(cloud_points)
    neighborhoods = (
        kdtree.query_radius(query_points, radius)
        if nghbrd_search.lower() == "spherical"
        else kdtree.query(query_points, k=k, return_distance=False)
        if nghbrd_search.lower() == "knn"
        else None
    )

    neighborhood_sizes = [neighborhood.shape[0] for neighborhood in neighborhoods]
    # checking the sizes of the neighborhoods and plotting the histogram
    if nghbrd_search.lower() == "spherical" and verbose:
        print(
            f"Average size of neighborhoods: {np.mean(neighborhood_sizes):.4f}\n"
            f"Standard deviation: {np.std(neighborhood_sizes):.4f}\n"
            f"Min: {np.min(neighborhood_sizes)}, max: {np.max(neighborhood_sizes)}\n"
        )
        hist_values, _, __ = plt.hist(neighborhood_sizes, bins="auto")
        plt.title(
            f"Histogram of the neighborhood sizes for {hist_values.shape[0]} bins"
        )
        plt.xlabel("Neighborhood size")
        plt.ylabel("Number of neighborhoods")
        plt.show()

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    moments = np.zeros((query_points.shape[0], 8))

    for i, point in enumerate(query_points):
        all_eigenvalues[i], all_eigenvectors[i], moments[i] = pca(
            cloud_points[neighborhoods[i]]
        )

    return all_eigenvalues, all_eigenvectors, moments, neighborhood_sizes


def compute_basic_features(
    query_points: np.ndarray, cloud_points: np.ndarray, radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes PCA-based descriptors on a point cloud.
    """
    all_eigenvalues, all_eigenvectors, _, __ = compute_local_pca(
        query_points, cloud_points, radius=radius
    )
    lbd3, lbd2, lbd1 = (
        all_eigenvalues[:, 0],
        all_eigenvalues[:, 1],
        all_eigenvalues[:, 2],
    )
    lbd1 += 1e-6

    normals = all_eigenvectors[:, :, 0]

    verticality = 2 * np.arcsin(np.abs(normals[:, 2])) / np.pi
    linearity = 1 - lbd2 / lbd1
    planarity = (lbd2 - lbd3) / lbd1
    sphericity = lbd3 / lbd1

    return verticality, linearity, planarity, sphericity


def compute_features(query_points: np.ndarray, cloud_points: np.ndarray, radius: float):
    """
    Computes PCA-based descriptors on a point cloud.
    """
    all_eigenvalues, all_eigenvectors, moments, neighborhood_sizes = compute_local_pca(
        query_points, cloud_points, radius=radius
    )
    lbd3, lbd2, lbd1 = (
        all_eigenvalues[:, 0],
        all_eigenvalues[:, 1],
        all_eigenvalues[:, 2],
    )
    lbd1 += 1e-6

    normals = all_eigenvectors[:, :, 0]
    principal_axis = all_eigenvectors[:, :, 2]

    eigensum = all_eigenvalues.sum(axis=-1)
    eigen_square_sum = (all_eigenvalues**2).sum(axis=-1)
    omnivariance = all_eigenvalues.prod(axis=-1)
    eigenentropy = (-all_eigenvalues * np.log(all_eigenvalues + 1e-6)).sum(axis=-1)

    linearity = 1 - lbd2 / lbd1
    planarity = (lbd2 - lbd3) / lbd1
    sphericity = lbd3 / lbd1
    curvature_change = lbd3 / eigensum

    verticality = 2 * np.arcsin(np.abs(normals[:, 2])) / np.pi
    lin_verticality = 2 * np.arcsin(np.abs(principal_axis[:, 2])) / np.pi
    horizontalityx = 2 * np.arcsin(np.abs(normals[:, 0])) / np.pi
    horizontalityy = 2 * np.arcsin(np.abs(normals[:, 1])) / np.pi

    return np.hstack(
        (
            eigensum[:, None],
            eigen_square_sum[:, None],
            omnivariance[:, None],
            eigenentropy[:, None],
            linearity[:, None],
            planarity[:, None],
            sphericity[:, None],
            curvature_change[:, None],
            verticality[:, None],
            lin_verticality[:, None],
            horizontalityx[:, None],
            horizontalityy[:, None],
            moments,
            np.array(neighborhood_sizes)[:, None],
        )
    )
