"""
Implementation of local PCA on point clouds for normal computations and feature extraction (PCA-based descriptors).
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree

from utils.perf_monitoring import timeit


def pca(
    points: np.ndarray[np.float64],
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """
    Computes the eigenvalues and eigenvectors of the covariance matrix that describes a point cloud.
    """
    barycenter = points.mean(axis=0)
    centered_points = points - barycenter
    cov_matrix = centered_points.T @ centered_points / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    return eigenvalues, eigenvectors


def compute_normals(
    query_points: np.ndarray[np.float64],
    cloud_points: np.ndarray[np.float64],
    *,
    k: int | None = None,
    radius: float | None = None,
    pre_computed_normals: np.ndarray[np.float64] | None = None,
) -> np.ndarray[np.float64]:
    """
    Computes PCA-based normals on a point cloud.
    Reorients normals based on pre-computed normals if provided.
    """
    assert (
        k is not None or radius is not None
    ), "No parameter provided for the neighborhood search."
    normals = np.zeros((query_points.shape[0], 3))
    neighborhoods = (
        KDTree(cloud_points).query(query_points, k=k, return_distance=False)
        if k is not None
        else KDTree(cloud_points).query_radius(query_points, radius)
    )
    for i in range(query_points.shape[0]):
        normals[i] = pca(cloud_points[neighborhoods[i]])[1][:, 0]
        # reorienting the normal using a rough pre-computation of the normals
        if (
            pre_computed_normals is not None
            and normals[i].dot(pre_computed_normals[i]) < 0
        ):
            normals[i] *= -1

    return normals


def compute_sphericity(
    query_points: np.ndarray[np.float64],
    cloud_points: np.ndarray[np.float64],
    radius: float,
) -> np.ndarray[np.float64]:
    """
    Computes the sphericity on a point cloud.
    """
    eigenvalues = np.zeros((query_points.shape[0], 3))
    neighborhoods = KDTree(cloud_points).query_radius(query_points, radius)
    for i in range(query_points.shape[0]):
        eigenvalues[i] = pca(cloud_points[neighborhoods[i]])[0]
    return eigenvalues[:, 0] / (eigenvalues[:, 2] + 1e-6)


def compute_local_pca_with_moments(
    query_points: np.ndarray[np.float64],
    cloud_points: np.ndarray[np.float64],
    nghbrd_search: str = "spherical",
    radius: float | None = None,
    k: int | None = None,
    verbose: bool = False,
) -> tuple[
    np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64], list[int]
]:
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
        barycenter = cloud_points[neighborhoods[i]].mean(axis=0)
        centered_points = cloud_points[neighborhoods[i]] - barycenter
        cov_matrix = centered_points.T @ centered_points / neighborhoods[i].shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        moment = centered_points @ eigenvectors.T
        vert_moment = centered_points[:, 2]

        all_eigenvalues[i], all_eigenvectors[i], moments[i] = (
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

    return all_eigenvalues, all_eigenvectors, moments, neighborhood_sizes


@timeit
def compute_pca_based_basic_features(
    query_points: np.ndarray[np.float64],
    cloud_points: np.ndarray[np.float64],
    radius: float,
) -> tuple[
    np.ndarray[np.float64],
    np.ndarray[np.float64],
    np.ndarray[np.float64],
    np.ndarray[np.float64],
]:
    """
    Computes PCA-based descriptors on a point cloud.
    """
    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    neighborhoods = KDTree(cloud_points).query_radius(query_points, radius)
    for i in range(query_points.shape[0]):
        all_eigenvalues[i], all_eigenvectors[i] = pca(cloud_points[neighborhoods[i]])

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


@timeit
def compute_pca_based_features(
    query_points: np.ndarray[np.float64],
    cloud_points: np.ndarray[np.float64],
    radius: float,
) -> np.ndarray[np.float64]:
    """
    Computes PCA-based descriptors on a point cloud.
    """
    (
        all_eigenvalues,
        all_eigenvectors,
        moments,
        neighborhood_sizes,
    ) = compute_local_pca_with_moments(query_points, cloud_points, radius=radius)
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
    omnivariance = np.cbrt(all_eigenvalues.prod(axis=-1))
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
