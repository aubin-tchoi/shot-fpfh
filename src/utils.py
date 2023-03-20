from typing import Tuple

import numpy as np


def best_rigid_transform(
    data: np.ndarray, ref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    """

    data_barycenter = data.mean(axis=1)
    ref_barycenter = ref.mean(axis=1)
    covariance_matrix = (data - data_barycenter[:, np.newaxis]).squeeze() @ (
        ref - ref_barycenter[:, np.newaxis]
    ).squeeze().T
    u, sigma, v = np.linalg.svd(covariance_matrix)
    rotation = v.T @ u.T

    # ensuring that we have a direct rotation (determinant equal to 1 and not -1)
    if np.linalg.det(rotation) < 0:
        u_transpose = u.T
        u_transpose[-1] *= -1
        rotation = v.T @ u_transpose

    translation = ref_barycenter - rotation.dot(data_barycenter)

    return rotation, translation


def compute_rigid_transform_error(
    data: np.ndarray,
    reference: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Computes the RMS error between a reference point cloud and data that went through the rigid transformation described
    by the rotation and the translation.
    """
    transformed_data = rotation.dot(data) + translation[:, np.newaxis]
    return (
        np.sqrt(
            np.sum(
                (transformed_data - reference) ** 2,
                axis=0,
            ).mean()
        ),
        transformed_data,
    )
