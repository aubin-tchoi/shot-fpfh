from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree

from base_computation import Transformation


def quaternion_to_rotation_matrix(
    quaternion: List[float],
) -> np.ndarray[np.float64, (3, 3)]:
    """
    Converts a quaternion into a full three-dimensional rotation matrix.
    """
    q3, q0, q1, q2 = quaternion
    return Rotation.from_quat([q0, q1, q2, q3]).as_matrix().squeeze()


def read_conf_file(file_path: str) -> Dict[str, Transformation]:
    """
    Reads a .conf file from the Stanford 3D Scanning Repository to output the (translation, rotation) for each ply file.
    """
    with open(file_path, "r") as file:
        transformations = {
            line_values[1].replace(".ply", ""): Transformation(
                quaternion_to_rotation_matrix([float(val) for val in line_values[5:]]),
                np.array([float(val) for val in line_values[2:5]]),
            )
            for line in file.readlines()
            if (line_values := line.split(" "))[0] == "bmesh"
        }
    return transformations


def get_resulting_transform(
    scan_file_name: str, ref_file_name: str, conf: Dict[str, Transformation]
) -> Transformation:
    """
    Outputs the rotation and translation that send the data corresponding to file_name onto the data corresponding to
    ref_file_name using a config that contains rotations and translations to send them to a common coordinate system.
    """
    scan_transformation = conf[scan_file_name.split("/")[-1].replace(".ply", "")]
    ref_transformation = conf[ref_file_name.split("/")[-1].replace(".ply", "")]

    return (~ref_transformation) @ scan_transformation


def count_correct_matches(
    scan: np.ndarray[np.float64],
    ref: np.ndarray[np.float64],
    transformation: Transformation,
) -> int:
    """
    Counts the number of matches between two sets of points.

    Args:
        scan: data to align.
        ref: reference data.
        transformation: transformation that sends scan on ref.

    Returns:
        n_matches: number of correct matches.
    """
    return (np.linalg.norm(transformation[scan] - ref, axis=1) < 0.1).sum()


def plot_distance_hists(
    scan: np.ndarray[np.float64],
    ref: np.ndarray[np.float64],
    transformation: Transformation,
    descriptors: np.ndarray[np.float64],
    descriptors_ref: np.ndarray[np.float64],
) -> None:
    """
    Validates the double_matching_with_rejects by plotting histograms of the ratio between the distances to the nearest
    neighbor and to the second-nearest neighbor in the descriptor space.

    Args:
        scan: data to align.
        ref: reference data.
        transformation: transformation that sends scan on ref.
        descriptors: descriptors computed on scan.
        descriptors_ref: descriptors computed on ref.
    """
    kdtree = KDTree(ref)
    dist_points, indices_points = kdtree.query(transformation[scan])

    kdtree = KDTree(descriptors_ref)
    distances_desc, indices_desc = kdtree.query(descriptors, 2)
    correct_matches = (indices_desc[:, 0] == indices_points.squeeze()) & (
        dist_points.squeeze() < 1e-2
    )

    # noinspection PyArgumentEqualDefault
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.hist(
        distances_desc[correct_matches, 0] / distances_desc[correct_matches, 1],
        bins=50,
        label="Correct matches",
    )
    ax2.hist(
        distances_desc[~correct_matches, 0] / distances_desc[~correct_matches, 1],
        bins=50,
        label="Incorrect matches",
    )
    ax1.legend()
    ax2.legend()
    ax1.set(title="Ratio between the nearest neighbor and the second nearest one")
    ax2.set(title="Ratio between the nearest neighbor and the second nearest one")
    plt.show()
