import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree

from shot_fpfh.core import RigidTransform


def quaternion_to_rotation_matrix(
    quaternion: list[float],
) -> np.ndarray[np.float64, (3, 3)]:
    """
    Converts a quaternion into a full three-dimensional rotation matrix.
    """
    q3, q0, q1, q2 = quaternion
    return Rotation.from_quat([q0, q1, q2, q3]).as_matrix().squeeze()


def read_conf_file(file_path: str) -> dict[str, RigidTransform]:
    """
    Reads a .conf file from the Stanford 3D Scanning Repository to output the (translation, rotation) for each ply file.
    """
    with open(file_path) as file:
        transformations = {
            line_values[1].replace(".ply", ""): RigidTransform(
                quaternion_to_rotation_matrix([float(val) for val in line_values[5:]]),
                np.array([float(val) for val in line_values[2:5]]),
            )
            for line in file.readlines()
            if (line_values := line.split(" "))[0] == "bmesh"
        }
    return transformations


def get_transform_from_conf_file(
    conf_file_name: str, scan_file_name: str, ref_file_name: str
) -> RigidTransform:
    """
    Outputs the rotation and translation that send the data corresponding to file_name onto the data corresponding to
    ref_file_name using a config that contains rotations and translations to send them to a common coordinate system.
    """
    conf = read_conf_file(conf_file_name)

    return (
        conf[ref_file_name.split("/")[-1].replace(".ply", "")].inv()
        @ conf[scan_file_name.split("/")[-1].replace(".ply", "")]
    )


def check_transform(
    scan: npt.NDArray[np.float64],
    ref: npt.NDArray[np.float64],
    transformation: RigidTransform,
) -> None:
    """
    Describes the covering between the two point clouds knowing the exact transformation between the two.
    """
    aligned_points = transformation[scan]
    plt.hist(KDTree(ref).query(aligned_points)[0].squeeze(), bins=100)
    plt.show()
