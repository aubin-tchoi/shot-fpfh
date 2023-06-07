"""
Base class to handle 4x4 transformation.
"""
import numpy as np

from scipy.spatial.transform import Rotation


class Transformation:
    """
    Class to wrap 4x4 transformations.
    """

    def __init__(
        self,
        rotation: np.ndarray[np.float64, (3, 3)] = np.eye(3),
        translation: np.ndarray[np.float64, 3] = np.zeros(3),
    ):
        self.rotation = rotation
        self.translation = translation

    def __repr__(self) -> str:
        """
        Returns a string representation without scientific notation that can almost directly be copy-pasted into
        CloudCompare.

        Returns:
            A string representation of the 4x4 transformation matrix.
        """
        with np.printoptions(suppress=True):
            return str(
                np.vstack(
                    (
                        np.hstack((self.rotation, self.translation[:, None])),
                        np.array([0, 0, 0, 1]),
                    )
                )
            )

    def normalize_rotation(self) -> None:
        """
        Normalizes the rotation by normalizing the quaternions.
        """
        rotation_quat = Rotation.from_matrix(self.rotation).as_quat()
        self.rotation = Rotation.from_quat(
            rotation_quat / np.linalg.norm(rotation_quat)
        ).as_matrix()

    def __matmul__(self, other_transformation):
        """
        Matrix composition of two transformations.

        Args:
            other_transformation: The transformation on the right (the first one to apply).

        Returns:
            The matrix product of the two transformations.
        """
        product = Transformation(
            self.rotation @ other_transformation.rotation,
            self.rotation @ other_transformation.translation + self.translation,
        )
        product.normalize_rotation()

        return product

    def __invert__(self):
        """
        Inverts the transformation by inverting the rotation and taking the opposite of the translation.

        Returns:
            The Transformation corresponding to the inverse transformation.
        """
        return Transformation(self.rotation.T, -self.translation)

    def __getitem__(self, points: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """
        Applies the transformation to a line-representation of a point or to an (N, 3) array of N points.

        Returns:
            The transformed points.
        """
        return points.dot(self.rotation.T) + self.translation

    def transform(self, points: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """
        Applies the transformation to a line-representation of a point or to an (N, 3) array of N points.

        Returns:
            The transformed points.
        """
        return self[points]

    def inv(self):
        """
        Inverts the transformation by inverting the rotation and taking the opposite of the translation.

        Returns:
            The Transformation corresponding to the inverse transformation.
        """
        return ~self
