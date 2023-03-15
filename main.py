import argparse

import numpy as np

from src import read_ply, compute_shot_descriptor


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments. Also produces the help message.
    """
    parser = argparse.ArgumentParser(
        description="SHOT descriptor performance evaluation"
    )

    parser.add_argument(
        "--file_path",
        type=str,
        default="./data/bunny_normals.ply",
        help="Path to the point cloud to use",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data = read_ply(args.file_path)
    points = np.vstack((data["x"], data["y"], data["z"])).T
    normals = np.vstack((data["nx"], data["ny"], data["nz"])).T

    compute_shot_descriptor(points, points, normals, radius=1e-3)
