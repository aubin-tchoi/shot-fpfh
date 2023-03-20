import argparse

import numpy as np

from src import get_data, compute_shot_descriptor, compute_fpfh_descriptor


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

    points, normals = get_data(args.file_path)

    compute_fpfh_descriptor(points, points, normals, radius=5 * 1e-3, k=12, n_bins=5)
    compute_shot_descriptor(points, points, normals, radius=5 * 1e-3)
