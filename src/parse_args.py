import argparse


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments. Also produces the help message.
    """
    parser = argparse.ArgumentParser(
        description="Feature-based registration using SHOT and FPFH descriptors."
    )

    parser.add_argument(
        "--file_path",
        type=str,
        default="./data/bunny_returned.ply",
        help="Path to the first point cloud to use",
    )
    parser.add_argument(
        "--ref_file_path",
        type=str,
        default="./data/bunny_original.ply",
        help="Path to the second point cloud to use (reference point cloud)",
    )
    parser.add_argument(
        "--conf_file_path",
        type=str,
        default="./data/bunny/bun.conf",
        help="Path to the .conf file used to count correct matches. Leave empty to ignore",
    )
    parser.add_argument(
        "--disable_ply_writing",
        action="store_true",
        help="Skips saving the registered point cloud as ply files.",
    )
    parser.add_argument(
        "--shot_first",
        action="store_true",
        help="Computes SHOT first and FPFH second.",
    )
    parser.add_argument(
        "--query_points_selection",
        choices=["random", "iterative", "subsampling"],
        type=str,
        default="random",
        help="Choice of the algorithm to select query points to compute descriptors on.",
    )
    parser.add_argument(
        "--matching_algorithm",
        choices=["simple", "double", "ransac"],
        type=str,
        default="simple",
        help="Choice of the algorithm to match descriptors.",
    )
    parser.add_argument(
        "--fpfh_radius",
        type=float,
        default=1e-2,
        help="Radius in the neighborhood search when computing SPFH.",
    )
    parser.add_argument(
        "--fpfh_n_bins", type=int, default=5, help="Number of bins in FPFH."
    )
    parser.add_argument(
        "--shot_radius",
        type=float,
        default=1e-1,
        help="Radius in the neighborhood search when computing SHOT.",
    )
    parser.add_argument(
        "--reject_threshold",
        type=float,
        default=0.8,
        help="Threshold in the double matching algorithm.",
    )
    parser.add_argument(
        "--icp_d_max",
        type=float,
        default=1e-2,
        help="Maximum distance between two inliers in the ICP.",
    )

    return parser.parse_args()
