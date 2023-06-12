import argparse


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments. Also produces the help message.
    """
    parser = argparse.ArgumentParser(
        description="Feature-based registration using SHOT and FPFH descriptors."
    )

    # i/o
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
    # keypoint selection
    parser.add_argument(
        "--keypoint_selection",
        choices=["random", "iterative", "subsampling"],
        type=str,
        default="random",
        help="Choice of the algorithm to select keypoints to compute descriptors on.",
    )
    parser.add_argument(
        "--normals_z_threshold",
        type=float,
        default=0.8,
        help="Threshold used in the keypoint selection process on the vertical component of the normals.",
    )
    parser.add_argument(
        "--sphericity_threshold",
        type=float,
        default=0.16,
        help="Threshold used in the keypoint selection process on the sphericity.",
    )
    parser.add_argument(
        "--sphericity_computation_radius",
        type=float,
        default=0.3,
        help="Radius used in the keypoint selection process to compute the sphericity.",
    )
    # descriptors
    parser.add_argument(
        "--radius",
        type=float,
        default=1e-1,
        help="Radius in the neighborhood search when computing either FPFH or SHOT.",
    )
    parser.add_argument(
        "--fpfh_n_bins", type=int, default=5, help="Number of bins in FPFH."
    )
    # matching
    parser.add_argument(
        "--matching_algorithm",
        choices=["simple", "double", "ransac"],
        type=str,
        default="simple",
        help="Choice of the algorithm to match descriptors.",
    )
    parser.add_argument(
        "--reject_threshold",
        type=float,
        default=0.8,
        help="Threshold in the double matching algorithm.",
    )
    parser.add_argument(
        "--threshold_multiplier",
        type=float,
        default=10,
        help="Threshold multiplier in the threshold-based matching algorithm.",
    )
    # icp
    parser.add_argument(
        "--icp_d_max",
        type=float,
        default=1e-2,
        help="Maximum distance between two inliers in the ICP.",
    )

    return parser.parse_args()
