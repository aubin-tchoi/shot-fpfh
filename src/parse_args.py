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
        help="Path to the first point cloud to use.",
    )
    parser.add_argument(
        "--ref_file_path",
        type=str,
        default="./data/bunny_original.ply",
        help="Path to the second point cloud to use (reference point cloud).",
    )
    parser.add_argument(
        "--conf_file_path",
        type=str,
        default="./data/bunny/bun.conf",
        help="Path to the .conf file used to count correct matches. Leave empty to ignore.",
    )
    parser.add_argument(
        "--disable_ply_writing",
        action="store_true",
        help="Skips saving the registered point cloud as ply files.",
    )
    parser.add_argument(
        "--disable_progress_bars",
        action="store_true",
        help="Disables the progress bars in SHOT computation, RANSAC and ICP.",
    )
    # keypoint selection
    parser.add_argument(
        "--keypoint_selection",
        choices=["random", "iterative", "subsampling"],
        type=str,
        default="subsampling",
        help="Choice of the algorithm to select keypoints to compute descriptors on.",
    )
    parser.add_argument(
        "--keypoint_voxel_size",
        type=float,
        default=0.005,
        help="Size of the voxels in the subsampling-based keypoint selection.",
    )
    parser.add_argument(
        "--keypoint_density_threshold",
        type=int,
        default=100,
        help="Minimum number of neighbors used to filter keypoints.",
    )
    parser.add_argument(
        "--normals_computation_k",
        type=int,
        default=30,
        help="Number of neighbors used to compute normals.",
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
        "--descriptor_choice",
        choices=["fpfh", "shot"],
        type=str,
        default="shot",
        help="Choice of the descriptor.",
    )
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
    # RANSAC
    parser.add_argument(
        "--n_ransac_draws",
        type=int,
        default=5000,
        help="Number of draws in the RANSAC method for descriptors matching.",
    )
    parser.add_argument(
        "--ransac_draw_size",
        type=int,
        default=4,
        help="Size of the draws in the RANSAC method for descriptors matching.",
    )
    parser.add_argument(
        "--ransac_max_inliers_dist",
        type=float,
        default=1,
        help="Threshold on the distance between inliers in the RANSAC method.",
    )
    # ICP
    parser.add_argument(
        "--icp_max_iter",
        type=int,
        default=70,
        help="Maximum number of iteration in the ICP.",
    )
    parser.add_argument(
        "--icp_rms_threshold",
        type=float,
        default=0.01,
        help="RMS threshold set on the ICP.",
    )
    parser.add_argument(
        "--icp_d_max",
        type=float,
        default=1,
        help="Maximum distance between two inliers in the ICP.",
    )
    parser.add_argument(
        "--icp_voxel_size",
        type=float,
        default=0.8,
        help="Size of the voxels in the subsampling performed to select a subset of the points for the ICP.",
    )

    return parser.parse_args()
