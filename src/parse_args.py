import argparse


def add_io_parameters(parser) -> None:
    parser.add_argument(
        "--scan_file_path",
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


def add_keypoint_parameters(parser) -> None:
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


def add_descriptor_parameters(parser) -> None:
    parser.add_argument(
        "--descriptor_choice",
        choices=["fpfh", "shot_single_scale", "shot_bi_scale", "shot_multiscale"],
        type=str,
        default="shot_single_scale",
        help="Choice of the descriptor.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1e-2,
        help="Radius in the neighborhood search when computing either FPFH or SHOT.",
    )
    parser.add_argument(
        "--fpfh_n_bins", type=int, default=5, help="Number of bins in FPFH."
    )
    parser.add_argument(
        "--share_local_rfs",
        action="store_true",
        help="Shares the local reference frames between the scales.",
    )
    parser.add_argument(
        "--n_scales",
        type=int,
        default=3,
        help="Number of scales in the multiscale part",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=2,
        help="Parameter that controls the evolution of the radii in the multiscale part",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=5,
        help="Parameter that controls the subsampling size of the support point cloud in the multiscale part",
    )


def add_matching_parameters(parser) -> None:
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
        default=0.001,
        help="Threshold on the distance between inliers in the RANSAC method.",
    )


def add_icp_parameters(parser) -> None:
    parser.add_argument(
        "--icp_type",
        choices=["point_to_point", "point_to_plane"],
        default="point_to_plane",
        help="Type of ICP performed.",
    )
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
        default=0.001,
        help="Size of the voxels in the subsampling performed to select a subset of the points for the ICP.",
    )


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments. Also produces the help message.
    """
    parser = argparse.ArgumentParser(
        description="Feature-based registration using SHOT or FPFH descriptors."
    )
    add_io_parameters(parser.add_argument_group("I/O"))
    add_keypoint_parameters(parser.add_argument_group("Keypoint selection"))
    add_descriptor_parameters(parser.add_argument_group("Descriptors"))
    add_matching_parameters(parser.add_argument_group("Matching and RANSAC"))
    add_icp_parameters(parser.add_argument_group("ICP"))

    return parser.parse_args()
