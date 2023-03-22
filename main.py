import argparse
import warnings
import os

# noinspection PyUnresolvedReferences
from src import (
    get_data,
    write_ply,
    checkpoint,
    best_rigid_transform,
    compute_rigid_transform_error,
    # query points sampling
    select_query_points,
    select_query_points_randomly,
    select_query_indices_randomly,
    # descriptors
    compute_shot_descriptor,
    compute_fpfh_descriptor,
    compute_pca_based_features,
    # matching algorithms
    basic_matching,
    double_matching_with_rejects,
    ransac_matching,
    icp_point_to_point,
)

warnings.filterwarnings("ignore")


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
        "--disable_ply_writing",
        action="store_true",
        help="Skips saving the registered point cloud as ply files.",
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
        "--fpfh_k",
        type=int,
        default=12,
        help="Number of neighbors in the neighborhood search when computing FPFH.",
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    timer = checkpoint()
    points, normals = get_data(args.file_path)
    points_ref, normals_ref = get_data(args.ref_file_path)
    timer("Time spent retrieving the data")

    points_subset = select_query_indices_randomly(points.shape[0], points.shape[0] // 2)
    points_ref_subset = select_query_indices_randomly(
        points_ref.shape[0], points_ref.shape[0] // 2
    )
    timer("Time spent selecting the key points")

    fpfh = compute_fpfh_descriptor(
        points_subset,
        points,
        normals,
        radius=args.fpfh_radius,
        k=args.fpfh_k,
        n_bins=args.fpfh_n_bins,
    )
    shot = compute_shot_descriptor(
        points[points_subset], points, normals, radius=args.shot_radius
    )
    timer(
        f"Time spent computing the descriptors on the point cloud to align ({points.shape[0]} points)"
    )

    fpfh_ref = compute_fpfh_descriptor(
        points_ref_subset,
        points_ref,
        normals_ref,
        radius=args.fpfh_radius,
        k=args.fpfh_k,
        n_bins=args.fpfh_n_bins,
    )
    shot_ref = compute_shot_descriptor(
        points_ref[points_ref_subset], points_ref, normals_ref, radius=args.shot_radius
    )
    timer(
        f"Time spent computing the descriptors on the reference point cloud ({points_ref.shape[0]} points)"
    )

    if args.matching_algorithm == "simple":
        print("Matching descriptors using simple matching to closest neighbor.")
        matches_fpfh = basic_matching(fpfh, fpfh_ref)
        timer("Time spent finding matches between the FPFH descriptors")

        rms_fpfh, points_aligned_fpfh = compute_rigid_transform_error(
            points,
            points_ref,
            *best_rigid_transform(
                points[points_subset],
                points_ref[points_ref_subset][matches_fpfh],
            ),
        )

        matches_shot = basic_matching(shot, shot_ref)
        timer("Time spent finding matches between the SHOT descriptors")

        rms_shot, points_aligned_shot = compute_rigid_transform_error(
            points,
            points_ref,
            *best_rigid_transform(
                points[points_subset],
                points_ref[points_ref_subset][matches_shot],
            ),
        )
    elif args.matching_algorithm == "ransac":
        print("Matching descriptors using ransac-type matching.")
        rms_fpfh, (rotation, translation) = ransac_matching(
            fpfh,
            points,
            points[points_subset],
            fpfh_ref,
            points_ref,
            points_ref[points_ref_subset],
        )
        timer("Time spent finding matches between the FPFH descriptors")
        points_aligned_fpfh = points.dot(rotation.T) + translation
        timer()

        rms_shot, (rotation, translation) = ransac_matching(
            shot,
            points,
            points[points_subset],
            shot_ref,
            points_ref,
            points_ref[points_ref_subset],
        )
        timer("Time spent finding matches between the SHOT descriptors")
        points_aligned_shot = points.dot(rotation.T) + translation
        timer()
    elif args.matching_algorithm == "double":
        print("Matching descriptors using double matching with rejects.")
        matches_fpfh, matches_fpfh_ref = double_matching_with_rejects(
            fpfh, fpfh_ref, args.reject_threshold
        )
        timer("Time spent finding matches between the FPFH descriptors")

        rms_fpfh, points_aligned_fpfh = compute_rigid_transform_error(
            points,
            points_ref,
            *best_rigid_transform(
                points[points_subset][matches_fpfh],
                points_ref[points_ref_subset][matches_fpfh_ref],
            ),
        )
        timer()

        matches_shot, matches_shot_ref = double_matching_with_rejects(
            shot, shot_ref, args.reject_threshold
        )
        timer("Time spent finding matches between the SHOT descriptors")

        rms_shot, points_aligned_shot = compute_rigid_transform_error(
            points,
            points_ref,
            *best_rigid_transform(
                points[points_subset][matches_shot],
                points_ref[points_ref_subset][matches_shot_ref],
            ),
        )
        timer()
    else:
        raise ValueError("Incorrect matching algorithm selection.")

    print(f"RMS error with FPFH: {rms_fpfh:.2f}.")
    print(f"RMS error with SHOT: {rms_shot:.2f}.")

    points_aligned_fpfh_icp, has_fpfh_converged = icp_point_to_point(
        points_aligned_fpfh, points_ref
    )
    points_aligned_shot_icp, has_shot_converged = icp_point_to_point(
        points_aligned_shot, points_ref
    )

    print(
        f"The ICP starting from the registration obtained by matching"
        f" FPFH descriptors has{'' if has_fpfh_converged else ' not'} converged."
    )
    print(
        f"The ICP starting from the registration obtained by matching"
        f" SHOT descriptors has{'' if has_shot_converged else ' not'} converged."
    )

    if not args.disable_ply_writing:
        if not os.path.isdir("./data/results"):
            os.mkdir("./data/results")

        file_name = args.file_path.split("/")[-1].replace(".ply", "")
        write_ply(
            f"./data/results/{file_name}_registered_fpfh-{args.matching_algorithm}.ply",
            [points_aligned_fpfh],
            ["x", "y", "z"],
        )
        write_ply(
            f"./data/results/{file_name}_registered_shot-{args.matching_algorithm}.ply",
            [points_aligned_shot],
            ["x", "y", "z"],
        )
        write_ply(
            f"./data/results/{file_name}_registered_fpfh_icp-{args.matching_algorithm}.ply",
            [points_aligned_fpfh_icp],
            ["x", "y", "z"],
        )
        write_ply(
            f"./data/results/{file_name}_registered_shot_icp-{args.matching_algorithm}.ply",
            [points_aligned_shot_icp],
            ["x", "y", "z"],
        )
