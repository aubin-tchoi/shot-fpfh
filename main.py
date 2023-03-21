import argparse
import warnings

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
        default="./data/bunny_perturbed.ply",
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    timer = checkpoint()
    points, normals = get_data(args.file_path)
    points_ref, normals_ref = get_data(args.ref_file_path)
    timer("Time spent retrieving the data")

    points_subset = select_query_indices_randomly(
        points.shape[0], points.shape[0] // 10
    )
    points_ref_subset = select_query_indices_randomly(
        points_ref.shape[0], points_ref.shape[0] // 10
    )
    timer("Time spent selecting the key points")

    fpfh = compute_fpfh_descriptor(
        points_subset, points, normals, radius=5 * 1e-3, k=12, n_bins=5
    )
    shot = compute_shot_descriptor(
        points[points_subset], points, normals, radius=5 * 1e-3
    )
    timer(
        f"Time spent computing the descriptors on the point cloud to align ({points.shape[0]} points)"
    )

    fpfh_ref = compute_fpfh_descriptor(
        points_ref_subset, points_ref, normals_ref, radius=5 * 1e-3, k=12, n_bins=5
    )
    shot_ref = compute_shot_descriptor(
        points_ref[points_ref_subset], points_ref, normals_ref, radius=5 * 1e-3
    )
    timer(
        f"Time spent computing the descriptors on the reference point cloud ({points_ref.shape[0]} points)"
    )

    # choose the matching algorithm here
    matches_fpfh, matches_fpfh_ref = double_matching_with_rejects(fpfh, fpfh_ref, 0.8)
    timer("Time spent finding matches between the FPFH descriptors")

    rms_fpfh, points_aligned_fpfh = compute_rigid_transform_error(
        points,
        points_ref,
        *best_rigid_transform(
            points[points_subset][matches_fpfh],
            points_ref[points_ref_subset][matches_fpfh_ref],
        ),
    )
    print(f"RMS error with FPFH: {rms_fpfh:.2f}.")

    matches_shot, matches_shot_ref = double_matching_with_rejects(shot, shot_ref, 0.8)
    timer("Time spent finding matches between the SHOT descriptors")

    rms_shot, points_aligned_shot = compute_rigid_transform_error(
        points,
        points_ref,
        *best_rigid_transform(
            points[points_subset][matches_shot],
            points_ref[points_ref_subset][matches_shot_ref],
        ),
    )
    print(f"RMS error with SHOT: {rms_shot:.2f}.")

    if not args.disable_ply_writing:
        write_ply(
            "./data/bunny_registered_fpfh.ply",
            [points_aligned_fpfh],
            ["x", "y", "z"],
        )
        write_ply(
            "./data/bunny_registered_shot.ply",
            [points_aligned_shot],
            ["x", "y", "z"],
        )
