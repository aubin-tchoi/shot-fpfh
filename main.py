import gc
import warnings
from pathlib import Path

import numpy as np

from src import (
    parse_args,
    get_data,
    write_ply,
    checkpoint,
    check_transform,
    get_incorrect_matches,
    get_transform_from_conf_file,
    RegistrationPipeline,
    compute_normals,
)

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    args = parse_args()

    global_timer = checkpoint()
    timer = checkpoint()
    scan, scan_normals = get_data(
        args.scan_file_path,
        k=args.normals_computation_k,
        normals_computation_callback=compute_normals,
    )
    ref, ref_normals = get_data(
        args.ref_file_path,
        k=args.normals_computation_k,
        normals_computation_callback=compute_normals,
    )

    exact_transformation = None
    try:
        exact_transformation = get_transform_from_conf_file(
            args.conf_file_path, args.scan_file_path, args.ref_file_path
        )
    except (FileNotFoundError, KeyError):
        print(
            f"Conf file not found or incorrect under {args.conf_file_path}, ignoring it."
        )

    run_check_transform = False  # describes the covering between the two point clouds
    if run_check_transform:
        check_transform(scan, ref, exact_transformation)

    timer("Time spent retrieving the data")

    pipeline = RegistrationPipeline(
        scan=scan, scan_normals=scan_normals, ref=ref, ref_normals=ref_normals
    )
    pipeline.select_keypoints(
        args.keypoint_selection,
        neighborhood_size=args.keypoint_voxel_size,
        min_n_neighbors=args.keypoint_density_threshold,
    )
    timer("Time spent selecting the key points")

    pipeline.compute_descriptors(
        descriptor_choice=args.descriptor_choice,
        radius=args.radius,
        fpfh_n_bins=args.fpfh_n_bins,
        disable_progress_bars=args.disable_progress_bars,
    )
    timer(
        f"Time spent computing the descriptors on the reference point cloud ({ref.shape[0]} points)"
    )
    gc.collect()

    pipeline.find_descriptors_matches(
        args.matching_algorithm,
        reject_threshold=args.reject_threshold,
        threshold_multiplier=args.threshold_multiplier,
    )
    timer(f"Time spent finding matches between the descriptors")

    if exact_transformation is not None:
        correct_matches = get_incorrect_matches(
            pipeline.scan[pipeline.scan_keypoints][pipeline.matches[0]],
            pipeline.ref[pipeline.ref_keypoints][pipeline.matches[1]],
            exact_transformation,
        )
        print(
            f"{correct_matches.sum()} correct matches out of {pipeline.scan_descriptors.shape[0]} descriptors."
        )

    transformation_ransac, inliers_ratio = pipeline.run_ransac(
        n_draws=args.n_ransac_draws,
        draw_size=args.ransac_draw_size,
        max_inliers_distance=args.ransac_max_inliers_dist,
        exact_transformation=exact_transformation,
        disable_progress_bar=args.disable_progress_bars,
    )
    timer("Time spent on RANSAC")
    print(
        f"\nRatio of inliers pre-ICP: {inliers_ratio * 100:.2f}%\nTransformation pre-ICP:"
    )
    print(transformation_ransac)
    gc.collect()

    transformation_icp, distance_to_map, has_icp_converged = pipeline.run_icp(
        args.icp_type,
        transformation_ransac,
        d_max=args.icp_d_max,
        voxel_size=args.icp_voxel_size,
        max_iter=args.icp_max_iter,
        rms_threshold=args.icp_rms_threshold,
        disable_progress_bar=args.disable_progress_bars,
    )
    overlap, inliers_ratio_post_icp = pipeline.compute_metrics_post_icp(
        transformation_icp,
        args.icp_d_max,
    )
    timer("Time spent on ICP")
    print(
        f"\nRMS error on the ICP:  {distance_to_map:.4f} (has converged: {'OK'[::has_icp_converged * 2 - 1]})."
        f"\nProportion of inliers: {inliers_ratio_post_icp * 100:.2f}%"
        f"\nPercentage of overlap: {overlap * 100:.2f}%"
        f"\nTransformation post-ICP:"
    )
    print(transformation_icp)

    if not args.disable_ply_writing:
        print(
            "\n -- Writing the aligned points cloud in ply files under './data/results' --"
        )
        (results_folder := Path("./data/results")).mkdir(exist_ok=True, parents=True)
        file_name = (
            f"{Path(args.scan_file_path).stem}_on_{Path(args.ref_file_path).stem}"
        )
        is_scan = np.hstack(
            (
                np.ones(pipeline.scan.shape[0], dtype=bool),
                np.zeros(pipeline.ref.shape[0], dtype=bool),
            )
        )[:, None]
        write_ply(
            str(results_folder / f"{file_name}_post_ransac.ply"),
            [
                np.hstack(
                    (
                        np.vstack((transformation_ransac[pipeline.scan], pipeline.ref)),
                        is_scan,
                    )
                )
            ],
            ["x", "y", "z", "is_scan"],
        )
        write_ply(
            str(results_folder / f"{file_name}_post_icp.ply"),
            [
                np.hstack(
                    (
                        np.vstack((transformation_icp[pipeline.scan], pipeline.ref)),
                        is_scan,
                    )
                )
            ],
            ["x", "y", "z", "is_scan"],
        )

    global_timer("\nTotal time spent")
