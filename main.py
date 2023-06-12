import gc
import os
import warnings

from src import (
    parse_args,
    get_data,
    write_ply,
    checkpoint,
    check_transform,
    get_incorrect_matches,
    get_transform_from_conf_file,
    RegistrationPipeline,
)

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    args = parse_args()

    global_timer = checkpoint()
    timer = checkpoint()
    scan, scan_normals = get_data(args.file_path, radius=args.fpfh_radius)
    ref, ref_normals = get_data(args.ref_file_path, radius=args.fpfh_radius)

    exact_transformation = None
    try:
        exact_transformation = get_transform_from_conf_file(
            args.conf_file_path, args.file_path, args.ref_file_path
        )
    except FileNotFoundError:
        print(f"Conf file not found under {args.conf_file_path}, ignoring it.")

    run_check_transform = False  # describes the covering between the two point clouds
    if run_check_transform:
        check_transform(scan, ref, exact_transformation)

    timer("Time spent retrieving the data")

    pipeline = RegistrationPipeline(
        scan=scan, scan_normals=scan_normals, ref=ref, ref_normals=ref_normals
    )
    pipeline.select_keypoints(
        args.keypoint_selection,
        neighborhood_size=args.shot_radius / 50,
        min_n_neighbors=args.min_n_neighbors,
    )
    timer("Time spent selecting the key points")

    pipeline.apply_filter_on_keypoints(
        normals_z_threshold=args.normals_z_threshold,
        sphericity_threshold=args.sphericity_threshold,
        sphericity_computation_radius=args.sphericity_computation_radius,
    )
    timer("Time spent filtering the key points")

    pipeline.compute_descriptors(
        args.descriptor_choice, radius=args.radius, n_bins=args.fpfh_n_bins
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

    transformation_ransac, inliers_ratio = pipeline.run_ransac()
    timer("Time spent on RANSAC")
    print(
        f"\nRatio of inliers pre-ICP: {inliers_ratio * 100:.2f}%\nTransformation pre-ICP:"
    )
    print(transformation_ransac)
    gc.collect()

    transformation_icp, distance_to_map, has_icp_converged = pipeline.run_icp(
        args.icp_type, transformation_ransac, d_max=args.icp_d_max
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
        if not os.path.isdir("./data/results"):
            os.mkdir("./data/results")

        file_name = args.file_path.split("/")[-1].replace(".ply", "")
        write_ply(
            f"./data/results/{file_name}_post_ransac.ply",
            [transformation_ransac[pipeline.scan]],
            ["x", "y", "z"],
        )
        write_ply(
            f"./data/results/{file_name}_post_icp.ply",
            [transformation_icp[pipeline.scan]],
            ["x", "y", "z"],
        )

    global_timer("\nTotal time spent")
