import gc
import os
import warnings

from src import (
    parse_args,
    get_data,
    write_ply,
    checkpoint,
    check_transform,
    get_transform_from_conf_file,
    solver_point_to_point,
    compute_point_to_point_error,
    # query points sampling
    select_query_indices_randomly,
    select_keypoints_iteratively,
    select_keypoints_subsampling,
    # descriptors
    compute_shot_descriptor,
    compute_fpfh_descriptor,
    # matching algorithms
    basic_matching,
    double_matching_with_rejects,
    ransac_on_matches,
    icp_point_to_point_with_sampling,
    get_resulting_transform,
    read_conf_file,
    count_correct_matches,
    plot_distance_hists,
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

    if args.query_points_selection == "random":
        print("\n-- Selecting query points randomly --")
        points_subset = select_query_indices_randomly(scan.shape[0], scan.shape[0] // 1)
        points_ref_subset = select_query_indices_randomly(
            ref.shape[0], ref.shape[0] // 1
        )
    elif args.query_points_selection == "iterative":
        print("\n-- Selecting query points iteratively --")
        points_subset = select_keypoints_iteratively(scan, args.shot_radius / 50)
        points_ref_subset = select_keypoints_iteratively(ref, args.shot_radius / 50)
    elif args.query_points_selection == "subsampling":
        print(
            "\n-- Selecting query points based on a subsampling on the point cloud --"
        )
        points_subset = select_keypoints_subsampling(scan, args.shot_radius / 50)
        points_ref_subset = select_keypoints_subsampling(ref, args.shot_radius / 50)
    else:
        raise ValueError("Incorrect query points selection algorithm.")
    timer("Time spent selecting the key points")

    print("\n-- Computing the descriptors --")
    if args.shot_first:
        shot = compute_shot_descriptor(
            scan[points_subset], scan, scan_normals, radius=args.shot_radius
        )
        fpfh = compute_fpfh_descriptor(
            points_subset,
            scan,
            scan_normals,
            radius=args.fpfh_radius,
            n_bins=args.fpfh_n_bins,
        )
    else:
        fpfh = compute_fpfh_descriptor(
            points_subset,
            scan,
            scan_normals,
            radius=args.fpfh_radius,
            n_bins=args.fpfh_n_bins,
        )
        shot = compute_shot_descriptor(
            scan[points_subset], scan, scan_normals, radius=args.shot_radius
        )
    timer(
        f"Time spent computing the descriptors on the point cloud to align ({scan.shape[0]} points)"
    )

    if args.shot_first:
        shot_ref = compute_shot_descriptor(
            ref[points_ref_subset],
            ref,
            ref_normals,
            radius=args.shot_radius,
        )
        fpfh_ref = compute_fpfh_descriptor(
            points_ref_subset,
            ref,
            ref_normals,
            radius=args.fpfh_radius,
            n_bins=args.fpfh_n_bins,
        )
    else:
        fpfh_ref = compute_fpfh_descriptor(
            points_ref_subset,
            ref,
            ref_normals,
            radius=args.fpfh_radius,
            n_bins=args.fpfh_n_bins,
        )
        shot_ref = compute_shot_descriptor(
            ref[points_ref_subset],
            ref,
            ref_normals,
            radius=args.shot_radius,
        )
    timer(
        f"Time spent computing the descriptors on the reference point cloud ({ref.shape[0]} points)"
    )
    gc.collect()

    # matching
    if args.matching_algorithm == "simple":
        print("\n-- Matching descriptors using simple matching to closest neighbor --")
        matches_fpfh = basic_matching(fpfh, fpfh_ref)
        timer("Time spent finding matches between the FPFH descriptors")

        # validation
        assert (
            matches_fpfh.shape[0] == fpfh.shape[0]
        ), "Not as many matches as FPFH descriptors"
        if use_conf_file:
            n_correct_matches_fpfh = count_correct_matches(
                scan[points_subset],
                ref[points_ref_subset][matches_fpfh],
                exact_transformation,
            )
            print(
                f"FPFH: {n_correct_matches_fpfh} correct matches out of {fpfh.shape[0]} descriptors."
            )

        rms_fpfh, points_aligned_fpfh = compute_point_to_point_error(
            scan,
            ref,
            solver_point_to_point(
                scan[points_subset],
                ref[points_ref_subset][matches_fpfh],
            ),
        )
        timer()

        matches_shot = basic_matching(shot, shot_ref)
        timer("Time spent finding matches between the SHOT descriptors")
        assert (
            matches_shot.shape[0] == shot.shape[0]
        ), "Not as many matches as SHOT descriptors"
        if use_conf_file:
            n_correct_matches_shot = count_correct_matches(
                scan[points_subset],
                ref[points_ref_subset][matches_shot],
                exact_transformation,
            )
            print(
                f"SHOT: {n_correct_matches_shot} correct matches out of {shot.shape[0]} descriptors."
            )

        rms_shot, points_aligned_shot = compute_point_to_point_error(
            scan,
            ref,
            solver_point_to_point(
                scan[points_subset],
                ref[points_ref_subset][matches_shot],
            ),
        )
        timer()
    elif args.matching_algorithm == "double":
        print("\n-- Matching descriptors using double matching with rejects --")
        matches_fpfh, matches_fpfh_ref = double_matching_with_rejects(
            fpfh, fpfh_ref, args.reject_threshold
        )
        timer("Time spent finding matches between the FPFH descriptors")

        if use_conf_file:
            n_correct_matches_fpfh = count_correct_matches(
                scan[points_subset][matches_fpfh],
                ref[points_ref_subset][matches_fpfh_ref],
                exact_transformation,
            )
            print(
                f"FPFH: {n_correct_matches_fpfh} correct matches out of {matches_fpfh.shape[0]} matches."
            )
            plot_distance_hists(
                scan[points_subset],
                ref[points_ref_subset],
                exact_transformation,
                fpfh,
                fpfh_ref,
            )

        rms_fpfh, points_aligned_fpfh = compute_point_to_point_error(
            scan,
            ref,
            solver_point_to_point(
                scan[points_subset][matches_fpfh],
                ref[points_ref_subset][matches_fpfh_ref],
            ),
        )
        timer()

        matches_shot, matches_shot_ref = double_matching_with_rejects(
            shot, shot_ref, args.reject_threshold
        )
        timer("Time spent finding matches between the SHOT descriptors")

        if use_conf_file:
            n_correct_matches_shot = count_correct_matches(
                scan[points_subset][matches_shot],
                ref[points_ref_subset][matches_shot_ref],
                exact_transformation,
            )
            print(
                f"SHOT: {n_correct_matches_shot} correct matches out of {matches_shot.shape[0]} matches."
            )
            plot_distance_hists(
                scan[points_subset],
                ref[points_ref_subset],
                exact_transformation,
                shot,
                shot_ref,
            )

        rms_shot, points_aligned_shot = compute_point_to_point_error(
            scan,
            ref,
            solver_point_to_point(
                scan[points_subset][matches_shot],
                ref[points_ref_subset][matches_shot_ref],
            ),
        )
        timer()
    elif args.matching_algorithm == "ransac":
        print("\n -- Matching descriptors using ransac-like matching --")
        rms_fpfh, transformation_fpfh = ransac_on_matches(
            fpfh,
            scan,
            scan[points_subset],
            fpfh_ref,
            ref,
            ref[points_ref_subset],
        )
        timer("Time spent finding matches between the FPFH descriptors")
        points_aligned_fpfh = transformation_fpfh[scan]
        if use_conf_file:
            rotation_diff = (
                transformation_fpfh.rotation @ exact_transformation.rotation.T
            )
            print(
                f"Norm of the angle between the two rotations: "
                f"{np.abs(np.arccos((np.trace(rotation_diff) - 1) / 2)):.2f}\n"
                f"Norm of the difference between the two translation: "
                f"{np.linalg.norm(transformation_fpfh.translation - exact_transformation.translation):.2f}"
            )
        timer()

        rms_shot, transformation_shot = ransac_on_matches(
            shot,
            scan,
            scan[points_subset],
            shot_ref,
            ref,
            ref[points_ref_subset],
        )
        timer("Time spent finding matches between the SHOT descriptors")
        points_aligned_shot = transformation_shot[scan]
        if use_conf_file:
            rotation_diff = (
                transformation_shot.rotation @ exact_transformation.rotation.T
            )
            print(
                f"Norm of the angle between the two rotations: "
                f"{np.abs(np.arccos((np.trace(rotation_diff) - 1) / 2)):.2f}\n"
                f"Norm of the difference between the two translation: "
                f"{np.linalg.norm(transformation_shot.translation - exact_transformation.translation):.2f}"
            )
        timer()
    else:
        raise ValueError("Incorrect matching algorithm selection.")
    gc.collect()

    print(f"\nRMS error with FPFH: {rms_fpfh:.2f}")
    print(f"RMS error with SHOT: {rms_shot:.2f}")

    # fine registration using an icp
    print("\n-- Running ICPs --")
    (
        points_aligned_fpfh_icp,
        rms_fpfh_icp,
        has_fpfh_converged,
    ) = icp_point_to_point_with_sampling(points_aligned_fpfh, ref, args.icp_d_max)
    (
        points_aligned_shot_icp,
        rms_shot_icp,
        has_shot_converged,
    ) = icp_point_to_point_with_sampling(points_aligned_shot, ref, args.icp_d_max)
    print(f"RMS error with FPFH + ICP: {rms_fpfh_icp:.2f}")
    print(f"RMS error with SHOT + ICP: {rms_shot_icp:.2f}")

    print(
        f"The ICP starting from the registration obtained by matching"
        f" FPFH descriptors has{'' if has_fpfh_converged else ' not'} converged."
    )
    print(
        f"The ICP starting from the registration obtained by matching"
        f" SHOT descriptors has{'' if has_shot_converged else ' not'} converged."
    )
    timer("Time spent on ICP")

    if not args.disable_ply_writing:
        print(
            "\n -- Writing the aligned points cloud in ply files under './data/results' --"
        )
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

    global_timer("\nTotal time spent")
