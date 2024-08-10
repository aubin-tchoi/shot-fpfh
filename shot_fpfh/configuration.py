"""
Classes that can contain the configuration required by every step of the pipeline.
"""

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, TypedDict

import yaml


@dataclass
class Config(ABC):
    """Base class that describes the structure of every config class and implements a type casting behavior."""

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            try:
                if not isinstance(value, field.type):
                    warnings.warn(
                        f"Expected {field.name} to be {field.type}, got {repr(value)} of type {type(value)}"
                    )
                    setattr(self, field.name, field.type(value))  # recasting the value
            except TypeError:
                ...

    def __repr__(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @abstractmethod
    def help_message(self) -> str:
        """
        Creates a help message describing the behavior of the method with the set of parameters specified.

        Returns:
            The help message.
        """
        ...


@dataclass
class KeypointSelectionConfig(Config):
    """Parameters involved in the selection of keypoints."""

    selection_algorithm: Literal[
        "random", "iterative", "subsampling", "subsampling_with_density"
    ] = "subsampling_with_density"
    neighborhood_size: float | None = None
    min_n_neighbors: int | None = None

    def help_message(self) -> str:
        return (
            f"Keypoint selection parameters:\n"
            f" -- strategy: {self.selection_algorithm}\n"
            f" -- neighborhood size: {self.neighborhood_size}\n"
            f" -- minimal number of neighbors: {self.min_n_neighbors}"
        )


@dataclass
class DescriptorConfig(Config):
    """Parameters that control the computation of the descriptors."""

    radius: float = 3.0
    descriptor_choice: Literal[
        "fpfh", "shot_single_scale", "shot_bi_scale", "shot_multiscale"
    ] = "shot_single_scale"
    fpfh_n_bins: int = 5
    phi: float = 3.0
    rho: float = 10.0
    n_scales: int = 2
    subsample_support: bool = True
    normalize: bool = True
    share_local_rfs: bool = True
    min_neighborhood_size: int = 100
    n_procs: int = 8
    disable_progress_bars: bool = False

    def help_message(self) -> str:
        match self.descriptor_choice:
            case "shot_single_scale":
                return (
                    f"SHOT parameters:\n"
                    f" -- radius: {self.radius}\n"
                    f" -- minimum neighborhood size: {self.min_neighborhood_size}\n"
                    f" -- the descriptors will{' not' if not self.normalize else ''} be normalized.\n"
                    f" -- the support will{' not' if not self.subsample_support else ''} be subsampled"
                    f"{f' with a voxel size of {self.radius / self.rho:.2f}' if self.subsample_support else '.'}\n"
                    f" -- number of processes: {self.n_procs}"
                )
            case "shot_bi_scale":
                return (
                    f"SHOT parameters:\n"
                    f" -- local reference frames computation radius: {self.radius}\n"
                    f" -- SHOT radius: {self.radius * self.phi}\n"
                    f" -- minimum neighborhood size: {self.min_neighborhood_size}\n"
                    f" -- the descriptors will{' not' if not self.normalize else ''} be normalized.\n"
                    f" -- the support will{' not' if not self.subsample_support else ''} be subsampled"
                    f"{f' with a voxel size of {self.radius / self.rho:.2f}' if self.subsample_support else '.'}\n"
                    f" -- number of processes: {self.n_procs}"
                )
            case "shot_multi_scale":
                shot_radii = ", ".join(
                    f"{self.radius * self.phi ** scale:.2f}"
                    for scale in range(self.n_scales)
                )
                subsampling_factors = ", ".join(
                    f"{self.radius * self.phi ** scale / self.rho:.2f}"
                    for scale in range(self.n_scales)
                )
                return (
                    f"SHOT parameters:\n"
                    f" -- SHOT radii: {shot_radii}\n"
                    f" -- minimum neighborhood size: {self.min_neighborhood_size}\n"
                    f" -- the descriptors will{' not' if not self.normalize else ''} be normalized.\n"
                    f" -- the local reference frames will{' not' if not self.share_local_rfs else ''} be shared.\n"
                    f" -- the support will{' not' if not self.subsample_support else ''} be subsampled"
                    f"{f' with voxel sizes of {subsampling_factors}' if self.subsample_support else '.'}\n"
                    f" -- number of processes: {self.n_procs}"
                )
            case "fpfh":
                return (
                    f"FPFH parameters:\n"
                    f" -- radius: {self.radius}\n"
                    f" -- number of bins: {self.fpfh_n_bins}"
                )

    def minimal_config(self) -> dict[str, bool | int]:
        """
        Retrieves a subset of the dataclass as a dict with the parameters effectively used by a ShotMultiprocessor as
        attributes.

        Returns:
            A dict containing the parameters used by an instance of ShotMultiprocessor (the keys of this dict match the
            attribute names).
        """
        return {
            param: value
            for param, value in asdict(self).items()
            if param
            in [
                "normalize",
                "share_local_rfs",
                "min_neighborhood_size",
                "n_procs",
                "disable_progress_bar",
                "verbose",
            ]
        }


@dataclass
class MatchingConfig(Config):
    """Parameters of the matching method."""

    matching_algorithm: Literal["simple", "double", "threshold"] = "simple"
    reject_threshold: float = 0.8
    threshold_multiplier: float = 10

    def help_message(self) -> str:
        return (
            f"Matching parameters:\n"
            f" -- matching strategy: {self.matching_algorithm}\n"
            f" -- rejection threshold (double): {self.reject_threshold}\n"
            f" -- threshold multiplier (threshold): {self.threshold_multiplier}"
        )


@dataclass
class RansacConfig(Config):
    """Parameters of the RANSAC method."""

    n_draws: int = 10000
    draw_size: int = 4
    max_inliers_distance: float = 1.0

    def help_message(self) -> str:
        return (
            f"RANSAC parameters:\n"
            f" -- number of draws: {self.n_draws}\n"
            f" -- draw size: {self.draw_size}\n"
            f" -- maximum inlier distance: {self.max_inliers_distance}"
        )


@dataclass
class IcpConfig(Config):
    """Parameters of the ICP."""

    icp_type: Literal["point_to_point", "point_to_plane"] = "point_to_plane"
    d_max: float = 0.5
    voxel_size: float = 0.2
    max_iter: int = 50
    rms_threshold: float = 1e-3

    def help_message(self) -> str:
        return (
            f"ICP parameters:\n"
            f" -- ICP type: {self.icp_type}\n"
            f" -- maximum number of iterations: {self.max_iter}\n"
            f" -- distance to map threshold: {self.rms_threshold}\n"
            f" -- d_max: {self.d_max}\n"
            f" -- subsampling voxel size: {self.voxel_size}"
        )


@dataclass
class RegistrationEvaluationConfig(Config):
    """
    Parameters used to detect the output of a registration as a correct or an incorrect one.
    The inliers are either defined as a ratio or a sheer number of inliers.
    """

    overlap_threshold: float = 0.6
    distance_to_map_threshold: float = 0.1
    inliers_threshold: float = 0.5

    def help_message(self) -> str:
        return (
            f"Registration evaluation parameters:\n"
            f" -- overlap > {self.overlap_threshold * 100:.0f}%\n"
            f" -- distance to map < {self.distance_to_map_threshold * 100:.0f} cm\n"
            f" -- inliers > {self.inliers_threshold:.2f}"
        )

    def eval_registration(
        self, *, overlap: float, distance_to_map: float, inliers: float | int
    ) -> bool:
        """
        Evaluates a registration to decide whether it has converged to an acceptable solution or not.
        """
        return (
            overlap > self.overlap_threshold
            and distance_to_map < self.distance_to_map_threshold
            and inliers > self.inliers_threshold
        )


class PipelineConfig(TypedDict):
    keypoint_selection: KeypointSelectionConfig
    descriptor: DescriptorConfig
    matching: MatchingConfig
    ransac: RansacConfig
    icp: IcpConfig
    registration_evaluation: RegistrationEvaluationConfig


def load_config_from_yaml(
    config_file_path: str, command_line_args: dict[str, Any]
) -> PipelineConfig:
    """
    Loads a YAML config file and overrides its values with the non-null values found in override_values.
    """

    def get_values(
        default_values: dict[str, Any], override_values: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Overrides entries from a dictionary when values are non-null.
        """
        return {
            **default_values,
            **{
                k: v
                for k, v in override_values.items()
                if k in default_values and v is not None
            },
        }

    with open(config_file_path) as f:
        config = yaml.safe_load(f.read())["registration"]

    return {
        "keypoint_selection": KeypointSelectionConfig(
            **get_values(config["keypoint_selection"], command_line_args),
        ),
        "descriptor": DescriptorConfig(
            **get_values(config["descriptor"], command_line_args),
        ),
        "matching": MatchingConfig(
            **get_values(config["matching"], command_line_args),
        ),
        "ransac": RansacConfig(
            **get_values(config["ransac"], command_line_args),
        ),
        "icp": IcpConfig(
            **get_values(config["icp"], command_line_args),
        ),
        "registration_evaluation": RegistrationEvaluationConfig(
            **get_values(config["registration_evaluation"], command_line_args),
        ),
    }
