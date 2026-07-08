from __future__ import annotations

from dataclasses import dataclass

from disk_robot.walk_config import WalkTaskConfig


REWARD_TERM_NAMES = (
    "velocity",
    "forward",
    "lateral",
    "yaw",
    "heading",
    "lin_vel_z",
    "ang_vel_xy",
    "joint_vel",
    "upright",
    "upright_positive",
    "height",
    "height_target",
    "contact",
    "contact_schedule",
    "feet_air_time",
    "disk_contact",
    "action",
    "action_delta",
    "foot_slip",
    "termination",
    "alive",
)


@dataclass(frozen=True)
class WalkRewardInputs:
    forward_velocity: float
    lateral_velocity: float
    yaw_rate: float
    heading_error: float
    vertical_velocity: float
    roll_pitch_rate_mean_square: float
    joint_velocity_mean_square: float
    torso_height: float
    upright: float
    disk_contact_count: int
    foot_contact_count: int
    contact_schedule_match: float
    action_mean_square: float
    action_delta_mean_square: float
    foot_slip_mean_square: float = 0.0
    failed: bool = False


@dataclass(frozen=True)
class WalkReward:
    total: float
    terms: dict[str, float]


def compute_walk_reward(*, config: WalkTaskConfig, inputs: WalkRewardInputs) -> WalkReward:
    velocity_error = inputs.forward_velocity - config.command_velocity
    height_error = inputs.torso_height - config.target_torso_height
    terms = {
        "velocity": config.reward_velocity
        * pow(2.718281828459045, -(velocity_error * velocity_error) / config.tracking_sigma),
        "forward": config.reward_forward * inputs.forward_velocity,
        "lateral": -config.reward_lateral * inputs.lateral_velocity * inputs.lateral_velocity,
        "yaw": -config.penalty_yaw_rate * inputs.yaw_rate * inputs.yaw_rate,
        "heading": -config.penalty_heading_error * inputs.heading_error * inputs.heading_error,
        "lin_vel_z": -config.penalty_lin_vel_z * inputs.vertical_velocity * inputs.vertical_velocity,
        "ang_vel_xy": -config.penalty_ang_vel_xy * inputs.roll_pitch_rate_mean_square,
        "joint_vel": -config.penalty_joint_vel * inputs.joint_velocity_mean_square,
        "upright": -config.reward_upright * max(0.0, 1.0 - inputs.upright),
        "upright_positive": config.reward_upright_positive * max(0.0, inputs.upright),
        "height": -config.reward_height * max(0.0, config.min_torso_height - inputs.torso_height),
        "height_target": config.reward_height_target
        * pow(2.718281828459045, -(height_error * height_error) / config.height_tracking_sigma),
        "contact": config.reward_contact * min(float(inputs.foot_contact_count), 4.0) / 4.0,
        "contact_schedule": config.reward_contact_schedule * inputs.contact_schedule_match,
        "feet_air_time": 0.0,
        "disk_contact": -config.penalty_disk_contact * float(inputs.disk_contact_count),
        "action": -config.penalty_action * inputs.action_mean_square,
        "action_delta": -config.penalty_action_delta * inputs.action_delta_mean_square,
        "foot_slip": -config.penalty_foot_slip * inputs.foot_slip_mean_square,
        "termination": -config.penalty_termination * float(inputs.failed),
        "alive": config.reward_alive,
    }
    return WalkReward(total=float(sum(terms.values())), terms={k: float(v) for k, v in terms.items()})

