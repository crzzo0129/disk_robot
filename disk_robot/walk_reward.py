from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from disk_robot.walk_config import WalkTaskConfig


REWARD_TERM_NAMES = (
    "velocity_xy",
    "yaw_rate",
    "stand",
    "lin_vel_z",
    "ang_vel_xy",
    "upright",
    "joint_vel",
    "disk_contact",
    "action",
    "action_delta",
    "foot_slip",
    "termination",
)


@dataclass(frozen=True)
class WalkRewardInputs:
    velocity_x: float
    velocity_y: float
    yaw_rate: float
    command_x: float
    command_y: float
    command_yaw: float
    vertical_velocity: float
    roll_pitch_rate_mean_square: float
    joint_velocity_mean_square: float
    upright: float
    disk_contact_count: int
    action_mean_square: float
    action_delta_mean_square: float
    foot_slip_mean_square: float = 0.0
    failed: bool = False


@dataclass(frozen=True)
class WalkReward:
    total: float
    terms: dict[str, float]


def reward_terms(xp, config: WalkTaskConfig, inputs: WalkRewardInputs | dict):
    get = inputs.get if isinstance(inputs, dict) else lambda name: getattr(inputs, name)
    vx_error = get("velocity_x") - get("command_x")
    vy_error = get("velocity_y") - get("command_y")
    yaw_error = get("yaw_rate") - get("command_yaw")
    command_norm_sq = get("command_x") ** 2 + get("command_y") ** 2 + get("command_yaw") ** 2
    motion_norm_sq = get("velocity_x") ** 2 + get("velocity_y") ** 2 + get("yaw_rate") ** 2
    zero_command = xp.where(command_norm_sq < 1e-6, 1.0, 0.0)
    return {
        "velocity_xy": config.reward_velocity_xy
        * xp.exp(-(vx_error * vx_error + vy_error * vy_error) / config.velocity_tracking_sigma),
        "yaw_rate": config.reward_yaw_rate * xp.exp(-(yaw_error * yaw_error) / config.yaw_tracking_sigma),
        "stand": config.reward_stand * zero_command * xp.exp(-motion_norm_sq / config.stand_tracking_sigma),
        "lin_vel_z": -config.penalty_lin_vel_z * get("vertical_velocity") ** 2,
        "ang_vel_xy": -config.penalty_ang_vel_xy * get("roll_pitch_rate_mean_square"),
        "upright": -config.penalty_upright * xp.maximum(0.0, 1.0 - get("upright")) ** 2,
        "joint_vel": -config.penalty_joint_vel * get("joint_velocity_mean_square"),
        "disk_contact": -config.penalty_disk_contact * get("disk_contact_count"),
        "action": -config.penalty_action * get("action_mean_square"),
        "action_delta": -config.penalty_action_delta * get("action_delta_mean_square"),
        "foot_slip": -config.penalty_foot_slip * get("foot_slip_mean_square"),
        "termination": -config.penalty_termination * get("failed"),
    }


def compute_walk_reward(*, config: WalkTaskConfig, inputs: WalkRewardInputs) -> WalkReward:
    terms = reward_terms(np, config, inputs)
    return WalkReward(total=float(sum(terms.values())), terms={name: float(value) for name, value in terms.items()})
