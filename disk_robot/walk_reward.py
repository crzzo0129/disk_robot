from __future__ import annotations

from dataclasses import dataclass

from disk_robot.walk_config import WalkTaskConfig


@dataclass(frozen=True)
class WalkReward:
    total: float
    terms: dict[str, float]


def compute_walk_reward(
    *,
    config: WalkTaskConfig,
    forward_velocity: float,
    lateral_velocity: float,
    vertical_velocity: float,
    angular_velocity_xy_mean_square: float,
    joint_velocity_mean_square: float,
    torso_height: float,
    upright: float,
    disk_contact_count: int,
    foot_contact_count: int,
    action_mean_square: float,
    action_delta_mean_square: float,
    done: bool = False,
) -> WalkReward:
    velocity_error = forward_velocity - config.command_velocity
    velocity = config.reward_velocity * pow(2.718281828459045, -(velocity_error * velocity_error) / config.tracking_sigma)
    forward = config.reward_forward * forward_velocity
    lateral = -config.reward_lateral * lateral_velocity * lateral_velocity
    lin_vel_z = -config.penalty_lin_vel_z * vertical_velocity * vertical_velocity
    ang_vel_xy = -config.penalty_ang_vel_xy * angular_velocity_xy_mean_square
    joint_vel = -config.penalty_joint_vel * joint_velocity_mean_square
    upright_term = -config.reward_upright * max(0.0, 1.0 - upright)
    upright_positive = config.reward_upright_positive * max(0.0, upright)
    height = -config.reward_height * max(0.0, config.min_torso_height - torso_height)
    height_error = torso_height - config.target_torso_height
    height_target = config.reward_height_target * pow(
        2.718281828459045,
        -(height_error * height_error) / config.height_tracking_sigma,
    )
    contact = config.reward_contact * min(float(foot_contact_count), 4.0) / 4.0
    disk_contact = -config.penalty_disk_contact * float(disk_contact_count)
    action = -config.penalty_action * action_mean_square
    action_delta = -config.penalty_action_delta * action_delta_mean_square
    termination = -config.penalty_termination * float(done)
    alive = config.reward_alive
    terms = {
        "velocity": velocity,
        "forward": forward,
        "lateral": lateral,
        "lin_vel_z": lin_vel_z,
        "ang_vel_xy": ang_vel_xy,
        "joint_vel": joint_vel,
        "upright": upright_term,
        "upright_positive": upright_positive,
        "height": height,
        "height_target": height_target,
        "foot_contact": contact,
        "disk_contact": disk_contact,
        "action": action,
        "action_delta": action_delta,
        "termination": termination,
        "alive": alive,
    }
    return WalkReward(total=float(sum(terms.values())), terms={k: float(v) for k, v in terms.items()})

