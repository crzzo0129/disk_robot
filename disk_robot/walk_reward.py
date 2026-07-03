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
    torso_height: float,
    upright: float,
    disk_contact_count: int,
    foot_contact_count: int,
    action_mean_square: float,
    action_delta_mean_square: float,
) -> WalkReward:
    velocity_error = forward_velocity - config.command_velocity
    velocity = config.reward_velocity * (1.0 - velocity_error * velocity_error)
    lateral = -config.reward_lateral * lateral_velocity * lateral_velocity
    upright_term = -config.reward_upright * max(0.0, 1.0 - upright)
    height = -config.reward_height * max(0.0, config.min_torso_height - torso_height)
    contact = config.reward_contact * min(float(foot_contact_count), 4.0) / 4.0
    disk_contact = -config.penalty_disk_contact * float(disk_contact_count)
    action = -config.penalty_action * action_mean_square
    action_delta = -config.penalty_action_delta * action_delta_mean_square
    terms = {
        "velocity": velocity,
        "lateral": lateral,
        "upright": upright_term,
        "height": height,
        "foot_contact": contact,
        "disk_contact": disk_contact,
        "action": action,
        "action_delta": action_delta,
        "alive": 0.1,
    }
    return WalkReward(total=float(sum(terms.values())), terms={k: float(v) for k, v in terms.items()})

