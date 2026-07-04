from __future__ import annotations

from dataclasses import dataclass


JOINT_NAMES = (
    "fl_hip_abd",
    "fl_hip_flex",
    "fl_knee",
    "fr_hip_abd",
    "fr_hip_flex",
    "fr_knee",
    "hl_hip_abd",
    "hl_hip_flex",
    "hl_knee",
    "hr_hip_abd",
    "hr_hip_flex",
    "hr_knee",
)
ACTUATOR_NAMES = tuple(f"{name}_act" for name in JOINT_NAMES)
FOOT_GEOMS = ("fl_foot", "fr_foot", "hl_foot", "hr_foot")


@dataclass(frozen=True)
class WalkTaskConfig:
    command_velocity: float = 0.45
    action_scale: float = 0.3
    action_repeat: int = 5
    max_episode_steps: int = 250
    reset_joint_noise: float = 0.03
    reset_height_noise: float = 0.002
    min_torso_height: float = 0.20
    terminate_upright: float = 0.45
    reward_velocity: float = 2.0
    reward_forward: float = 0.5
    tracking_sigma: float = 0.25
    reward_lateral: float = 0.4
    reward_upright: float = 1.0
    reward_height: float = 0.4
    reward_alive: float = 1.0
    reward_upright_positive: float = 1.0
    reward_height_target: float = 1.0
    target_torso_height: float = 0.406
    height_tracking_sigma: float = 0.0025
    reward_contact: float = 0.0
    reward_feet_air_time: float = 0.3
    min_feet_air_time: float = 0.08
    penalty_disk_contact: float = 2.0
    penalty_termination: float = 5.0
    penalty_lin_vel_z: float = 0.5
    penalty_ang_vel_xy: float = 0.05
    penalty_joint_vel: float = 0.001
    penalty_action: float = 0.03
    penalty_action_delta: float = 0.7

    @property
    def action_size(self) -> int:
        return len(JOINT_NAMES)

    @property
    def observation_size(self) -> int:
        # quat, linear velocity, angular velocity, qpos, qvel, previous action,
        # foot contacts, and commanded forward velocity.
        return 4 + 3 + 3 + self.action_size + self.action_size + self.action_size + 4 + 1

