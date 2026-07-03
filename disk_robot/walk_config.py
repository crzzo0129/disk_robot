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
    action_scale: float = 0.08
    action_repeat: int = 10
    max_episode_steps: int = 250
    reset_joint_noise: float = 0.03
    reset_height_noise: float = 0.01
    min_torso_height: float = 0.20
    terminate_upright: float = 0.45
    reward_velocity: float = 2.0
    reward_lateral: float = 0.4
    reward_upright: float = 0.8
    reward_height: float = 0.4
    reward_contact: float = 0.2
    penalty_disk_contact: float = 2.0
    penalty_action: float = 0.03
    penalty_action_delta: float = 0.05

    @property
    def action_size(self) -> int:
        return len(JOINT_NAMES)

    @property
    def observation_size(self) -> int:
        # quat, linear velocity, angular velocity, qpos, qvel, previous action,
        # foot contacts, and commanded forward velocity.
        return 4 + 3 + 3 + self.action_size + self.action_size + self.action_size + 4 + 1

