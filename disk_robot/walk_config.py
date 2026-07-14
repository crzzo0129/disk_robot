from __future__ import annotations

from dataclasses import dataclass, replace

from disk_robot.model_contract import ACTUATOR_NAMES, FOOT_BODY_NAMES, JOINT_NAMES


ACTION_SCALE = (
    0.35, 0.60, 0.85,
    0.35, 0.60, 0.85,
    0.35, 0.60, 0.85,
    0.35, 0.60, 0.85,
)


@dataclass(frozen=True)
class WalkTaskConfig:
    action_scale: tuple[float, ...] = ACTION_SCALE
    action_repeat: int = 5
    max_episode_steps: int = 500
    reset_joint_noise: float = 0.03
    reset_height_noise: float = 0.002
    reset_foot_clearance: float = 0.001
    min_torso_height: float = 0.18
    terminate_upright: float = 0.55
    observation_history: int = 4

    command_vx_min: float = 0.15
    command_vx_max: float = 0.30
    command_vy_min: float = 0.0
    command_vy_max: float = 0.0
    command_yaw_min: float = 0.0
    command_yaw_max: float = 0.0
    command_zero_probability: float = 0.0
    command_resample_steps: int = 100

    reward_velocity_xy: float = 1.0
    reward_yaw_rate: float = 0.25
    reward_directional_progress: float = 1.0
    reward_stand: float = 0.1
    velocity_tracking_sigma: float = 0.015
    yaw_tracking_sigma: float = 0.25
    stand_tracking_sigma: float = 0.04
    penalty_lin_vel_z: float = 0.15
    penalty_ang_vel_xy: float = 0.05
    penalty_upright: float = 0.2
    penalty_joint_vel: float = 0.0002
    penalty_action: float = 0.005
    penalty_action_delta: float = 0.02
    penalty_foot_slip: float = 0.05
    penalty_disk_contact: float = 0.5
    penalty_termination: float = 5.0

    @property
    def action_size(self) -> int:
        return len(JOINT_NAMES)

    @property
    def observation_frame_size(self) -> int:
        # body gyro, projected gravity, estimated body velocity, q-q_stand,
        # qvel, previous action, command
        return 3 + 3 + 3 + self.action_size + self.action_size + self.action_size + 3

    @property
    def observation_size(self) -> int:
        return self.observation_frame_size * self.observation_history


def command_profile(name: str, config: WalkTaskConfig | None = None) -> WalkTaskConfig:
    cfg = config or WalkTaskConfig()
    if name == "forward":
        return replace(
            cfg,
            command_vx_min=0.15,
            command_vx_max=0.30,
            command_vy_min=0.0,
            command_vy_max=0.0,
            command_yaw_min=0.0,
            command_yaw_max=0.0,
            command_zero_probability=0.0,
            reward_directional_progress=1.0,
        )
    if name == "omni":
        return replace(
            cfg,
            command_vx_min=-0.20,
            command_vx_max=0.35,
            command_vy_min=-0.18,
            command_vy_max=0.18,
            command_yaw_min=-1.0,
            command_yaw_max=1.0,
            command_zero_probability=0.15,
            reward_directional_progress=0.3,
        )
    if name == "full":
        return replace(
            cfg,
            command_vx_min=-0.30,
            command_vx_max=0.50,
            command_vy_min=-0.30,
            command_vy_max=0.30,
            command_yaw_min=-1.5,
            command_yaw_max=1.5,
            command_zero_probability=0.15,
            reward_directional_progress=0.2,
        )
    raise ValueError(f"Unknown command profile: {name}")


__all__ = ["ACTUATOR_NAMES", "FOOT_BODY_NAMES", "JOINT_NAMES", "WalkTaskConfig", "command_profile"]
