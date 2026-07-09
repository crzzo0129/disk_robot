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
    command_velocity: float = 0.0
    action_scale: float = 0.5
    action_repeat: int = 5
    max_episode_steps: int = 250
    reset_joint_noise: float = 0.03
    reset_height_noise: float = 0.002
    min_torso_height: float = 0.29
    terminate_upright: float = 0.75
    reward_velocity: float = 1.0
    reward_forward: float = 1.0
    tracking_sigma: float = 0.08
    reward_lateral: float = 1.0
    penalty_yaw_rate: float = 0.5
    penalty_heading_error: float = 0.5
    reward_upright: float = 0.0
    reward_height: float = 0.4
    reward_alive: float = 0.5
    reward_upright_positive: float = 0.5
    reward_height_target: float = 0.0
    target_torso_height: float = 0.406
    height_tracking_sigma: float = 0.0025
    reward_contact: float = 0.6
    reward_contact_schedule: float = 0.2
    reward_feet_air_time: float = 0.3
    min_feet_air_time: float = 0.2
    penalty_disk_contact: float = 2.0
    penalty_termination: float = 100.0
    penalty_lin_vel_z: float = 1.0
    penalty_ang_vel_xy: float = 0.5
    penalty_joint_vel: float = 0.001
    penalty_action: float = 0.05
    penalty_action_delta: float = 0.5
    penalty_foot_slip: float = 0.1
    observation_history: int = 20
    use_open_loop_gait: bool = True
    residual_action_scale: float = 0.05
    gait_frequency: float = 1.0
    gait_hip_stance_amplitude: float = 0.10
    gait_hip_swing_amplitude: float = 0.10
    gait_knee_lift_amplitude: float = 0.22
    gait_abd_amplitude: float = 0.0
    gait_duty: float = 0.55
    gait_mode: str = "trot"
    gait_direction: float = -1.0
    gait_time_offset: float = 0.5
    gait_front_knee_sign: float = -1.0
    gait_hind_knee_sign: float = -1.0
    gait_march_hip_compensation: float = 0.0

    @property
    def action_size(self) -> int:
        return len(JOINT_NAMES)

    @property
    def observation_frame_size(self) -> int:
        # quat, linear velocity, angular velocity, torso height, qpos, qvel,
        # previous action, actual foot contacts, commanded forward velocity, gait phase sin/cos,
        # heading sin/cos relative to the world x-axis, and desired foot contacts.
        return 4 + 3 + 3 + 1 + self.action_size + self.action_size + self.action_size + 4 + 1 + 2 + 2 + 4

    @property
    def observation_size(self) -> int:
        return self.observation_frame_size * self.observation_history

