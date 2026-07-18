from __future__ import annotations

from dataclasses import dataclass

from disk_robot.walk_config import ACTION_SCALE


RESIDUAL_SCALE = (
    0.10, 0.16, 0.20,
    0.10, 0.16, 0.20,
    0.10, 0.16, 0.20,
    0.10, 0.16, 0.20,
)


@dataclass(frozen=True)
class ForwardTeacherStudentConfig:
    action_repeat: int = 5
    max_episode_steps: int = 500
    reset_joint_noise: float = 0.015
    reset_height_noise: float = 0.002
    reset_foot_clearance: float = 0.001
    startup_blend_steps: int = 25
    min_torso_height: float = 0.16
    terminate_upright: float = 0.65
    observation_history: int = 4

    command_vx: float = 0.08
    student_action_scale: tuple[float, ...] = ACTION_SCALE
    residual_scale: tuple[float, ...] = RESIDUAL_SCALE
    residual_filter_alpha: float = 0.15
    student_phase_conditioned: bool = False
    student_previous_action_input: bool = True
    student_phase_frequency: float = 1.2
    student_command_deadzone: float = 0.01
    # Read-only diagnostics can pin all policy roles to the same controller phase.
    # None preserves the training behavior (random Teacher phase, zero Student phase).
    fixed_reset_phase: float | None = None

    disturbance_enabled: bool = False
    push_step_min: int = 100
    push_step_max: int = 350
    push_velocity_x: float = 0.50
    push_velocity_y: float = 0.40
    motor_strength_min: float = 0.85
    motor_strength_max: float = 1.15
    control_delay_probability: float = 0.50
    disturbance_reset_joint_noise: float = 0.030
    disturbance_reset_height_noise: float = 0.005
    recovery_window_steps: int = 100
    recovery_velocity_ema_alpha: float = 0.10
    recovery_forward_tolerance: float = 0.04
    recovery_lateral_tolerance: float = 0.04
    recovery_required_steps: int = 4

    actuator_kp: float = 10.0
    actuator_kd: float = 0.4
    torque_limit: float = 3.0

    reward_velocity: float = 2.0
    velocity_sigma: float = 0.01
    reward_progress: float = 1.0
    reward_yaw: float = 0.4
    yaw_sigma: float = 0.10
    reward_alive: float = 0.2
    penalty_vertical_velocity: float = 0.2
    penalty_roll_pitch_rate: float = 0.20
    penalty_orientation: float = 0.8
    penalty_joint_velocity: float = 0.0005
    penalty_foot_slip: float = 0.05
    penalty_disk_contact: float = 0.8
    penalty_residual: float = 0.20
    penalty_residual_rate: float = 0.05
    penalty_contact_mismatch: float = 0.02
    penalty_termination: float = 5.0

    @property
    def action_size(self) -> int:
        return 12

    @property
    def student_frame_size(self) -> int:
        return 3 + 3 + 3 + 12 + 12 + 12 + 3

    @property
    def student_observation_size(self) -> int:
        return self.student_frame_size * self.observation_history

    @property
    def student_policy_frame_size(self) -> int:
        return self.student_frame_size if self.student_previous_action_input else self.student_frame_size - 12

    @property
    def student_policy_sensor_history_size(self) -> int:
        return self.student_policy_frame_size * self.observation_history

    @property
    def student_internal_state_size(self) -> int:
        # Internal oscillator sin/cos and the controller-owned startup/stop blend.
        return 3 if self.student_phase_conditioned else 0

    @property
    def student_policy_observation_size(self) -> int:
        return self.student_policy_sensor_history_size + self.student_internal_state_size

    @property
    def privileged_size(self) -> int:
        # phase, blend, contacts, IK error, residual, push, motor strength, control delay
        return 2 + 1 + 4 + 4 + 12 + 12 + 2 + 1 + 1

    @property
    def teacher_observation_size(self) -> int:
        return self.student_observation_size + self.privileged_size


__all__ = ["ForwardTeacherStudentConfig", "RESIDUAL_SCALE"]
