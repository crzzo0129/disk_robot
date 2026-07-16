from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhaseClockConfig:
    frequency_hz: float = 1.2
    startup_blend_steps: int = 25
    command_deadzone: float = 0.01


@dataclass(frozen=True)
class PhaseClockState:
    phase: float = 0.0
    gait_blend: float = 0.0


def phase_clock_observation(state: PhaseClockState) -> np.ndarray:
    angle = 2.0 * math.pi * (state.phase % 1.0)
    return np.asarray(
        (math.sin(angle), math.cos(angle), state.gait_blend),
        dtype=np.float32,
    )


def update_phase_clock(
    state: PhaseClockState,
    command,
    dt: float,
    config: PhaseClockConfig,
) -> PhaseClockState:
    command = np.asarray(command, dtype=np.float64)
    if command.shape != (3,):
        raise ValueError("phase clock command must be [vx, vy, yaw_rate]")
    if dt <= 0.0:
        raise ValueError("phase clock dt must be positive")
    if config.frequency_hz < 0.0:
        raise ValueError("phase clock frequency must be non-negative")
    if config.startup_blend_steps < 1:
        raise ValueError("startup_blend_steps must be at least 1")

    command_magnitude = float(np.linalg.norm(command[:2]) + abs(command[2]))
    active = command_magnitude > config.command_deadzone
    phase = (
        state.phase + (config.frequency_hz * dt if active else 0.0)
    ) % 1.0
    blend_delta = 1.0 / config.startup_blend_steps
    gait_blend = np.clip(
        state.gait_blend + (blend_delta if active else -blend_delta),
        0.0,
        1.0,
    )
    return PhaseClockState(phase=float(phase), gait_blend=float(gait_blend))


def append_phase_clock_observation(sensor_history, state: PhaseClockState) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(sensor_history, dtype=np.float32),
            phase_clock_observation(state),
        )
    )


__all__ = [
    "PhaseClockConfig",
    "PhaseClockState",
    "append_phase_clock_observation",
    "phase_clock_observation",
    "update_phase_clock",
]
