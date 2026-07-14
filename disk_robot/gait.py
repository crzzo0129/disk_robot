from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


LEG_ORDER = ("front_r", "front_l", "back_r", "back_l")
LEG_SLICES = {leg: slice(index * 3, index * 3 + 3) for index, leg in enumerate(LEG_ORDER)}


@dataclass(frozen=True)
class GaitParams:
    frequency: float = 1.2
    hip_stance_amplitude: float = 0.20
    hip_swing_amplitude: float = 0.20
    knee_lift_amplitude: float = 0.12
    abd_amplitude: float = 0.0
    duty: float = 0.6
    mode: str = "trot"
    direction: float = 1.0
    front_knee_sign: float = -1.0
    hind_knee_sign: float = -1.0
    march_hip_compensation: float = 0.0


def leg_phase_offsets(mode: str) -> np.ndarray:
    if mode == "trot":
        values = (0.0, 0.5, 0.5, 0.0)
    elif mode == "pace":
        values = (0.0, 0.5, 0.0, 0.5)
    elif mode == "bound":
        values = (0.0, 0.0, 0.5, 0.5)
    elif mode == "march":
        values = (0.0, 0.5, 0.75, 0.25)
    else:
        values = (0.0, 0.25, 0.75, 0.5)
    return np.array(values, dtype=np.float64)


def phase_at_time(t: float, frequency: float) -> float:
    return float((t * frequency) % 1.0)


def phase_observation(phase: float) -> np.ndarray:
    angle = 2.0 * math.pi * phase
    return np.array([math.sin(angle), math.cos(angle)], dtype=np.float64)


def desired_contacts_at_time(t: float, params: GaitParams) -> np.ndarray:
    duty = min(max(params.duty, 1e-6), 1.0 - 1e-6)
    phases = (t * params.frequency + leg_phase_offsets(params.mode)) % 1.0
    return (phases < duty).astype(np.float64)


def _phase_components(phase: float, duty: float) -> tuple[float, float, float]:
    phase = phase % 1.0
    duty = min(max(duty, 1e-6), 1.0 - 1e-6)
    if phase < duty:
        stance_u = phase / duty
        return math.cos(math.pi * stance_u), 0.0, 0.0
    swing_u = (phase - duty) / (1.0 - duty)
    return 0.0, math.sin(math.pi * swing_u), -math.cos(math.pi * swing_u)


def make_open_loop_targets(neutral: np.ndarray, t: float, params: GaitParams) -> np.ndarray:
    targets = np.array(neutral, dtype=np.float64, copy=True)
    offsets = leg_phase_offsets(params.mode)
    for leg_index, leg in enumerate(LEG_ORDER):
        stance_push, swing_lift, swing_return = _phase_components(t * params.frequency + offsets[leg_index], params.duty)
        leg_slice = LEG_SLICES[leg]
        side_sign = 1.0 if leg.endswith("_l") else -1.0
        knee_sign = params.front_knee_sign if leg.startswith("front") else params.hind_knee_sign
        targets[leg_slice.start + 0] += side_sign * params.abd_amplitude * stance_push
        if params.mode == "march":
            targets[leg_slice.start + 1] += params.march_hip_compensation * swing_lift
        else:
            hip_offset = params.hip_stance_amplitude * stance_push + params.hip_swing_amplitude * swing_return
            targets[leg_slice.start + 1] += params.direction * hip_offset
        targets[leg_slice.start + 2] += knee_sign * params.knee_lift_amplitude * swing_lift
    return targets


def make_open_loop_targets_jax(jp, neutral, t, params: GaitParams, phase_offsets):
    duty = jp.clip(jp.array(params.duty), 1e-6, 1.0 - 1e-6)
    phases = jp.mod(t * params.frequency + phase_offsets, 1.0)
    stance = phases < duty
    stance_u = phases / duty
    swing_u = (phases - duty) / (1.0 - duty)
    stance_push = jp.where(stance, jp.cos(jp.pi * stance_u), 0.0)
    swing_lift = jp.where(stance, 0.0, jp.sin(jp.pi * swing_u))
    swing_return = jp.where(stance, 0.0, -jp.cos(jp.pi * swing_u))

    side_sign = jp.array([1.0, -1.0, 1.0, -1.0])
    knee_sign = jp.array([
        params.front_knee_sign,
        params.front_knee_sign,
        params.hind_knee_sign,
        params.hind_knee_sign,
    ])
    abd_offsets = side_sign * params.abd_amplitude * stance_push
    if params.mode == "march":
        hip_offsets = params.march_hip_compensation * swing_lift
    else:
        hip_offsets = params.direction * (
            params.hip_stance_amplitude * stance_push + params.hip_swing_amplitude * swing_return
        )
    knee_offsets = knee_sign * params.knee_lift_amplitude * swing_lift

    offsets = jp.zeros_like(neutral)
    offsets = offsets.at[0::3].set(abd_offsets)
    offsets = offsets.at[1::3].set(hip_offsets)
    offsets = offsets.at[2::3].set(knee_offsets)
    return neutral + offsets


def desired_contacts_at_time_jax(jp, t, params: GaitParams, phase_offsets):
    duty = jp.clip(jp.array(params.duty), 1e-6, 1.0 - 1e-6)
    phases = jp.mod(t * params.frequency + phase_offsets, 1.0)
    return (phases < duty).astype(jp.float32)


def phase_observation_jax(jp, phase):
    angle = 2.0 * jp.pi * phase
    return jp.array([jp.sin(angle), jp.cos(angle)])
