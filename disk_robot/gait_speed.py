from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ForwardGaitPlan:
    target_speed: float
    frequency: float
    stride_length: float
    motion_scale: float


# Measured on pupper_v3_disk_structure_candidate.xml with trot, duty=0.72,
# Kp=10, Kd=0.4 and a 3 Nm torque limit.  Frequency is held at 1.2 Hz
# because higher cadence increased tracking error without increasing speed.
_SPEED_STRIDE_CALIBRATION = (
    (0.0000, 0.0000),
    (0.0017, 0.0100),
    (0.0127, 0.0200),
    (0.0239, 0.0300),
    (0.0353, 0.0400),
    (0.0469, 0.0500),
    (0.0585, 0.0600),
    (0.0725, 0.0700),
    (0.0903, 0.0800),
    (0.1000, 0.0900),
)

CALIBRATED_FREQUENCY = 1.2
MAX_CALIBRATED_FORWARD_SPEED = _SPEED_STRIDE_CALIBRATION[-1][0]


def plan_forward_gait(target_speed: float) -> ForwardGaitPlan:
    """Maps a forward speed command to the candidate model's calibrated IK gait."""

    if not math.isfinite(target_speed):
        raise ValueError("target speed must be finite")
    if target_speed < 0.0:
        raise ValueError("target speed must be nonnegative")
    if target_speed > MAX_CALIBRATED_FORWARD_SPEED + 1e-9:
        raise ValueError(
            f"target speed exceeds calibrated limit {MAX_CALIBRATED_FORWARD_SPEED:.4f} m/s"
        )

    for (speed0, stride0), (speed1, stride1) in zip(
        _SPEED_STRIDE_CALIBRATION,
        _SPEED_STRIDE_CALIBRATION[1:],
    ):
        if target_speed <= speed1:
            alpha = (target_speed - speed0) / (speed1 - speed0)
            stride = stride0 + alpha * (stride1 - stride0)
            motion_scale = min(1.0, target_speed / 0.02)
            return ForwardGaitPlan(target_speed, CALIBRATED_FREQUENCY, stride, motion_scale)

    return ForwardGaitPlan(
        target_speed,
        CALIBRATED_FREQUENCY,
        _SPEED_STRIDE_CALIBRATION[-1][1],
        1.0,
    )


__all__ = [
    "CALIBRATED_FREQUENCY",
    "ForwardGaitPlan",
    "MAX_CALIBRATED_FORWARD_SPEED",
    "plan_forward_gait",
]
