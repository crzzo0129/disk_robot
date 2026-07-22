from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from disk_robot.ik_gait import FootSpaceIKGait, FootTrajectoryParams, phase_offsets
from disk_robot.model_contract import resolve_model_contract


@dataclass(frozen=True)
class IKReferenceSpec:
    samples: int = 256
    frequency: float = 0.8
    stride_length: float = 0.04
    step_height: float = 0.025
    duty: float = 0.72
    mode: str = "trot"

    def trajectory_params(self) -> FootTrajectoryParams:
        return FootTrajectoryParams(
            frequency=self.frequency,
            stride_length=self.stride_length,
            step_height=self.step_height,
            duty=self.duty,
            mode=self.mode,
        )


@dataclass(frozen=True)
class IKReferenceTable:
    joint_targets: np.ndarray
    desired_contacts: np.ndarray
    stand_q: np.ndarray
    spec: IKReferenceSpec


@dataclass(frozen=True)
class IKReferenceBank:
    command_vx: np.ndarray
    joint_targets: np.ndarray
    desired_contacts: np.ndarray
    stand_q: np.ndarray
    specs: tuple[IKReferenceSpec, ...]


def build_ik_reference(xml_path: str | Path, spec: IKReferenceSpec | None = None) -> IKReferenceTable:
    """Builds one periodic IK cycle around the XML stand keyframe."""

    import mujoco

    spec = spec or IKReferenceSpec()
    model = mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser().resolve()))
    return build_ik_reference_from_model(model, spec)


def build_ik_reference_from_model(model, spec: IKReferenceSpec | None = None) -> IKReferenceTable:
    """Builds an IK cycle from a compiled model, including in-memory structure variants."""

    spec = spec or IKReferenceSpec()
    if spec.samples < 8:
        raise ValueError("IK reference requires at least 8 samples")
    contract = resolve_model_contract(model)
    gait = FootSpaceIKGait(model, contract, spec.trajectory_params())

    # Run one warm-up cycle so the iterative IK branch is periodic and continuous.
    targets = []
    for sample in range(2 * spec.samples):
        phase = (sample % spec.samples) / spec.samples
        target = gait.targets(phase / spec.frequency)
        if sample >= spec.samples:
            targets.append(target)

    phases = np.arange(spec.samples, dtype=np.float64) / spec.samples
    leg_phases = (phases[:, None] + phase_offsets(spec.mode)[None, :]) % 1.0
    contacts = (leg_phases < spec.duty).astype(np.float32)
    table = np.asarray(targets, dtype=np.float32)
    if not np.all(np.isfinite(table)):
        raise ValueError("IK reference contains non-finite joint targets")
    return IKReferenceTable(
        joint_targets=table,
        desired_contacts=contacts,
        stand_q=contract.stand_q.astype(np.float32, copy=True),
        spec=spec,
    )


def build_ik_reference_bank(
    xml_path: str | Path,
    command_vx,
    specs,
) -> IKReferenceBank:
    """Builds aligned IK cycles for a sorted set of episode-fixed forward commands."""

    import mujoco

    commands = np.asarray(command_vx, dtype=np.float32)
    specs = tuple(specs)
    if commands.ndim != 1 or len(commands) != len(specs) or len(commands) < 2:
        raise ValueError("IK reference bank requires at least two command/spec pairs")
    if not np.all(np.isfinite(commands)) or np.any(np.diff(commands) <= 0.0):
        raise ValueError("IK reference bank commands must be finite and strictly increasing")
    model = mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser().resolve()))
    tables = tuple(build_ik_reference_from_model(model, spec) for spec in specs)
    first = tables[0]
    for table in tables[1:]:
        if table.joint_targets.shape != first.joint_targets.shape:
            raise ValueError("IK reference bank tables must have identical shapes")
        if not np.allclose(table.stand_q, first.stand_q):
            raise ValueError("IK reference bank tables must share the same stand keyframe")
        if (
            table.spec.frequency != first.spec.frequency
            or table.spec.duty != first.spec.duty
            or table.spec.mode != first.spec.mode
        ):
            raise ValueError("IK reference bank entries must share frequency, duty, and mode")
    return IKReferenceBank(
        command_vx=commands,
        joint_targets=np.stack([table.joint_targets for table in tables]),
        desired_contacts=np.stack([table.desired_contacts for table in tables]),
        stand_q=first.stand_q.copy(),
        specs=specs,
    )


def interpolate_reference_jax(jp, table, phase):
    """Periodically interpolates a lookup table for scalar or batched phases."""

    sample = jp.mod(phase, 1.0) * table.shape[0]
    lower = jp.floor(sample).astype(jp.int32)
    upper = jp.mod(lower + 1, table.shape[0])
    alpha = sample - jp.floor(sample)
    return table[lower] + alpha[..., None] * (table[upper] - table[lower])


def interpolate_reference_bank_jax(jp, commands, tables, command_vx, phase):
    """Interpolates first in gait phase, then across adjacent speed anchors."""

    phase_sample = jp.mod(phase, 1.0) * tables.shape[1]
    phase_lower = jp.floor(phase_sample).astype(jp.int32)
    phase_upper = jp.mod(phase_lower + 1, tables.shape[1])
    phase_alpha = phase_sample - jp.floor(phase_sample)
    phase_targets = tables[:, phase_lower] + phase_alpha * (
        tables[:, phase_upper] - tables[:, phase_lower]
    )
    speed_upper = jp.clip(jp.searchsorted(commands, command_vx, side="right"), 1, commands.shape[0] - 1)
    speed_lower = speed_upper - 1
    denominator = jp.maximum(commands[speed_upper] - commands[speed_lower], 1e-6)
    speed_alpha = jp.clip(
        (command_vx - commands[speed_lower]) / denominator, 0.0, 1.0
    )
    return phase_targets[speed_lower] + speed_alpha * (
        phase_targets[speed_upper] - phase_targets[speed_lower]
    )


__all__ = [
    "IKReferenceSpec",
    "IKReferenceBank",
    "IKReferenceTable",
    "build_ik_reference",
    "build_ik_reference_bank",
    "build_ik_reference_from_model",
    "interpolate_reference_bank_jax",
    "interpolate_reference_jax",
]
