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


def interpolate_reference_jax(jp, table, phase):
    """Periodically interpolates a lookup table for scalar or batched phases."""

    sample = jp.mod(phase, 1.0) * table.shape[0]
    lower = jp.floor(sample).astype(jp.int32)
    upper = jp.mod(lower + 1, table.shape[0])
    alpha = sample - jp.floor(sample)
    return table[lower] + alpha[..., None] * (table[upper] - table[lower])


__all__ = [
    "IKReferenceSpec",
    "IKReferenceTable",
    "build_ik_reference",
    "build_ik_reference_from_model",
    "interpolate_reference_jax",
]
