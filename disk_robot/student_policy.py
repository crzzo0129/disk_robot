from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StudentPolicyArtifact:
    params: tuple[tuple[np.ndarray, np.ndarray], ...]
    obs_mean: np.ndarray
    obs_std: np.ndarray
    metadata: dict


def load_student_policy(path: str | Path) -> StudentPolicyArtifact:
    path = Path(path).expanduser().resolve()
    with np.load(path) as archive:
        obs_mean = archive["obs_mean"].copy()
        obs_std = archive["obs_std"].copy()
        params = []
        index = 0
        while f"weight_{index}" in archive:
            params.append((archive[f"weight_{index}"].copy(), archive[f"bias_{index}"].copy()))
            index += 1
    if not params:
        raise ValueError(f"No MLP layers found in student policy: {path}")
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return StudentPolicyArtifact(tuple(params), obs_mean, obs_std, metadata)


def apply_student_policy_numpy(artifact: StudentPolicyArtifact, observation) -> np.ndarray:
    value = np.clip(
        (np.asarray(observation, dtype=np.float32) - artifact.obs_mean) / artifact.obs_std,
        -10.0,
        10.0,
    )
    for weight, bias in artifact.params[:-1]:
        value = value @ weight + bias
        value = np.where(value > 0.0, value, np.expm1(value))
    weight, bias = artifact.params[-1]
    return np.tanh(value @ weight + bias)


def make_student_policy_jax(artifact: StudentPolicyArtifact):
    import jax.numpy as jp

    params = tuple((jp.asarray(weight), jp.asarray(bias)) for weight, bias in artifact.params)
    obs_mean = jp.asarray(artifact.obs_mean)
    obs_std = jp.asarray(artifact.obs_std)

    def policy(observation):
        value = jp.clip((observation - obs_mean) / obs_std, -10.0, 10.0)
        for weight, bias in params[:-1]:
            preactivation = value @ weight + bias
            value = jp.where(preactivation > 0.0, preactivation, jp.expm1(preactivation))
        weight, bias = params[-1]
        return jp.tanh(value @ weight + bias)

    return policy


__all__ = [
    "StudentPolicyArtifact",
    "apply_student_policy_numpy",
    "load_student_policy",
    "make_student_policy_jax",
]
