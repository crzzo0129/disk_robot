from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from disk_robot.gait import LEG_ORDER
from disk_robot.model_contract import ModelContract


@dataclass(frozen=True)
class FootTrajectoryParams:
    frequency: float = 0.8
    stride_length: float = 0.04
    step_height: float = 0.025
    duty: float = 0.72
    mode: str = "crawl"
    damping: float = 1e-4
    max_iterations: int = 30
    tolerance: float = 2e-5


def phase_offsets(mode: str) -> np.ndarray:
    if mode == "crawl":
        values = (0.0, 0.5, 0.75, 0.25)
    elif mode == "trot":
        values = (0.0, 0.5, 0.5, 0.0)
    else:
        raise ValueError(f"Unknown IK gait mode: {mode}")
    return np.asarray(values, dtype=np.float64)


def foot_offset(phase: float, params: FootTrajectoryParams) -> np.ndarray:
    """Returns a smooth body-frame x/z offset from the neutral foot position."""

    phase %= 1.0
    duty = float(np.clip(params.duty, 1e-4, 1.0 - 1e-4))
    if phase < duty:
        u = phase / duty
        x = params.stride_length * (0.5 - u)
        z = 0.0
    else:
        u = (phase - duty) / (1.0 - duty)
        swing_fraction = 1.0 - duty
        x0 = -0.5 * params.stride_length
        x1 = 0.5 * params.stride_length
        # Match the stance velocity at liftoff and touchdown for a C1 cycle.
        tangent = -params.stride_length * swing_fraction / duty
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        x = h00 * x0 + h10 * tangent + h01 * x1 + h11 * tangent
        # This sixth-order bump has zero velocity and acceleration at contact.
        z = 64.0 * params.step_height * u**3 * (1.0 - u) ** 3
    return np.array((x, 0.0, z), dtype=np.float64)


class FootSpaceIKGait:
    """Converts body-frame foot trajectories into joint position targets."""

    def __init__(self, model, contract: ModelContract, params: FootTrajectoryParams | None = None):
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.contract = contract
        self.params = params or FootTrajectoryParams()
        self.data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, self.data, contract.stand_key_id)
        self.data.qpos[contract.qpos_indices] = contract.stand_q
        mujoco.mj_forward(model, self.data)

        torso_pos = self.data.xpos[contract.torso_body_id].copy()
        torso_rot = self.data.xmat[contract.torso_body_id].reshape(3, 3).copy()
        self.torso_pos = torso_pos
        self.torso_rot = torso_rot
        self.neutral_feet = np.asarray(
            [torso_rot.T @ (self.data.site_xpos[site_id] - torso_pos) for site_id in contract.foot_site_ids]
        )
        self.solution = contract.stand_q.copy()
        self.offsets = phase_offsets(self.params.mode)
        self.last_errors = np.zeros(4, dtype=np.float64)

    def reset(self) -> None:
        self.solution = self.contract.stand_q.copy()
        self.last_errors[:] = 0.0

    def targets(self, t: float) -> np.ndarray:
        phases = (t * self.params.frequency + self.offsets) % 1.0
        desired_rel = np.asarray(
            [self.neutral_feet[i] + foot_offset(phases[i], self.params) for i in range(4)]
        )
        desired_world = self.torso_pos + desired_rel @ self.torso_rot.T

        self.data.qpos[self.contract.qpos_indices] = self.solution
        self.mujoco.mj_forward(self.model, self.data)
        for leg_index, site_id in enumerate(self.contract.foot_site_ids):
            q_slice = slice(leg_index * 3, leg_index * 3 + 3)
            qpos_ids = self.contract.qpos_indices[q_slice]
            dof_ids = self.contract.dof_indices[q_slice]
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)

            for _ in range(self.params.max_iterations):
                self.mujoco.mj_forward(self.model, self.data)
                error = desired_world[leg_index] - self.data.site_xpos[site_id]
                if np.linalg.norm(error) <= self.params.tolerance:
                    break
                self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, int(site_id))
                jac = jacp[:, dof_ids]
                system = jac @ jac.T + self.params.damping * np.eye(3)
                delta = jac.T @ np.linalg.solve(system, error)
                delta = np.clip(delta, -0.15, 0.15)
                self.data.qpos[qpos_ids] = np.clip(
                    self.data.qpos[qpos_ids] + delta,
                    self.contract.ctrl_low[q_slice],
                    self.contract.ctrl_high[q_slice],
                )

            self.solution[q_slice] = self.data.qpos[qpos_ids]
            self.last_errors[leg_index] = np.linalg.norm(
                desired_world[leg_index] - self.data.site_xpos[site_id]
            )

        return self.solution.copy()


__all__ = ["FootSpaceIKGait", "FootTrajectoryParams", "foot_offset", "phase_offsets"]
