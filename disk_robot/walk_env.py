from __future__ import annotations

from pathlib import Path

import numpy as np

from disk_robot.walk_config import ACTUATOR_NAMES, FOOT_GEOMS, JOINT_NAMES, WalkTaskConfig
from disk_robot.walk_reward import WalkReward, compute_walk_reward


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_ROOT / "assets" / "disk_quadruped_extreme.xml"


class DiskRobotWalkEnv:
    """Small MuJoCo smoke environment for validating the walk task locally."""

    def __init__(self, config: WalkTaskConfig | None = None, xml_path: Path = DEFAULT_XML, seed: int = 0):
        import mujoco

        self.mujoco = mujoco
        self.config = config or WalkTaskConfig()
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.step_count = 0
        self.previous_action = np.zeros(self.config.action_size, dtype=np.float64)
        self.last_reward = WalkReward(0.0, {})

        self.key_id = self.model.key("stand").id
        self.torso_body_id = self.model.body("disk_torso").id
        self.torso_geom_id = self.model.geom("torso_disk").id
        self.floor_geom_id = self.model.geom("floor").id
        self.foot_geom_ids = np.array([self.model.geom(name).id for name in FOOT_GEOMS], dtype=np.int32)
        self.qpos_indices = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in JOINT_NAMES])
        self.dof_indices = np.array([self.model.jnt_dofadr[self.model.joint(name).id] for name in JOINT_NAMES])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in ACTUATOR_NAMES])
        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].copy()

    @property
    def observation_size(self) -> int:
        return self.config.observation_size

    @property
    def action_size(self) -> int:
        return self.config.action_size

    def reset(self):
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        joint_noise = self.rng.uniform(
            -self.config.reset_joint_noise,
            self.config.reset_joint_noise,
            size=self.config.action_size,
        )
        height_noise = self.rng.uniform(-self.config.reset_height_noise, self.config.reset_height_noise)
        joint_q = np.clip(self.data.qpos[self.qpos_indices] + joint_noise, self.ctrl_low, self.ctrl_high)
        self.data.qpos[self.qpos_indices] = joint_q
        self.data.qpos[2] += height_noise
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.actuator_ids] = joint_q
        self.step_count = 0
        self.previous_action[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return self._obs(), self._info(0.0, 0.0, 0.0, 0)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -self.config.action_scale, self.config.action_scale)
        old_x = float(self.data.xpos[self.torso_body_id][0])
        target_ctrl = np.clip(self.data.ctrl[self.actuator_ids] + action, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[self.actuator_ids] = target_ctrl
        for _ in range(max(1, self.config.action_repeat)):
            self.mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        dt = self.model.opt.timestep * max(1, self.config.action_repeat)
        forward_velocity = (float(self.data.xpos[self.torso_body_id][0]) - old_x) / max(dt, 1e-9)
        lateral_velocity = float(self.data.cvel[self.torso_body_id][4])
        torso_height = float(self.data.xpos[self.torso_body_id][2])
        upright = self._upright()
        disk_contacts = self._disk_contact_count()
        foot_contacts = self._foot_contacts()
        action_delta = action - self.previous_action
        reward = compute_walk_reward(
            config=self.config,
            forward_velocity=forward_velocity,
            lateral_velocity=lateral_velocity,
            torso_height=torso_height,
            upright=upright,
            disk_contact_count=disk_contacts,
            foot_contact_count=int(np.sum(foot_contacts)),
            action_mean_square=float(np.mean(np.square(action))),
            action_delta_mean_square=float(np.mean(np.square(action_delta))),
        )
        self.previous_action = action
        self.last_reward = reward
        terminated = torso_height < self.config.min_torso_height or upright < self.config.terminate_upright
        truncated = self.step_count >= self.config.max_episode_steps
        info = self._info(forward_velocity, lateral_velocity, upright, disk_contacts)
        return self._obs(), reward.total, bool(terminated), bool(truncated), info

    def _obs(self):
        foot_contacts = self._foot_contacts()
        obs = np.concatenate(
            [
                self.data.xquat[self.torso_body_id],
                self.data.cvel[self.torso_body_id][3:6],
                self.data.cvel[self.torso_body_id][0:3],
                self.data.qpos[self.qpos_indices],
                self.data.qvel[self.dof_indices],
                self.previous_action,
                foot_contacts.astype(np.float64),
                np.array([self.config.command_velocity], dtype=np.float64),
            ]
        )
        return obs.astype(np.float64, copy=False)

    def _upright(self) -> float:
        mat = self.data.xmat[self.torso_body_id].reshape(3, 3)
        return float(mat[2, 2])

    def _foot_contacts(self):
        contacts = np.zeros(len(self.foot_geom_ids), dtype=bool)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.dist > 0.005:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            for foot_index, foot_geom in enumerate(self.foot_geom_ids):
                if (geom1 == foot_geom and geom2 == self.floor_geom_id) or (
                    geom2 == foot_geom and geom1 == self.floor_geom_id
                ):
                    contacts[foot_index] = True
        return contacts

    def _disk_contact_count(self) -> int:
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.dist > 0.005:
                continue
            geoms = {int(contact.geom1), int(contact.geom2)}
            if self.torso_geom_id in geoms and self.floor_geom_id in geoms:
                count += 1
        return count

    def _info(self, forward_velocity: float, lateral_velocity: float, upright: float, disk_contacts: int):
        foot_contacts = self._foot_contacts()
        return {
            "step_count": self.step_count,
            "forward_velocity": float(forward_velocity),
            "lateral_velocity": float(lateral_velocity),
            "torso_height": float(self.data.xpos[self.torso_body_id][2]),
            "upright": float(upright),
            "disk_contact_count": int(disk_contacts),
            "foot_contact_count": int(np.sum(foot_contacts)),
            "reward_terms": self.last_reward.terms,
        }

