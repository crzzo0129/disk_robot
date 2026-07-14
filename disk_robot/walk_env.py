from __future__ import annotations

from pathlib import Path

import numpy as np

from disk_robot.gait import PUPPER_FORWARD_TEACHER, make_open_loop_targets
from disk_robot.model_contract import resolve_model_contract
from disk_robot.walk_config import WalkTaskConfig
from disk_robot.walk_reward import WalkReward, WalkRewardInputs, compute_walk_reward


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_ROOT / "assets" / "pupper_v3_disk_visual.xml"


class DiskRobotWalkEnv:
    """CPU MuJoCo reference for the gait-free command-conditioned task."""

    def __init__(self, config: WalkTaskConfig | None = None, xml_path: Path = DEFAULT_XML, seed: int = 0):
        import mujoco

        self.mujoco = mujoco
        self.config = config or WalkTaskConfig()
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.contract = resolve_model_contract(self.model)
        self.action_scale = np.asarray(self.config.action_scale, dtype=np.float64)
        if self.action_scale.shape != (self.config.action_size,):
            raise ValueError(f"action_scale must contain {self.config.action_size} values")

        self.step_count = 0
        self.command = np.zeros(3, dtype=np.float64)
        self.previous_action = np.zeros(self.config.action_size, dtype=np.float64)
        self.obs_history = np.zeros(self.config.observation_size, dtype=np.float64)
        self.last_foot_pos = np.zeros((4, 3), dtype=np.float64)
        self.last_reward = WalkReward(0.0, {})

    @property
    def observation_size(self) -> int:
        return self.config.observation_size

    @property
    def action_size(self) -> int:
        return self.config.action_size

    def reset(self):
        c = self.contract
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, c.stand_key_id)
        joint_noise = self.rng.uniform(
            -self.config.reset_joint_noise,
            self.config.reset_joint_noise,
            size=self.config.action_size,
        )
        self.data.qpos[c.qpos_indices] = np.clip(c.stand_q + joint_noise, c.ctrl_low, c.ctrl_high)
        self.data.qpos[2] += self.rng.uniform(-self.config.reset_height_noise, self.config.reset_height_noise)
        self.data.qvel[:] = 0.0
        self.data.ctrl[c.actuator_ids] = c.stand_q
        self.mujoco.mj_forward(self.model, self.data)
        foot_bottom = self.data.geom_xpos[c.foot_geom_ids, 2] - c.foot_radii
        self.data.qpos[2] += self.config.reset_foot_clearance - float(np.min(foot_bottom))
        self.mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.command = self._sample_command()
        self.previous_action[:] = 0.0
        self.obs_history[:] = 0.0
        self.last_foot_pos = self.data.site_xpos[c.foot_site_ids].copy()
        self.last_reward = WalkReward(0.0, {})
        obs = self._update_obs_history(self._obs_frame())
        return obs, self._info(np.zeros(3), self.command)

    def step(self, action):
        c = self.contract
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        command_used = self.command.copy()
        old_pos = self.data.xpos[c.torso_body_id].copy()
        teacher_action = self._teacher_action(self.step_count)
        blended_action = self.config.teacher_blend * teacher_action + (1.0 - self.config.teacher_blend) * action
        target_ctrl = np.clip(c.stand_q + self.action_scale * blended_action, c.ctrl_low, c.ctrl_high)
        self.data.ctrl[c.actuator_ids] = target_ctrl
        for _ in range(max(1, self.config.action_repeat)):
            self.mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        dt = self.model.opt.timestep * max(1, self.config.action_repeat)
        world_velocity = (self.data.xpos[c.torso_body_id] - old_pos) / max(dt, 1e-9)
        torso_mat = self.data.xmat[c.torso_body_id].reshape(3, 3)
        body_velocity = torso_mat.T @ world_velocity
        body_angular_velocity = torso_mat.T @ self.data.cvel[c.torso_body_id, :3]
        body_linear_velocity = torso_mat.T @ self.data.cvel[c.torso_body_id, 3:6]
        upright = float(torso_mat[2, 2])
        torso_height = float(self.data.xpos[c.torso_body_id, 2])
        disk_contacts = self._disk_contact_count()
        foot_contacts = self._foot_contacts()
        foot_pos = self.data.site_xpos[c.foot_site_ids].copy()
        foot_xy_velocity = (foot_pos[:, :2] - self.last_foot_pos[:, :2]) / max(dt, 1e-9)
        foot_slip = float(np.mean(np.sum(np.square(foot_xy_velocity), axis=1) * foot_contacts))
        terminated = torso_height < self.config.min_torso_height or upright < self.config.terminate_upright
        truncated = self.step_count >= self.config.max_episode_steps
        action_delta = action - self.previous_action
        inputs = WalkRewardInputs(
            velocity_x=float(body_velocity[0]),
            velocity_y=float(body_velocity[1]),
            yaw_rate=float(body_angular_velocity[2]),
            command_x=float(command_used[0]),
            command_y=float(command_used[1]),
            command_yaw=float(command_used[2]),
            vertical_velocity=float(world_velocity[2]),
            roll_pitch_rate_mean_square=float(np.mean(np.square(body_angular_velocity[:2]))),
            joint_velocity_mean_square=float(np.mean(np.square(self.data.qvel[c.dof_indices]))),
            upright=upright,
            disk_contact_count=disk_contacts,
            action_mean_square=float(np.mean(np.square(action))),
            action_delta_mean_square=float(np.mean(np.square(action_delta))),
            foot_slip_mean_square=foot_slip,
            failed=bool(terminated),
            teacher_action_error=float(np.mean(np.square(action - teacher_action))),
        )
        reward = compute_walk_reward(config=self.config, inputs=inputs)
        self.previous_action = action
        self.last_foot_pos = foot_pos
        self.last_reward = reward
        info = self._info(body_velocity, command_used)

        if self.step_count % max(1, self.config.command_resample_steps) == 0:
            self.command = self._sample_command()
        obs = self._update_obs_history(self._obs_frame())
        return obs, reward.total, bool(terminated), bool(truncated), info

    def _teacher_action(self, step_count):
        dt = self.model.opt.timestep * max(1, self.config.action_repeat)
        targets = make_open_loop_targets(
            self.contract.stand_q,
            step_count * dt,
            PUPPER_FORWARD_TEACHER,
        )
        return np.clip((targets - self.contract.stand_q) / self.action_scale, -1.0, 1.0)

    def _sample_command(self):
        if self.rng.random() < self.config.command_zero_probability:
            return np.zeros(3, dtype=np.float64)
        return np.array(
            [
                self.rng.uniform(self.config.command_vx_min, self.config.command_vx_max),
                self.rng.uniform(self.config.command_vy_min, self.config.command_vy_max),
                self.rng.uniform(self.config.command_yaw_min, self.config.command_yaw_max),
            ],
            dtype=np.float64,
        )

    def _obs_frame(self):
        c = self.contract
        torso_mat = self.data.xmat[c.torso_body_id].reshape(3, 3)
        body_angular_velocity = torso_mat.T @ self.data.cvel[c.torso_body_id, :3]
        body_linear_velocity = torso_mat.T @ self.data.cvel[c.torso_body_id, 3:6]
        projected_gravity = torso_mat.T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate(
            [
                body_angular_velocity,
                projected_gravity,
                body_linear_velocity,
                self.data.qpos[c.qpos_indices] - c.stand_q,
                self.data.qvel[c.dof_indices],
                self.previous_action,
                self.command,
            ]
        ).astype(np.float64, copy=False)

    def _update_obs_history(self, obs_frame):
        frame_size = self.config.observation_frame_size
        self.obs_history = np.roll(self.obs_history, frame_size)
        self.obs_history[:frame_size] = obs_frame
        return self.obs_history.copy()

    def _foot_contacts(self):
        c = self.contract
        contacts = np.zeros(4, dtype=np.float64)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.dist > 0.005:
                continue
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            for foot_index, foot_geom in enumerate(c.foot_geom_ids):
                if (geom1 == foot_geom and geom2 == c.floor_geom_id) or (
                    geom2 == foot_geom and geom1 == c.floor_geom_id
                ):
                    contacts[foot_index] = 1.0
        return contacts

    def _disk_contact_count(self):
        c = self.contract
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.dist <= 0.005 and {int(contact.geom1), int(contact.geom2)} == {
                c.torso_geom_id,
                c.floor_geom_id,
            }:
                count += 1
        return count

    def _info(self, body_velocity, command):
        c = self.contract
        torso_mat = self.data.xmat[c.torso_body_id].reshape(3, 3)
        body_angular_velocity = torso_mat.T @ self.data.cvel[c.torso_body_id, :3]
        return {
            "step_count": self.step_count,
            "velocity_x": float(body_velocity[0]),
            "velocity_y": float(body_velocity[1]),
            "yaw_rate": float(body_angular_velocity[2]),
            "command_x": float(command[0]),
            "command_y": float(command[1]),
            "command_yaw": float(command[2]),
            "torso_height": float(self.data.xpos[c.torso_body_id, 2]),
            "upright": float(torso_mat[2, 2]),
            "disk_contact_count": self._disk_contact_count(),
            "foot_contact_count": int(np.sum(self._foot_contacts())),
            "reward_terms": self.last_reward.terms,
            "target_ctrl": self.data.ctrl[c.actuator_ids].copy(),
        }
