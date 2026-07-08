from __future__ import annotations

from pathlib import Path

import numpy as np

from disk_robot.gait import GaitParams, desired_contacts_at_time, make_open_loop_targets, phase_observation
from disk_robot.walk_config import ACTUATOR_NAMES, FOOT_GEOMS, JOINT_NAMES, WalkTaskConfig
from disk_robot.walk_reward import WalkReward, WalkRewardInputs, compute_walk_reward


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_ROOT / "assets" / "disk_quadruped_extreme_train.xml"


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
        self.neutral_ctrl = np.zeros(self.config.action_size, dtype=np.float64)
        self.obs_history = np.zeros(self.config.observation_size, dtype=np.float64)
        self.last_foot_pos = np.zeros((len(FOOT_GEOMS), 3), dtype=np.float64)
        self.last_reward = WalkReward(0.0, {})

        self.key_id = self.model.key("walk_stand").id
        self.torso_body_id = self.model.body("disk_torso").id
        self.torso_geom_id = self.model.geom("torso_disk").id
        self.floor_geom_id = self.model.geom("floor").id
        self.foot_geom_ids = np.array([self.model.geom(name).id for name in FOOT_GEOMS], dtype=np.int32)
        self.qpos_indices = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in JOINT_NAMES])
        self.dof_indices = np.array([self.model.jnt_dofadr[self.model.joint(name).id] for name in JOINT_NAMES])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in ACTUATOR_NAMES])
        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].copy()
        self.gait_params = GaitParams(
            frequency=self.config.gait_frequency,
            hip_stance_amplitude=self.config.gait_hip_stance_amplitude,
            hip_swing_amplitude=self.config.gait_hip_swing_amplitude,
            knee_lift_amplitude=self.config.gait_knee_lift_amplitude,
            abd_amplitude=self.config.gait_abd_amplitude,
            duty=self.config.gait_duty,
            mode=self.config.gait_mode,
            direction=self.config.gait_direction,
            front_knee_sign=self.config.gait_front_knee_sign,
            hind_knee_sign=self.config.gait_hind_knee_sign,
            march_hip_compensation=self.config.gait_march_hip_compensation,
        )

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
        self.neutral_ctrl = joint_q.copy()
        self.data.qpos[2] += height_noise
        self.data.qvel[:] = 0.0
        self.step_count = 0
        gait_ctrl = self._gait_ctrl(self.neutral_ctrl, self.step_count)
        initial_ctrl = gait_ctrl if self.config.use_open_loop_gait else self.neutral_ctrl
        self.data.ctrl[self.actuator_ids] = initial_ctrl
        self.previous_action[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self.last_foot_pos = self.data.geom_xpos[self.foot_geom_ids].copy()
        self.obs_history[:] = 0.0
        obs = self._update_obs_history(self._obs_frame())
        return obs, self._info(0.0, 0.0, 0.0, float(self._heading_observation()[1]), 0.0, 0)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        old_pos = self.data.xpos[self.torso_body_id].copy()
        gait_ctrl = self._gait_ctrl(self.neutral_ctrl, self.step_count)
        action_scale = self.config.residual_action_scale if self.config.use_open_loop_gait else self.config.action_scale
        base_ctrl = gait_ctrl if self.config.use_open_loop_gait else self.neutral_ctrl
        target_ctrl = np.clip(base_ctrl + action_scale * action, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[self.actuator_ids] = target_ctrl
        for _ in range(max(1, self.config.action_repeat)):
            self.mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        dt = self.model.opt.timestep * max(1, self.config.action_repeat)
        world_velocity = (self.data.xpos[self.torso_body_id] - old_pos) / max(dt, 1e-9)
        torso_mat = self.data.xmat[self.torso_body_id].reshape(3, 3)
        torso_x_axis = torso_mat[:, 0]
        torso_y_axis = torso_mat[:, 1]
        forward_velocity = float(np.dot(world_velocity, torso_x_axis))
        lateral_velocity = float(np.dot(world_velocity, torso_y_axis))
        yaw_rate = float(self.data.cvel[self.torso_body_id][2])
        _, heading_sin = self._heading_observation()
        heading_error = float(heading_sin)
        vertical_velocity = float(self.data.cvel[self.torso_body_id][5])
        roll_pitch_rate_mean_square = float(np.mean(np.square(self.data.cvel[self.torso_body_id][0:2])))
        joint_velocity_mean_square = float(np.mean(np.square(self.data.qvel[self.dof_indices])))
        torso_height = float(self.data.xpos[self.torso_body_id][2])
        upright = self._upright()
        disk_contacts = self._disk_contact_count()
        foot_contacts = self._foot_contacts()
        foot_pos = self.data.geom_xpos[self.foot_geom_ids].copy()
        foot_xy_velocity = (foot_pos[:, :2] - self.last_foot_pos[:, :2]) / max(dt, 1e-9)
        foot_slip_mean_square = float(np.mean(np.sum(np.square(foot_xy_velocity), axis=1) * foot_contacts))
        terminated = torso_height < self.config.min_torso_height or upright < self.config.terminate_upright
        truncated = self.step_count >= self.config.max_episode_steps
        action_delta = action - self.previous_action
        reward_inputs = WalkRewardInputs(
            forward_velocity=forward_velocity,
            lateral_velocity=lateral_velocity,
            yaw_rate=yaw_rate,
            heading_error=heading_error,
            vertical_velocity=vertical_velocity,
            roll_pitch_rate_mean_square=roll_pitch_rate_mean_square,
            joint_velocity_mean_square=joint_velocity_mean_square,
            torso_height=torso_height,
            upright=upright,
            disk_contact_count=disk_contacts,
            foot_contact_count=int(np.sum(foot_contacts)),
            contact_schedule_match=self._contact_schedule_match(foot_contacts),
            action_mean_square=float(np.mean(np.square(action))),
            action_delta_mean_square=float(np.mean(np.square(action_delta))),
            foot_slip_mean_square=foot_slip_mean_square,
            failed=terminated,
        )
        reward = compute_walk_reward(config=self.config, inputs=reward_inputs)
        self.previous_action = action
        self.last_foot_pos = foot_pos
        self.last_reward = reward
        info = self._info(forward_velocity, lateral_velocity, yaw_rate, heading_error, upright, disk_contacts)
        return self._update_obs_history(self._obs_frame()), reward.total, bool(terminated), bool(truncated), info

    def _update_obs_history(self, obs_frame):
        frame_size = self.config.observation_frame_size
        self.obs_history = np.roll(self.obs_history, frame_size)
        self.obs_history[:frame_size] = obs_frame
        return self.obs_history.copy()

    def _gait_time(self, step_count):
        dt = self.model.opt.timestep * max(1, self.config.action_repeat)
        return self.config.gait_time_offset + step_count * dt

    def _gait_phase(self, step_count):
        return (self._gait_time(step_count) * self.config.gait_frequency) % 1.0

    def _gait_ctrl(self, neutral_ctrl, step_count):
        return make_open_loop_targets(neutral_ctrl, self._gait_time(step_count), self.gait_params)

    def _obs_frame(self):
        foot_contacts = self._foot_contacts()
        obs = np.concatenate(
            [
                self.data.xquat[self.torso_body_id],
                self.data.cvel[self.torso_body_id][3:6],
                self.data.cvel[self.torso_body_id][0:3],
                np.array([self.data.xpos[self.torso_body_id][2]], dtype=np.float64),
                self.data.qpos[self.qpos_indices],
                self.data.qvel[self.dof_indices],
                self.previous_action,
                foot_contacts.astype(np.float64),
                np.array([self.config.command_velocity], dtype=np.float64),
                phase_observation(self._gait_phase(self.step_count)),
                np.array(self._heading_observation(), dtype=np.float64),
                self._desired_contacts(),
            ]
        )
        return obs.astype(np.float64, copy=False)

    def _upright(self) -> float:
        mat = self.data.xmat[self.torso_body_id].reshape(3, 3)
        return float(mat[2, 2])

    def _heading_observation(self) -> tuple[float, float]:
        mat = self.data.xmat[self.torso_body_id].reshape(3, 3)
        heading = mat[:, 0]
        norm = max(float(np.linalg.norm(heading[:2])), 1e-6)
        return float(heading[0] / norm), float(heading[1] / norm)

    def _desired_contacts(self):
        return desired_contacts_at_time(self._gait_time(self.step_count), self.gait_params)

    def _contact_schedule_match(self, foot_contacts):
        desired = self._desired_contacts()
        actual = foot_contacts.astype(np.float64)
        return float(np.mean(1.0 - np.abs(actual - desired)))

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

    def _info(
        self,
        forward_velocity: float,
        lateral_velocity: float,
        yaw_rate: float,
        heading_error: float,
        upright: float,
        disk_contacts: int,
    ):
        foot_contacts = self._foot_contacts()
        heading_cos, heading_sin = self._heading_observation()
        torso_pos = self.data.xpos[self.torso_body_id]
        contact_schedule_match = self._contact_schedule_match(foot_contacts)
        return {
            "step_count": self.step_count,
            "forward_velocity": float(forward_velocity),
            "lateral_velocity": float(lateral_velocity),
            "yaw_rate": float(yaw_rate),
            "heading_error": float(heading_error),
            "heading_cos": float(heading_cos),
            "heading_sin": float(heading_sin),
            "torso_x": float(torso_pos[0]),
            "torso_y": float(torso_pos[1]),
            "torso_height": float(torso_pos[2]),
            "upright": float(upright),
            "disk_contact_count": int(disk_contacts),
            "foot_contact_count": int(np.sum(foot_contacts)),
            "contact_schedule_match": float(contact_schedule_match),
            "reward_terms": self.last_reward.terms,
        }

