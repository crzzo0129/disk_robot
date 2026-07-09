from __future__ import annotations

from pathlib import Path

from disk_robot.gait import (
    GaitParams,
    desired_contacts_at_time_jax,
    leg_phase_offsets,
    make_open_loop_targets_jax,
    phase_observation_jax,
)
from disk_robot.walk_config import ACTUATOR_NAMES, FOOT_GEOMS, JOINT_NAMES, WalkTaskConfig
from disk_robot.walk_reward import REWARD_TERM_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_XML_PATH = PROJECT_ROOT / "assets" / "disk_quadruped_extreme.xml"
TRAIN_XML_PATH = PROJECT_ROOT / "assets" / "disk_quadruped_extreme_train.xml"
XML_PATH = TRAIN_XML_PATH


def _resolve_xml_path(xml_path: str | Path | None) -> Path:
    if xml_path is None:
        return TRAIN_XML_PATH
    path = Path(xml_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def make_brax_env(
    config: WalkTaskConfig | None = None,
    seed: int = 0,
    settle_steps: int = 0,
    xml_path: str | Path | None = None,
):
    try:
        import jax
        import jax.numpy as jp
        import mujoco
        from brax.envs.base import Env, State
        from mujoco import mjx
    except ImportError as exc:
        raise RuntimeError(
            "MJX/Brax walk training dependencies are missing. Activate the cloud "
            "MJX environment with jax, mujoco, mujoco-mjx, and brax installed."
        ) from exc

    task_config = config or WalkTaskConfig()
    selected_xml_path = _resolve_xml_path(xml_path)

    class DiskRobotWalkBraxEnv(Env):
        def __init__(self):
            self.config = task_config
            self.seed = int(seed)
            self.settle_steps = int(settle_steps)
            self.xml_path = selected_xml_path
            self.model = mujoco.MjModel.from_xml_path(str(selected_xml_path))
            self.cpu_data = mujoco.MjData(self.model)
            key_id = self.model.key("walk_stand").id
            mujoco.mj_resetDataKeyframe(self.model, self.cpu_data, key_id)
            mujoco.mj_forward(self.model, self.cpu_data)
            self.mjx_model = mjx.put_model(self.model)
            self.base_data = mjx.put_data(self.model, self.cpu_data)

            self.qpos_indices = jp.array(
                [self.model.jnt_qposadr[self.model.joint(name).id] for name in JOINT_NAMES],
                dtype=jp.int32,
            )
            self.dof_indices = jp.array(
                [self.model.jnt_dofadr[self.model.joint(name).id] for name in JOINT_NAMES],
                dtype=jp.int32,
            )
            actuator_ids = [self.model.actuator(name).id for name in ACTUATOR_NAMES]
            self.actuator_ids = jp.array(actuator_ids, dtype=jp.int32)
            self.ctrl_low = jp.array(self.model.actuator_ctrlrange[actuator_ids, 0])
            self.ctrl_high = jp.array(self.model.actuator_ctrlrange[actuator_ids, 1])
            self.torso_body_id = int(self.model.body("disk_torso").id)
            self.torso_geom_id = int(self.model.geom("torso_disk").id)
            self.floor_geom_id = int(self.model.geom("floor").id)
            self.foot_geom_ids = jp.array([self.model.geom(name).id for name in FOOT_GEOMS], dtype=jp.int32)
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
            self.gait_phase_offsets = jp.array(leg_phase_offsets(self.config.gait_mode))

        @property
        def observation_size(self):
            return self.config.observation_size

        @property
        def action_size(self):
            return self.config.action_size

        @property
        def backend(self):
            return "mjx"

        def reset(self, rng):
            rng = jax.random.fold_in(rng, self.seed)
            noise_key, height_key = jax.random.split(rng)
            noise = jax.random.uniform(
                noise_key,
                shape=(self.config.action_size,),
                minval=-self.config.reset_joint_noise,
                maxval=self.config.reset_joint_noise,
            )
            height_noise = jax.random.uniform(
                height_key,
                minval=-self.config.reset_height_noise,
                maxval=self.config.reset_height_noise,
            )
            joint_q = jp.clip(self.base_data.qpos[self.qpos_indices] + noise, self.ctrl_low, self.ctrl_high)
            qpos = self.base_data.qpos.at[self.qpos_indices].set(joint_q)
            qpos = qpos.at[2].add(height_noise)
            qvel = jp.zeros_like(self.base_data.qvel)
            gait_ctrl = self._gait_ctrl(joint_q, jp.array(0, dtype=jp.int32))
            initial_ctrl = jp.where(self.config.use_open_loop_gait, gait_ctrl, joint_q)
            ctrl = self.base_data.ctrl.at[self.actuator_ids].set(initial_ctrl)
            data = self.base_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
            data = mjx.forward(self.mjx_model, data)

            def settle_step(carry, _):
                return mjx.step(self.mjx_model, carry), None

            if self.settle_steps > 0:
                data = jax.lax.scan(settle_step, data, (), length=self.settle_steps)[0]
            foot_contacts = self._foot_contacts(data)
            info = dict(
                neutral_ctrl=joint_q,
                target_ctrl=initial_ctrl,
                previous_action=jp.zeros(self.config.action_size),
                last_foot_contacts=foot_contacts,
                feet_air_time=jp.zeros_like(foot_contacts),
                last_foot_pos=data.geom_xpos[self.foot_geom_ids],
                obs_history=jp.zeros(self.config.observation_size),
                step_count=jp.array(0, dtype=jp.int32),
            )
            metrics = self._empty_metrics()
            obs = self._update_obs_history(info["obs_history"], self._obs_frame(data, info["previous_action"], info["step_count"]))
            info = {**info, "obs_history": obs}
            return State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

        def step(self, state, action):
            action = jp.clip(action, -1.0, 1.0)
            old_pos = state.pipeline_state.xpos[self.torso_body_id]
            gait_ctrl = self._gait_ctrl(state.info["neutral_ctrl"], state.info["step_count"])
            action_scale = jp.where(self.config.use_open_loop_gait, self.config.residual_action_scale, self.config.action_scale)
            base_ctrl = jp.where(self.config.use_open_loop_gait, gait_ctrl, state.info["neutral_ctrl"])
            target_ctrl = jp.clip(
                base_ctrl + action_scale * action,
                self.ctrl_low,
                self.ctrl_high,
            )
            data = state.pipeline_state.replace(
                ctrl=state.pipeline_state.ctrl.at[self.actuator_ids].set(target_ctrl)
            )

            def physics_step(carry, _):
                return mjx.step(self.mjx_model, carry), None

            data = jax.lax.scan(physics_step, data, (), length=max(1, self.config.action_repeat))[0]
            step_count = state.info["step_count"] + 1
            dt = self.model.opt.timestep * max(1, self.config.action_repeat)
            world_velocity = (data.xpos[self.torso_body_id] - old_pos) / jp.maximum(dt, 1e-9)
            torso_x_axis = data.xmat[self.torso_body_id, :, 0]
            torso_y_axis = data.xmat[self.torso_body_id, :, 1]
            forward_velocity = jp.dot(world_velocity, torso_x_axis)
            lateral_velocity = jp.dot(world_velocity, torso_y_axis)
            yaw_rate = data.cvel[self.torso_body_id][2]
            heading_cos, heading_sin = self._heading_observation(data)
            heading_error = jp.arctan2(heading_sin, heading_cos)
            vertical_velocity = data.cvel[self.torso_body_id][5]
            roll_pitch_rate_mean_square = jp.mean(jp.square(data.cvel[self.torso_body_id][0:2]))
            joint_velocity_mean_square = jp.mean(jp.square(data.qvel[self.dof_indices]))
            torso_height = data.xpos[self.torso_body_id][2]
            upright = self._upright(data)
            foot_contacts = self._foot_contacts(data)
            foot_contact_count = jp.sum(foot_contacts)
            first_contact = (state.info["feet_air_time"] > 0.0) * foot_contacts
            feet_air_time = state.info["feet_air_time"] + dt * (1.0 - foot_contacts)
            disk_contact_count = self._disk_contact_count(data)
            foot_pos = data.geom_xpos[self.foot_geom_ids]
            foot_xy_velocity = (foot_pos[:, :2] - state.info["last_foot_pos"][:, :2]) / jp.maximum(dt, 1e-9)
            foot_slip_mean_square = jp.mean(jp.sum(jp.square(foot_xy_velocity), axis=1) * foot_contacts)
            height_failed = torso_height < self.config.min_torso_height
            upright_failed = upright < self.config.terminate_upright
            timeout = step_count >= self.config.max_episode_steps
            failed = height_failed | upright_failed
            done = jp.where(failed | timeout, 1.0, 0.0)
            height_failed = height_failed.astype(jp.float32)
            upright_failed = upright_failed.astype(jp.float32)
            timeout = timeout.astype(jp.float32)
            failed = failed.astype(jp.float32)
            action_delta = action - state.info["previous_action"]
            reward, metrics = self._reward(
                forward_velocity,
                lateral_velocity,
                vertical_velocity,
                roll_pitch_rate_mean_square,
                joint_velocity_mean_square,
                torso_height,
                upright,
                disk_contact_count,
                foot_contact_count,
                yaw_rate,
                heading_error,
                self._contact_schedule_match(foot_contacts, step_count),
                state.info["feet_air_time"],
                first_contact,
                action,
                action_delta,
                foot_slip_mean_square,
                failed,
                timeout,
                height_failed,
                upright_failed,
            )
            info = {
                **state.info,
                "target_ctrl": target_ctrl,
                "previous_action": action,
                "last_foot_contacts": foot_contacts,
                "feet_air_time": feet_air_time * (1.0 - foot_contacts),
                "last_foot_pos": foot_pos,
                "step_count": step_count,
            }
            obs = self._update_obs_history(state.info["obs_history"], self._obs_frame(data, action, step_count))
            info = {**info, "obs_history": obs}
            return State(data, obs, reward, done, metrics, info)

        def _update_obs_history(self, obs_history, obs_frame):
            return jp.roll(obs_history, self.config.observation_frame_size).at[: self.config.observation_frame_size].set(
                obs_frame
            )

        def _gait_time(self, step_count):
            dt = self.model.opt.timestep * max(1, self.config.action_repeat)
            return self.config.gait_time_offset + step_count * dt

        def _gait_phase(self, step_count):
            return jp.mod(self._gait_time(step_count) * self.config.gait_frequency, 1.0)

        def _gait_ctrl(self, neutral_ctrl, step_count):
            return make_open_loop_targets_jax(jp, neutral_ctrl, self._gait_time(step_count), self.gait_params, self.gait_phase_offsets)

        def _obs_frame(self, data, previous_action, step_count):
            return jp.concatenate(
                [
                    data.xquat[self.torso_body_id],
                    data.cvel[self.torso_body_id][3:6],
                    data.cvel[self.torso_body_id][0:3],
                    jp.array([data.xpos[self.torso_body_id][2]]),
                    data.qpos[self.qpos_indices],
                    data.qvel[self.dof_indices],
                    previous_action,
                    self._foot_contacts(data),
                    jp.array([self.config.command_velocity]),
                    phase_observation_jax(jp, self._gait_phase(step_count)),
                    jp.array(self._heading_observation(data)),
                    self._desired_contacts(step_count),
                ]
            )

        def _upright(self, data):
            return data.xmat[self.torso_body_id, 2, 2]

        def _heading_observation(self, data):
            heading = data.xmat[self.torso_body_id, :, 0]
            norm = jp.maximum(jp.linalg.norm(heading[:2]), 1e-6)
            return heading[0] / norm, heading[1] / norm

        def _desired_contacts(self, step_count):
            return desired_contacts_at_time_jax(jp, self._gait_time(step_count), self.gait_params, self.gait_phase_offsets)

        def _contact_schedule_match(self, foot_contacts, step_count):
            desired = self._desired_contacts(step_count)
            return jp.mean(1.0 - jp.abs(foot_contacts - desired))

        def _contact_geoms(self, data):
            contact = data.contact
            if hasattr(contact, "geom1") and hasattr(contact, "geom2"):
                return contact.geom1, contact.geom2
            return contact.geom[:, 0], contact.geom[:, 1]

        def _foot_contacts(self, data):
            geom1, geom2 = self._contact_geoms(data)
            valid = (data.contact.dist <= 0.005) & (geom1 >= 0) & (geom2 >= 0)
            floor_pair = (geom1 == self.floor_geom_id) | (geom2 == self.floor_geom_id)
            touched = (geom1[:, None] == self.foot_geom_ids[None, :]) | (
                geom2[:, None] == self.foot_geom_ids[None, :]
            )
            return jp.any(valid[:, None] & floor_pair[:, None] & touched, axis=0).astype(jp.float32)

        def _disk_contact_count(self, data):
            geom1, geom2 = self._contact_geoms(data)
            valid = (data.contact.dist <= 0.005) & (geom1 >= 0) & (geom2 >= 0)
            disk_pair = (
                ((geom1 == self.torso_geom_id) & (geom2 == self.floor_geom_id))
                | ((geom2 == self.torso_geom_id) & (geom1 == self.floor_geom_id))
            )
            return jp.sum(valid & disk_pair).astype(jp.float32)

        def _reward(
            self,
            forward_velocity,
            lateral_velocity,
            vertical_velocity,
            roll_pitch_rate_mean_square,
            joint_velocity_mean_square,
            torso_height,
            upright,
            disk_contact_count,
            foot_contact_count,
            yaw_rate,
            heading_error,
            contact_schedule_match,
            feet_air_time,
            first_contact,
            action,
            action_delta,
            foot_slip_mean_square,
            failed,
            timeout,
            height_failed,
            upright_failed,
        ):
            cfg = self.config
            velocity_error = forward_velocity - cfg.command_velocity
            height_error = torso_height - cfg.target_torso_height
            terms = {
                "velocity": cfg.reward_velocity * jp.exp(-(velocity_error * velocity_error) / cfg.tracking_sigma),
                "forward": cfg.reward_forward * forward_velocity,
                "lateral": -cfg.reward_lateral * jp.abs( lateral_velocity),
                "yaw": -cfg.penalty_yaw_rate * yaw_rate * yaw_rate,
                "heading": -cfg.penalty_heading_error * heading_error * heading_error,
                "lin_vel_z": -cfg.penalty_lin_vel_z * vertical_velocity * vertical_velocity,
                "ang_vel_xy": -cfg.penalty_ang_vel_xy * roll_pitch_rate_mean_square,
                "joint_vel": -cfg.penalty_joint_vel * joint_velocity_mean_square,
                "upright": -cfg.reward_upright * jp.maximum(0.0, 1.0 - upright),
                "upright_positive": cfg.reward_upright_positive * jp.maximum(0.0, upright),
                "height": -cfg.reward_height * jp.maximum(0.0, cfg.min_torso_height - torso_height),
                "height_target": cfg.reward_height_target * jp.exp(
                    -(height_error * height_error) / cfg.height_tracking_sigma
                ),
                "contact": cfg.reward_contact * jp.minimum(foot_contact_count, 4.0) / 4.0,
                "contact_schedule": cfg.reward_contact_schedule * contact_schedule_match,
                "feet_air_time": cfg.reward_feet_air_time
                * jp.sum(jp.maximum(feet_air_time - cfg.min_feet_air_time, 0.0) * first_contact),
                "disk_contact": -cfg.penalty_disk_contact * disk_contact_count,
                "action": -cfg.penalty_action * jp.mean(jp.square(action)),
                "action_delta": -cfg.penalty_action_delta * jp.mean(jp.square(action_delta)),
                "foot_slip": -cfg.penalty_foot_slip * foot_slip_mean_square,
                "termination": -cfg.penalty_termination * failed,
                "alive": jp.array(cfg.reward_alive),
            }
            reward = sum(terms.values())
            metrics = {
                "reward": reward,
                **{f"reward_{name}": value for name, value in terms.items()},
                "forward_velocity": forward_velocity,
                "lateral_velocity": lateral_velocity,
                "yaw_rate": yaw_rate,
                "heading_error": heading_error,
                "contact_schedule_match": contact_schedule_match,
                "vertical_velocity": vertical_velocity,
                "roll_pitch_rate_mean_square": roll_pitch_rate_mean_square,
                "joint_velocity_mean_square": joint_velocity_mean_square,
                "foot_slip_mean_square": foot_slip_mean_square,
                "torso_height": torso_height,
                "upright": upright,
                "failed": failed,
                "timeout": timeout,
                "height_failed": height_failed,
                "upright_failed": upright_failed,
                "foot_contact_count": foot_contact_count,
                "feet_air_time": jp.sum(feet_air_time),
                "first_contact_count": jp.sum(first_contact),
                "disk_contact_count": disk_contact_count,
            }
            return reward, metrics

        def _empty_metrics(self):
            zero = jp.array(0.0)
            metrics = {
                "reward": zero,
                "forward_velocity": zero,
                "lateral_velocity": zero,
                "yaw_rate": zero,
                "heading_error": zero,
                "contact_schedule_match": zero,
                "vertical_velocity": zero,
                "roll_pitch_rate_mean_square": zero,
                "joint_velocity_mean_square": zero,
                "foot_slip_mean_square": zero,
                "torso_height": zero,
                "upright": zero,
                "failed": zero,
                "timeout": zero,
                "height_failed": zero,
                "upright_failed": zero,
                "foot_contact_count": zero,
                "feet_air_time": zero,
                "first_contact_count": zero,
                "disk_contact_count": zero,
            }
            metrics.update({f"reward_{name}": zero for name in REWARD_TERM_NAMES})
            return metrics

    return DiskRobotWalkBraxEnv()
