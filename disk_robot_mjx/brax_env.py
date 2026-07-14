from __future__ import annotations

from pathlib import Path

from disk_robot.model_contract import resolve_model_contract
from disk_robot.walk_config import WalkTaskConfig
from disk_robot.walk_reward import REWARD_TERM_NAMES, reward_terms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_XML_PATH = PROJECT_ROOT / "assets" / "pupper_v3_disk_visual.xml"
XML_PATH = TRAIN_XML_PATH


def _resolve_xml_path(xml_path: str | Path | None) -> Path:
    if xml_path is None:
        return TRAIN_XML_PATH
    path = Path(xml_path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


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
            "MJX/Brax dependencies are missing. Install requirements-mjx.txt in the training environment."
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
            self.contract = resolve_model_contract(self.model)
            self.cpu_data = mujoco.MjData(self.model)
            mujoco.mj_resetDataKeyframe(self.model, self.cpu_data, self.contract.stand_key_id)
            self.cpu_data.ctrl[self.contract.actuator_ids] = self.contract.stand_q
            mujoco.mj_forward(self.model, self.cpu_data)
            self.mjx_model = mjx.put_model(self.model)
            self.base_data = mjx.put_data(self.model, self.cpu_data)

            self.qpos_indices = jp.asarray(self.contract.qpos_indices, dtype=jp.int32)
            self.dof_indices = jp.asarray(self.contract.dof_indices, dtype=jp.int32)
            self.actuator_ids = jp.asarray(self.contract.actuator_ids, dtype=jp.int32)
            self.foot_geom_ids = jp.asarray(self.contract.foot_geom_ids, dtype=jp.int32)
            self.foot_radii = jp.asarray(self.contract.foot_radii)
            self.foot_site_ids = jp.asarray(self.contract.foot_site_ids, dtype=jp.int32)
            self.stand_q = jp.asarray(self.contract.stand_q)
            self.ctrl_low = jp.asarray(self.contract.ctrl_low)
            self.ctrl_high = jp.asarray(self.contract.ctrl_high)
            self.action_scale = jp.asarray(self.config.action_scale)
            if self.action_scale.shape != (self.config.action_size,):
                raise ValueError(f"action_scale must contain {self.config.action_size} values")
            self.torso_body_id = self.contract.torso_body_id
            self.torso_geom_id = self.contract.torso_geom_id
            self.floor_geom_id = self.contract.floor_geom_id

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
            noise_key, height_key, command_key, next_rng = jax.random.split(rng, 4)
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
            joint_q = jp.clip(self.stand_q + noise, self.ctrl_low, self.ctrl_high)
            qpos = self.base_data.qpos.at[self.qpos_indices].set(joint_q).at[2].add(height_noise)
            qvel = jp.zeros_like(self.base_data.qvel)
            ctrl = self.base_data.ctrl.at[self.actuator_ids].set(self.stand_q)
            data = mjx.forward(self.mjx_model, self.base_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl))
            foot_bottom = data.geom_xpos[self.foot_geom_ids, 2] - self.foot_radii
            grounded_qpos = data.qpos.at[2].add(self.config.reset_foot_clearance - jp.min(foot_bottom))
            data = mjx.forward(self.mjx_model, data.replace(qpos=grounded_qpos))

            def settle_step(carry, _):
                return mjx.step(self.mjx_model, carry), None

            if self.settle_steps > 0:
                data = jax.lax.scan(settle_step, data, (), length=self.settle_steps)[0]
            command = self._sample_command(command_key)
            previous_action = jp.zeros(self.config.action_size)
            history = jp.zeros(self.config.observation_size)
            info = {
                "rng": next_rng,
                "command": command,
                "previous_action": previous_action,
                "last_foot_pos": data.site_xpos[self.foot_site_ids],
                "obs_history": history,
                "step_count": jp.array(0, dtype=jp.int32),
                "target_ctrl": self.stand_q,
            }
            obs = self._update_obs_history(history, self._obs_frame(data, previous_action, command))
            info = {**info, "obs_history": obs}
            return State(data, obs, jp.array(0.0), jp.array(0.0), self._empty_metrics(), info)

        def step(self, state, action):
            action = jp.clip(action, -1.0, 1.0)
            command_used = state.info["command"]
            old_pos = state.pipeline_state.xpos[self.torso_body_id]
            target_ctrl = jp.clip(self.stand_q + self.action_scale * action, self.ctrl_low, self.ctrl_high)
            data = state.pipeline_state.replace(
                ctrl=state.pipeline_state.ctrl.at[self.actuator_ids].set(target_ctrl)
            )

            def physics_step(carry, _):
                return mjx.step(self.mjx_model, carry), None

            data = jax.lax.scan(physics_step, data, (), length=max(1, self.config.action_repeat))[0]
            step_count = state.info["step_count"] + 1
            dt = self.model.opt.timestep * max(1, self.config.action_repeat)
            world_velocity = (data.xpos[self.torso_body_id] - old_pos) / jp.maximum(dt, 1e-9)
            torso_mat = data.xmat[self.torso_body_id]
            body_velocity = torso_mat.T @ world_velocity
            body_angular_velocity = torso_mat.T @ data.cvel[self.torso_body_id, :3]
            upright = torso_mat[2, 2]
            torso_height = data.xpos[self.torso_body_id, 2]
            disk_contact_count = self._disk_contact_count(data)
            foot_contacts = self._foot_contacts(data)
            foot_pos = data.site_xpos[self.foot_site_ids]
            foot_xy_velocity = (foot_pos[:, :2] - state.info["last_foot_pos"][:, :2]) / jp.maximum(dt, 1e-9)
            foot_slip = jp.mean(jp.sum(jp.square(foot_xy_velocity), axis=1) * foot_contacts)
            height_failed = torso_height < self.config.min_torso_height
            upright_failed = upright < self.config.terminate_upright
            failed_bool = height_failed | upright_failed
            timeout_bool = step_count >= self.config.max_episode_steps
            failed = failed_bool.astype(jp.float32)
            timeout = timeout_bool.astype(jp.float32)
            done = (failed_bool | timeout_bool).astype(jp.float32)
            action_delta = action - state.info["previous_action"]
            inputs = {
                "velocity_x": body_velocity[0],
                "velocity_y": body_velocity[1],
                "yaw_rate": body_angular_velocity[2],
                "command_x": command_used[0],
                "command_y": command_used[1],
                "command_yaw": command_used[2],
                "vertical_velocity": world_velocity[2],
                "roll_pitch_rate_mean_square": jp.mean(jp.square(body_angular_velocity[:2])),
                "joint_velocity_mean_square": jp.mean(jp.square(data.qvel[self.dof_indices])),
                "upright": upright,
                "disk_contact_count": disk_contact_count,
                "action_mean_square": jp.mean(jp.square(action)),
                "action_delta_mean_square": jp.mean(jp.square(action_delta)),
                "foot_slip_mean_square": foot_slip,
                "failed": failed,
            }
            terms = reward_terms(jp, self.config, inputs)
            reward = sum(terms.values())
            rng, command_key = jax.random.split(state.info["rng"])
            sampled_command = self._sample_command(command_key)
            should_resample = jp.equal(
                jp.mod(step_count, max(1, self.config.command_resample_steps)), 0
            )
            next_command = jp.where(should_resample, sampled_command, command_used)
            history = self._update_obs_history(
                state.info["obs_history"], self._obs_frame(data, action, next_command)
            )
            info = {
                **state.info,
                "rng": rng,
                "command": next_command,
                "previous_action": action,
                "last_foot_pos": foot_pos,
                "obs_history": history,
                "step_count": step_count,
                "target_ctrl": target_ctrl,
            }
            metrics = {
                "reward": reward,
                **{f"reward_{name}": value for name, value in terms.items()},
                "velocity_x": body_velocity[0],
                "velocity_y": body_velocity[1],
                "yaw_rate": body_angular_velocity[2],
                "command_x": command_used[0],
                "command_y": command_used[1],
                "command_yaw": command_used[2],
                "velocity_error_xy": jp.sqrt(
                    jp.square(body_velocity[0] - command_used[0])
                    + jp.square(body_velocity[1] - command_used[1])
                ),
                "yaw_rate_error": jp.abs(body_angular_velocity[2] - command_used[2]),
                "torso_height": torso_height,
                "upright": upright,
                "disk_contact_count": disk_contact_count,
                "foot_contact_count": jp.sum(foot_contacts),
                "foot_slip_mean_square": foot_slip,
                "action_rms": jp.sqrt(jp.mean(jp.square(action))),
                "action_rate_rms": jp.sqrt(jp.mean(jp.square(action_delta))),
                "failed": failed,
                "timeout": timeout,
                "height_failed": height_failed.astype(jp.float32),
                "upright_failed": upright_failed.astype(jp.float32),
            }
            return State(data, history, reward, done, metrics, info)

        def _sample_command(self, rng):
            zero_key, vx_key, vy_key, yaw_key = jax.random.split(rng, 4)
            sampled = jp.array(
                [
                    jax.random.uniform(vx_key, minval=self.config.command_vx_min, maxval=self.config.command_vx_max),
                    jax.random.uniform(vy_key, minval=self.config.command_vy_min, maxval=self.config.command_vy_max),
                    jax.random.uniform(yaw_key, minval=self.config.command_yaw_min, maxval=self.config.command_yaw_max),
                ]
            )
            use_zero = jax.random.uniform(zero_key) < self.config.command_zero_probability
            return jp.where(use_zero, jp.zeros(3), sampled)

        def _obs_frame(self, data, previous_action, command):
            torso_mat = data.xmat[self.torso_body_id]
            body_angular_velocity = torso_mat.T @ data.cvel[self.torso_body_id, :3]
            projected_gravity = torso_mat.T @ jp.array([0.0, 0.0, -1.0])
            return jp.concatenate(
                [
                    body_angular_velocity,
                    projected_gravity,
                    data.qpos[self.qpos_indices] - self.stand_q,
                    data.qvel[self.dof_indices],
                    previous_action,
                    command,
                ]
            )

        def _update_obs_history(self, history, frame):
            size = self.config.observation_frame_size
            return jp.roll(history, size).at[:size].set(frame)

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
            disk_pair = ((geom1 == self.torso_geom_id) & (geom2 == self.floor_geom_id)) | (
                (geom2 == self.torso_geom_id) & (geom1 == self.floor_geom_id)
            )
            return jp.sum(valid & disk_pair).astype(jp.float32)

        def _empty_metrics(self):
            zero = jp.array(0.0)
            names = (
                "reward",
                "velocity_x",
                "velocity_y",
                "yaw_rate",
                "command_x",
                "command_y",
                "command_yaw",
                "velocity_error_xy",
                "yaw_rate_error",
                "torso_height",
                "upright",
                "disk_contact_count",
                "foot_contact_count",
                "foot_slip_mean_square",
                "action_rms",
                "action_rate_rms",
                "failed",
                "timeout",
                "height_failed",
                "upright_failed",
            )
            metrics = {name: zero for name in names}
            metrics.update({f"reward_{name}": zero for name in REWARD_TERM_NAMES})
            return metrics

    return DiskRobotWalkBraxEnv()
