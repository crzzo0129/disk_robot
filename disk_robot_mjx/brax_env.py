from __future__ import annotations

from pathlib import Path

from disk_robot.walk_config import ACTUATOR_NAMES, FOOT_GEOMS, JOINT_NAMES, WalkTaskConfig


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
            key_id = self.model.key("stand").id
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
            ctrl = self.base_data.ctrl.at[self.actuator_ids].set(joint_q)
            data = self.base_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
            data = mjx.forward(self.mjx_model, data)

            def settle_step(carry, _):
                return mjx.step(self.mjx_model, carry), None

            if self.settle_steps > 0:
                data = jax.lax.scan(settle_step, data, (), length=self.settle_steps)[0]
            foot_contacts = self._foot_contacts(data)
            info = dict(
                neutral_ctrl=joint_q,
                target_ctrl=joint_q,
                previous_action=jp.zeros(self.config.action_size),
                last_foot_contacts=foot_contacts,
                feet_air_time=jp.zeros_like(foot_contacts),
                step_count=jp.array(0, dtype=jp.int32),
            )
            metrics = self._empty_metrics()
            return State(data, self._obs(data, info["previous_action"]), jp.array(0.0), jp.array(0.0), metrics, info)

        def step(self, state, action):
            action = jp.clip(action, -1.0, 1.0)
            old_x = state.pipeline_state.xpos[self.torso_body_id][0]
            target_ctrl = jp.clip(
                state.info["neutral_ctrl"] + self.config.action_scale * action,
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
            forward_velocity = (data.xpos[self.torso_body_id][0] - old_x) / jp.maximum(dt, 1e-9)
            lateral_velocity = data.cvel[self.torso_body_id][4]
            vertical_velocity = data.cvel[self.torso_body_id][5]
            angular_velocity_xy_mean_square = jp.mean(jp.square(data.cvel[self.torso_body_id][0:2]))
            joint_velocity_mean_square = jp.mean(jp.square(data.qvel[self.dof_indices]))
            torso_height = data.xpos[self.torso_body_id][2]
            upright = self._upright(data)
            foot_contacts = self._foot_contacts(data)
            foot_contact_count = jp.sum(foot_contacts)
            first_contact = (state.info["feet_air_time"] > 0.0) * foot_contacts
            feet_air_time = state.info["feet_air_time"] + dt * (1.0 - foot_contacts)
            disk_contact_count = self._disk_contact_count(data)
            done = jp.where(
                (torso_height < self.config.min_torso_height)
                | (upright < self.config.terminate_upright)
                | (step_count >= self.config.max_episode_steps),
                1.0,
                0.0,
            )
            action_delta = action - state.info["previous_action"]
            reward, metrics = self._reward(
                forward_velocity,
                lateral_velocity,
                vertical_velocity,
                angular_velocity_xy_mean_square,
                joint_velocity_mean_square,
                torso_height,
                upright,
                disk_contact_count,
                foot_contact_count,
                state.info["feet_air_time"],
                first_contact,
                action,
                action_delta,
                done,
            )
            info = {
                **state.info,
                "target_ctrl": target_ctrl,
                "previous_action": action,
                "last_foot_contacts": foot_contacts,
                "feet_air_time": feet_air_time * (1.0 - foot_contacts),
                "step_count": step_count,
            }
            return State(data, self._obs(data, action), reward, done, metrics, info)

        def _obs(self, data, previous_action):
            return jp.concatenate(
                [
                    data.xquat[self.torso_body_id],
                    data.cvel[self.torso_body_id][3:6],
                    data.cvel[self.torso_body_id][0:3],
                    data.qpos[self.qpos_indices],
                    data.qvel[self.dof_indices],
                    previous_action,
                    self._foot_contacts(data),
                    jp.array([self.config.command_velocity]),
                ]
            )

        def _upright(self, data):
            return data.xmat[self.torso_body_id, 2, 2]

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
            angular_velocity_xy_mean_square,
            joint_velocity_mean_square,
            torso_height,
            upright,
            disk_contact_count,
            foot_contact_count,
            feet_air_time,
            first_contact,
            action,
            action_delta,
            done,
        ):
            cfg = self.config
            velocity_error = forward_velocity - cfg.command_velocity
            r_velocity = cfg.reward_velocity * jp.exp(-(velocity_error * velocity_error) / cfg.tracking_sigma)
            r_forward = cfg.reward_forward * forward_velocity
            r_lateral = -cfg.reward_lateral * lateral_velocity * lateral_velocity
            r_lin_vel_z = -cfg.penalty_lin_vel_z * vertical_velocity * vertical_velocity
            r_ang_vel_xy = -cfg.penalty_ang_vel_xy * angular_velocity_xy_mean_square
            r_joint_vel = -cfg.penalty_joint_vel * joint_velocity_mean_square
            r_upright = -cfg.reward_upright * jp.maximum(0.0, 1.0 - upright)
            r_upright_positive = cfg.reward_upright_positive * jp.maximum(0.0, upright)
            r_height = -cfg.reward_height * jp.maximum(0.0, cfg.min_torso_height - torso_height)
            height_error = torso_height - cfg.target_torso_height
            r_height_target = cfg.reward_height_target * jp.exp(
                -(height_error * height_error) / cfg.height_tracking_sigma
            )
            r_contact = cfg.reward_contact * jp.minimum(foot_contact_count, 4.0) / 4.0
            r_feet_air_time = cfg.reward_feet_air_time * jp.sum(
                jp.maximum(feet_air_time - cfg.min_feet_air_time, 0.0) * first_contact
            )
            r_disk_contact = -cfg.penalty_disk_contact * disk_contact_count
            r_action = -cfg.penalty_action * jp.mean(jp.square(action))
            r_action_delta = -cfg.penalty_action_delta * jp.mean(jp.square(action_delta))
            r_termination = -cfg.penalty_termination * done
            r_alive = jp.array(cfg.reward_alive)
            reward = (
                r_velocity
                + r_forward
                + r_lateral
                + r_lin_vel_z
                + r_ang_vel_xy
                + r_joint_vel
                + r_upright
                + r_upright_positive
                + r_height
                + r_height_target
                + r_contact
                + r_feet_air_time
                + r_disk_contact
                + r_action
                + r_action_delta
                + r_termination
                + r_alive
            )
            metrics = {
                "reward": reward,
                "reward_velocity": r_velocity,
                "reward_forward": r_forward,
                "reward_lateral": r_lateral,
                "reward_lin_vel_z": r_lin_vel_z,
                "reward_ang_vel_xy": r_ang_vel_xy,
                "reward_joint_vel": r_joint_vel,
                "reward_upright": r_upright,
                "reward_upright_positive": r_upright_positive,
                "reward_height": r_height,
                "reward_height_target": r_height_target,
                "reward_contact": r_contact,
                "reward_feet_air_time": r_feet_air_time,
                "reward_disk_contact": r_disk_contact,
                "reward_action": r_action,
                "reward_action_delta": r_action_delta,
                "reward_termination": r_termination,
                "reward_alive": r_alive,
                "forward_velocity": forward_velocity,
                "vertical_velocity": vertical_velocity,
                "angular_velocity_xy_mean_square": angular_velocity_xy_mean_square,
                "joint_velocity_mean_square": joint_velocity_mean_square,
                "torso_height": torso_height,
                "upright": upright,
                "done": done,
                "foot_contact_count": foot_contact_count,
                "feet_air_time": jp.sum(feet_air_time),
                "first_contact_count": jp.sum(first_contact),
                "disk_contact_count": disk_contact_count,
            }
            return reward, metrics

        def _empty_metrics(self):
            zero = jp.array(0.0)
            return {
                "reward": zero,
                "reward_velocity": zero,
                "reward_forward": zero,
                "reward_lateral": zero,
                "reward_lin_vel_z": zero,
                "reward_ang_vel_xy": zero,
                "reward_joint_vel": zero,
                "reward_upright": zero,
                "reward_upright_positive": zero,
                "reward_height": zero,
                "reward_height_target": zero,
                "reward_contact": zero,
                "reward_feet_air_time": zero,
                "reward_disk_contact": zero,
                "reward_action": zero,
                "reward_action_delta": zero,
                "reward_termination": zero,
                "reward_alive": zero,
                "forward_velocity": zero,
                "vertical_velocity": zero,
                "angular_velocity_xy_mean_square": zero,
                "joint_velocity_mean_square": zero,
                "torso_height": zero,
                "upright": zero,
                "done": zero,
                "foot_contact_count": zero,
                "feet_air_time": zero,
                "first_contact_count": zero,
                "disk_contact_count": zero,
            }

    return DiskRobotWalkBraxEnv()
