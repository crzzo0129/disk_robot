from __future__ import annotations

from pathlib import Path

from disk_robot.ik_reference import IKReferenceSpec, IKReferenceTable, build_ik_reference, interpolate_reference_jax
from disk_robot.model_paths import ACTIVE_MODEL_XML
from disk_robot.model_contract import resolve_model_contract
from disk_robot.teacher_student_config import ForwardTeacherStudentConfig


DEFAULT_XML = ACTIVE_MODEL_XML


def make_forward_teacher_student_env(
    role: str,
    config: ForwardTeacherStudentConfig | None = None,
    reference: IKReferenceTable | None = None,
    xml_path: str | Path = DEFAULT_XML,
    seed: int = 0,
):
    """Creates a teacher, DAgger collector, or IK-free student MJX env."""

    if role not in ("teacher", "dagger", "student"):
        raise ValueError("role must be 'teacher', 'dagger', or 'student'")
    try:
        import jax
        import jax.numpy as jp
        import mujoco
        from brax.envs.base import Env, State
        from mujoco import mjx
    except ImportError as exc:
        raise RuntimeError("Teacher/student MJX environment requires JAX, Brax, MuJoCo, and MJX") from exc

    cfg = config or ForwardTeacherStudentConfig()
    selected_xml = Path(xml_path).expanduser().resolve()
    ref = reference
    if role != "student" and ref is None:
        ref = build_ik_reference(selected_xml, IKReferenceSpec())

    class ForwardTeacherStudentEnv(Env):
        def __init__(self):
            self.role = role
            self.config = cfg
            self.reference = ref
            self.seed = int(seed)
            self.xml_path = selected_xml
            self.model = mujoco.MjModel.from_xml_path(str(selected_xml))
            self.contract = resolve_model_contract(self.model)
            if ref is not None and not jp.allclose(
                jp.asarray(ref.stand_q), jp.asarray(self.contract.stand_q)
            ):
                raise ValueError("IK reference stand_q does not match the selected XML stand keyframe")

            ids = self.contract.actuator_ids
            self.model.actuator_gainprm[ids, 0] = cfg.actuator_kp
            self.model.actuator_biasprm[ids, 1] = -cfg.actuator_kp
            self.model.actuator_biasprm[ids, 2] = -cfg.actuator_kd
            self.model.actuator_forcerange[ids, 0] = -cfg.torque_limit
            self.model.actuator_forcerange[ids, 1] = cfg.torque_limit

            cpu_data = mujoco.MjData(self.model)
            mujoco.mj_resetDataKeyframe(self.model, cpu_data, self.contract.stand_key_id)
            mujoco.mj_forward(self.model, cpu_data)
            self.mjx_model = mjx.put_model(self.model)
            self.base_data = mjx.put_data(self.model, cpu_data)

            self.qpos_indices = jp.asarray(self.contract.qpos_indices, dtype=jp.int32)
            self.dof_indices = jp.asarray(self.contract.dof_indices, dtype=jp.int32)
            self.actuator_ids = jp.asarray(ids, dtype=jp.int32)
            self.foot_geom_ids = jp.asarray(self.contract.foot_geom_ids, dtype=jp.int32)
            self.foot_radii = jp.asarray(self.contract.foot_radii)
            self.foot_site_ids = jp.asarray(self.contract.foot_site_ids, dtype=jp.int32)
            self.stand_q = jp.asarray(self.contract.stand_q)
            self.ctrl_low = jp.asarray(self.contract.ctrl_low)
            self.ctrl_high = jp.asarray(self.contract.ctrl_high)
            self.student_action_scale = jp.asarray(cfg.student_action_scale)
            self.residual_scale = jp.asarray(cfg.residual_scale)
            self.frequency = float(cfg.student_phase_frequency)
            if ref is not None:
                self.ik_table = jp.asarray(ref.joint_targets)
                self.phase_offsets = jp.asarray((0.0, 0.5, 0.5, 0.0))
                self.frequency = float(ref.spec.frequency)
                self.duty = float(ref.spec.duty)
            self.dt = self.model.opt.timestep * max(1, cfg.action_repeat)
            self.command = jp.asarray((cfg.command_vx, 0.0, 0.0))
            self.torso_body_id = self.contract.torso_body_id
            self.torso_geom_id = self.contract.torso_geom_id
            self.floor_geom_id = self.contract.floor_geom_id
            root_joint_id = int(self.model.body_jntadr[self.torso_body_id])
            if self.model.jnt_type[root_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("Teacher disturbances require a free root joint")
            root_dof = int(self.model.jnt_dofadr[root_joint_id])
            self.root_linear_dof_indices = jp.arange(root_dof, root_dof + 3, dtype=jp.int32)

        @property
        def observation_size(self):
            if self.role == "teacher":
                return self.config.teacher_observation_size
            return self.config.student_policy_observation_size

        @property
        def action_size(self):
            return self.config.action_size

        @property
        def backend(self):
            return "mjx"

        def reset(self, rng):
            rng = jax.random.fold_in(rng, self.seed)
            (
                phase_key,
                joint_key,
                height_key,
                push_step_key,
                push_angle_key,
                push_magnitude_key,
                motor_key,
                delay_key,
                next_rng,
            ) = jax.random.split(rng, 9)
            if self.config.fixed_reset_phase is not None:
                phase = jp.array(float(self.config.fixed_reset_phase))
                initial_target = self.stand_q
            elif self.role == "student":
                phase = jp.array(0.0)
                initial_target = self.stand_q
            else:
                phase = jax.random.uniform(phase_key)
                initial_target = self.stand_q
            joint_noise_limit = (
                self.config.disturbance_reset_joint_noise
                if self.config.disturbance_enabled
                else self.config.reset_joint_noise
            )
            joint_noise = jax.random.uniform(
                joint_key,
                shape=(self.config.action_size,),
                minval=-joint_noise_limit,
                maxval=joint_noise_limit,
            )
            joint_q = jp.clip(initial_target + joint_noise, self.ctrl_low, self.ctrl_high)
            height_noise_limit = (
                self.config.disturbance_reset_height_noise
                if self.config.disturbance_enabled
                else self.config.reset_height_noise
            )
            height_noise = jax.random.uniform(
                height_key,
                minval=-height_noise_limit,
                maxval=height_noise_limit,
            )
            qpos = self.base_data.qpos.at[self.qpos_indices].set(joint_q).at[2].add(height_noise)
            qvel = jp.zeros_like(self.base_data.qvel)
            ctrl = self.base_data.ctrl.at[self.actuator_ids].set(initial_target)
            data = mjx.forward(self.mjx_model, self.base_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl))
            foot_bottom = data.geom_xpos[self.foot_geom_ids, 2] - self.foot_radii
            qpos = data.qpos.at[2].add(self.config.reset_foot_clearance - jp.min(foot_bottom))
            data = mjx.forward(self.mjx_model, data.replace(qpos=qpos))

            previous_residual = jp.zeros(self.config.action_size)
            previous_student_action = self._target_to_student_action(initial_target)
            if self.config.disturbance_enabled:
                push_step = jax.random.randint(
                    push_step_key,
                    (),
                    self.config.push_step_min,
                    self.config.push_step_max + 1,
                )
                push_angle = jax.random.uniform(push_angle_key, (), minval=-jp.pi, maxval=jp.pi)
                push_magnitude = jax.random.uniform(
                    push_magnitude_key, (), minval=0.7, maxval=1.0
                )
                push_velocity = push_magnitude * jp.stack(
                    (
                        self.config.push_velocity_x * jp.cos(push_angle),
                        self.config.push_velocity_y * jp.sin(push_angle),
                    )
                )
                motor_strength = jax.random.uniform(
                    motor_key,
                    (),
                    minval=self.config.motor_strength_min,
                    maxval=self.config.motor_strength_max,
                )
                control_delay = jax.random.bernoulli(
                    delay_key, self.config.control_delay_probability
                ).astype(jp.float32)
            else:
                push_step = jp.array(self.config.max_episode_steps + 1, dtype=jp.int32)
                push_velocity = jp.zeros(2)
                motor_strength = jp.array(1.0)
                control_delay = jp.array(0.0)
            last_push = jp.zeros(2)
            smoothed_world_velocity = jp.zeros(2)
            recovery_streak = jp.array(0, dtype=jp.int32)
            student_history = jp.zeros(self.config.student_observation_size)
            student_obs = self._update_student_history(
                student_history,
                self._student_frame(data, previous_student_action),
            )
            gait_blend = jp.array(0.0)
            student_policy_obs = self._student_policy_obs(
                student_obs, phase, gait_blend
            )
            info = {
                "rng": next_rng,
                "step_count": jp.array(0, dtype=jp.int32),
                "previous_student_action": previous_student_action,
                "student_history": student_obs,
                "student_obs": student_obs,
                "student_policy_obs": student_policy_obs,
                "phase": phase,
                "gait_blend": gait_blend,
                "last_foot_pos": data.site_xpos[self.foot_site_ids],
                "target_ctrl": initial_target,
                "previous_target_ctrl": initial_target,
                "student_action": previous_student_action,
                "push_step": push_step,
                "push_velocity": push_velocity,
                "last_push": last_push,
                "motor_strength": motor_strength,
                "control_delay": control_delay,
                "smoothed_world_velocity": smoothed_world_velocity,
                "recovery_streak": recovery_streak,
            }
            if self.role != "student":
                actual_contacts = self._foot_contacts(data)
                teacher_obs = self._teacher_obs(
                    data,
                    student_obs,
                    phase,
                    initial_target,
                    actual_contacts,
                    previous_residual,
                    gait_blend,
                    last_push,
                    motor_strength,
                    control_delay,
                )
                info.update(
                    {
                        "previous_residual": previous_residual,
                        "teacher_obs": teacher_obs,
                        "ik_target": initial_target,
                    }
                )
            obs = (
                info["teacher_obs"]
                if self.role == "teacher"
                else info["student_policy_obs"]
            )
            return State(data, obs, jp.array(0.0), jp.array(0.0), self._empty_metrics(), info)

        def step(self, state, action):
            action = jp.clip(action, -1.0, 1.0)
            phase = state.info["phase"]
            gait_blend = state.info["gait_blend"]
            if self.role == "student":
                student_action = action
                target_ctrl = self.stand_q + self.student_action_scale * student_action
                residual_action = jp.zeros(self.config.action_size)
            else:
                ik_target = self._blended_ik_target(phase, gait_blend)
                if self.role == "teacher":
                    residual_action = self._filter_teacher_residual(
                        state.info["previous_residual"], action
                    )
                    target_ctrl = ik_target + self.residual_scale * residual_action
                    student_action = self._target_to_student_action(target_ctrl)
                else:
                    student_action = action
                    target_ctrl = self.stand_q + self.student_action_scale * student_action
                    residual_action = jp.clip(
                        (target_ctrl - ik_target) / jp.maximum(self.residual_scale, 1e-6),
                        -1.0,
                        1.0,
                    )
            target_ctrl = jp.clip(target_ctrl, self.ctrl_low, self.ctrl_high)

            joint_position = state.pipeline_state.qpos[self.qpos_indices]
            strength_target = joint_position + state.info["motor_strength"] * (
                target_ctrl - joint_position
            )
            delayed_target = jp.where(
                state.info["control_delay"] > 0.5,
                state.info["previous_target_ctrl"],
                strength_target,
            )
            applied_target = jp.clip(delayed_target, self.ctrl_low, self.ctrl_high)
            push_applied = state.info["step_count"] == state.info["push_step"]
            applied_push = jp.where(push_applied, state.info["push_velocity"], jp.zeros(2))

            old_pos = state.pipeline_state.xpos[self.torso_body_id]
            data = state.pipeline_state.replace(
                qvel=state.pipeline_state.qvel.at[self.root_linear_dof_indices[:2]].add(
                    applied_push
                ),
                ctrl=state.pipeline_state.ctrl.at[self.actuator_ids].set(applied_target),
            )

            def physics_step(carry, _):
                return mjx.step(self.mjx_model, carry), None

            data = jax.lax.scan(
                physics_step,
                data,
                (),
                length=max(1, self.config.action_repeat),
            )[0]
            step_count = state.info["step_count"] + 1

            world_velocity = (data.xpos[self.torso_body_id] - old_pos) / jp.maximum(self.dt, 1e-9)
            torso_mat = data.xmat[self.torso_body_id]
            body_velocity = torso_mat.T @ world_velocity
            body_angular_velocity = torso_mat.T @ data.cvel[self.torso_body_id, :3]
            upright = torso_mat[2, 2]
            torso_height = data.xpos[self.torso_body_id, 2]
            actual_contacts = self._foot_contacts(data)
            disk_contacts = self._disk_contact_count(data)
            foot_pos = data.site_xpos[self.foot_site_ids]
            foot_velocity = (foot_pos[:, :2] - state.info["last_foot_pos"][:, :2]) / jp.maximum(
                self.dt, 1e-9
            )
            foot_slip = jp.mean(jp.sum(jp.square(foot_velocity), axis=1) * actual_contacts)
            command_magnitude = (
                jp.linalg.norm(self.command[:2]) + jp.abs(self.command[2])
            )
            command_active = command_magnitude > self.config.student_command_deadzone
            phase_increment = jp.where(command_active, self.frequency * self.dt, 0.0)
            next_phase = jp.mod(phase + phase_increment, 1.0)
            blend_increment = 1.0 / max(1, self.config.startup_blend_steps)
            blend_direction = jp.where(command_active, 1.0, -1.0)
            next_gait_blend = jp.clip(
                gait_blend + blend_direction * blend_increment, 0.0, 1.0
            )
            if self.role == "student":
                residual_delta = jp.zeros(self.config.action_size)
                contact_mismatch = jp.array(0.0)
            else:
                next_ik_target = self._blended_ik_target(next_phase, next_gait_blend)
                desired_contacts = self._desired_contacts(next_phase)
                residual_delta = residual_action - state.info["previous_residual"]
                contact_mismatch = next_gait_blend * jp.mean(
                    jp.abs(desired_contacts - actual_contacts)
                )

            failed_bool = (torso_height < self.config.min_torso_height) | (
                upright < self.config.terminate_upright
            )
            timeout_bool = step_count >= self.config.max_episode_steps
            done = jp.maximum(state.done, (failed_bool | timeout_bool).astype(jp.float32))
            failed = failed_bool.astype(jp.float32)
            steps_since_push = step_count - state.info["push_step"]
            post_push = (
                (steps_since_push >= 0)
                & (steps_since_push < self.config.recovery_window_steps)
            ).astype(jp.float32)

            forward_velocity = world_velocity[0]
            velocity_error = forward_velocity - self.config.command_vx
            smoothed_world_velocity = state.info["smoothed_world_velocity"] + (
                self.config.recovery_velocity_ema_alpha
                * (world_velocity[:2] - state.info["smoothed_world_velocity"])
            )
            smoothed_velocity_error = jp.abs(
                smoothed_world_velocity[0] - self.config.command_vx
            )
            recovery_condition = (
                (smoothed_velocity_error <= self.config.recovery_forward_tolerance)
                & (
                    jp.abs(smoothed_world_velocity[1])
                    <= self.config.recovery_lateral_tolerance
                )
                & (upright >= 0.80)
            )
            recovery_streak = jp.where(
                (post_push > 0.5) & recovery_condition,
                state.info["recovery_streak"] + 1,
                0,
            ).astype(jp.int32)
            recovery_ready = recovery_streak >= self.config.recovery_required_steps
            reward_terms = {
                "velocity": self.config.reward_velocity
                * jp.exp(
                    -(velocity_error * velocity_error + world_velocity[1] ** 2)
                    / self.config.velocity_sigma
                ),
                "progress": self.config.reward_progress * jp.clip(forward_velocity, -0.3, 0.3),
                "yaw": self.config.reward_yaw
                * jp.exp(-(body_angular_velocity[2] ** 2) / self.config.yaw_sigma),
                "alive": self.config.reward_alive,
                "vertical_velocity": -self.config.penalty_vertical_velocity * world_velocity[2] ** 2,
                "roll_pitch_rate": -self.config.penalty_roll_pitch_rate
                * jp.sum(jp.square(body_angular_velocity[:2])),
                "orientation": -self.config.penalty_orientation * (1.0 - upright * upright),
                "joint_velocity": -self.config.penalty_joint_velocity
                * jp.mean(jp.square(data.qvel[self.dof_indices])),
                "foot_slip": -self.config.penalty_foot_slip * foot_slip,
                "disk_contact": -self.config.penalty_disk_contact * disk_contacts,
                "residual": -self.config.penalty_residual * jp.mean(jp.square(residual_action)),
                "residual_rate": -self.config.penalty_residual_rate
                * jp.mean(jp.square(residual_delta)),
                "contact_mismatch": -self.config.penalty_contact_mismatch * contact_mismatch,
                "termination": -self.config.penalty_termination * failed,
            }
            reward = sum(reward_terms.values())

            student_history = self._update_student_history(
                state.info["student_history"],
                self._student_frame(data, student_action),
            )
            student_policy_obs = self._student_policy_obs(
                student_history, next_phase, next_gait_blend
            )
            info = {
                **state.info,
                "step_count": step_count,
                "previous_student_action": student_action,
                "student_history": student_history,
                "student_obs": student_history,
                "student_policy_obs": student_policy_obs,
                "phase": next_phase,
                "gait_blend": next_gait_blend,
                "last_foot_pos": foot_pos,
                "target_ctrl": applied_target,
                "previous_target_ctrl": strength_target,
                "student_action": student_action,
                "last_push": applied_push,
                "smoothed_world_velocity": smoothed_world_velocity,
                "recovery_streak": recovery_streak,
            }
            if self.role != "student":
                teacher_obs = self._teacher_obs(
                    data,
                    student_history,
                    next_phase,
                    next_ik_target,
                    actual_contacts,
                    residual_action,
                    next_gait_blend,
                    applied_push,
                    state.info["motor_strength"],
                    state.info["control_delay"],
                )
                info.update(
                    {
                        "previous_residual": residual_action,
                        "teacher_obs": teacher_obs,
                        "ik_target": next_ik_target,
                    }
                )
            metrics = {
                "reward": reward,
                **{f"reward_{name}": value for name, value in reward_terms.items()},
                "velocity_x": forward_velocity,
                "velocity_y": world_velocity[1],
                "abs_velocity_y": jp.abs(world_velocity[1]),
                "body_velocity_x": body_velocity[0],
                "body_velocity_y": body_velocity[1],
                "yaw_rate": body_angular_velocity[2],
                "abs_yaw_rate": jp.abs(body_angular_velocity[2]),
                "velocity_error": jp.abs(velocity_error),
                "smoothed_velocity_error": smoothed_velocity_error,
                "roll_pitch_rate_rms": jp.sqrt(jp.mean(jp.square(body_angular_velocity[:2]))),
                "upright": upright,
                "torso_height": torso_height,
                "foot_slip": foot_slip,
                "disk_contact_count": disk_contacts,
                "contact_mismatch": contact_mismatch,
                "residual_rms": jp.sqrt(jp.mean(jp.square(residual_action))),
                "student_action_rms": jp.sqrt(jp.mean(jp.square(student_action))),
                "push_applied": push_applied.astype(jp.float32),
                "post_push": post_push,
                "recovery_ready": recovery_ready.astype(jp.float32),
                "failed": failed,
                "timeout": timeout_bool.astype(jp.float32),
            }
            obs = (
                info["teacher_obs"]
                if self.role == "teacher"
                else info["student_policy_obs"]
            )
            return State(data, obs, reward, done, metrics, info)

        def teacher_action_to_student_action(self, state, residual_action):
            residual_action = self._filter_teacher_residual(
                state.info["previous_residual"], residual_action
            )
            target = jp.clip(
                self._blended_ik_target(
                    state.info["phase"], state.info["gait_blend"]
                )
                + self.residual_scale * jp.clip(residual_action, -1.0, 1.0),
                self.ctrl_low,
                self.ctrl_high,
            )
            return self._target_to_student_action(target)

        def _filter_teacher_residual(self, previous_residual, residual_command):
            command = jp.clip(residual_command, -1.0, 1.0)
            return previous_residual + self.config.residual_filter_alpha * (
                command - previous_residual
            )

        def _ik_target(self, phase):
            return interpolate_reference_jax(jp, self.ik_table, phase)

        def _blended_ik_target(self, phase, blend):
            return self.stand_q + blend * (self._ik_target(phase) - self.stand_q)

        def _target_to_student_action(self, target):
            return jp.clip((target - self.stand_q) / self.student_action_scale, -1.0, 1.0)

        def _desired_contacts(self, phase):
            phases = jp.mod(phase + self.phase_offsets, 1.0)
            return (phases < self.duty).astype(jp.float32)

        def _student_frame(self, data, previous_student_action):
            torso_mat = data.xmat[self.torso_body_id]
            body_angular_velocity = torso_mat.T @ data.cvel[self.torso_body_id, :3]
            body_linear_velocity = torso_mat.T @ data.cvel[self.torso_body_id, 3:6]
            projected_gravity = torso_mat.T @ jp.asarray((0.0, 0.0, -1.0))
            return jp.concatenate(
                (
                    body_angular_velocity,
                    projected_gravity,
                    body_linear_velocity,
                    data.qpos[self.qpos_indices] - self.stand_q,
                    data.qvel[self.dof_indices],
                    previous_student_action,
                    self.command,
                )
            )

        def _update_student_history(self, history, frame):
            size = self.config.student_frame_size
            return jp.roll(history, size).at[:size].set(frame)

        def _student_policy_obs(self, student_history, phase, gait_blend):
            if not self.config.student_previous_action_input:
                frames = student_history.reshape(
                    self.config.observation_history, self.config.student_frame_size
                )
                frames = jp.concatenate((frames[:, :33], frames[:, 45:]), axis=1)
                student_history = frames.reshape(-1)
            if not self.config.student_phase_conditioned:
                return student_history
            phase_angle = 2.0 * jp.pi * phase
            internal_state = jp.stack(
                (jp.sin(phase_angle), jp.cos(phase_angle), gait_blend)
            )
            return jp.concatenate((student_history, internal_state))

        def _teacher_obs(
            self,
            data,
            student_obs,
            phase,
            ik_target,
            actual_contacts,
            previous_residual,
            gait_blend=0.0,
            last_push=None,
            motor_strength=1.0,
            control_delay=0.0,
        ):
            if last_push is None:
                last_push = jp.zeros(2)
            phase_angle = 2.0 * jp.pi * phase
            privileged = jp.concatenate(
                (
                    jp.stack((jp.sin(phase_angle), jp.cos(phase_angle))),
                    jp.atleast_1d(gait_blend),
                    self._desired_contacts(phase),
                    actual_contacts,
                    data.qpos[self.qpos_indices] - ik_target,
                    previous_residual,
                    last_push
                    / jp.asarray(
                        (
                            max(self.config.push_velocity_x, 1e-6),
                            max(self.config.push_velocity_y, 1e-6),
                        )
                    ),
                    jp.atleast_1d(motor_strength - 1.0),
                    jp.atleast_1d(control_delay),
                )
            )
            return jp.concatenate((student_obs, privileged))

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
            pair = ((geom1 == self.torso_geom_id) & (geom2 == self.floor_geom_id)) | (
                (geom2 == self.torso_geom_id) & (geom1 == self.floor_geom_id)
            )
            return jp.sum(valid & pair).astype(jp.float32)

        def _empty_metrics(self):
            zero = jp.array(0.0)
            names = (
                "reward",
                "velocity_x",
                "velocity_y",
                "abs_velocity_y",
                "body_velocity_x",
                "body_velocity_y",
                "yaw_rate",
                "abs_yaw_rate",
                "velocity_error",
                "smoothed_velocity_error",
                "roll_pitch_rate_rms",
                "upright",
                "torso_height",
                "foot_slip",
                "disk_contact_count",
                "contact_mismatch",
                "residual_rms",
                "student_action_rms",
                "push_applied",
                "post_push",
                "recovery_ready",
                "failed",
                "timeout",
            )
            reward_names = (
                "velocity",
                "progress",
                "yaw",
                "alive",
                "vertical_velocity",
                "roll_pitch_rate",
                "orientation",
                "joint_velocity",
                "foot_slip",
                "disk_contact",
                "residual",
                "residual_rate",
                "contact_mismatch",
                "termination",
            )
            metrics = {name: zero for name in names}
            metrics.update({f"reward_{name}": zero for name in reward_names})
            return metrics

    return ForwardTeacherStudentEnv()


__all__ = ["DEFAULT_XML", "make_forward_teacher_student_env"]
