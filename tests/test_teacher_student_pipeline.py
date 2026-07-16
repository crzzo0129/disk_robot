def test_pipeline_entry_imports_without_jax_or_brax():
    import sys

    jax_before = sys.modules.get("jax")
    brax_before = sys.modules.get("brax")
    from scripts import train_forward_teacher_student

    assert sys.modules.get("jax") is jax_before
    assert sys.modules.get("brax") is brax_before
    assert train_forward_teacher_student.parse_args([]).teacher_steps == 5_000_000


def test_pipeline_smoke_and_stand_defaults_are_explicit():
    from scripts.train_forward_teacher_student import parse_args

    args = parse_args(["--smoke", "--command-vx", "0.08"])

    assert args.smoke
    assert args.command_vx == 0.08
    assert args.ik_speed_mode == "command"
    assert args.ik_frequency == 0.8
    assert args.ik_stride == 0.04
    assert args.kp == 10.0
    assert args.kd == 0.4
    assert args.startup_steps == 25
    assert args.teacher_batch_size == 256
    assert args.teacher_minibatches == 32
    assert args.residual_scale_multiplier == 1.0
    assert args.residual_filter_alpha == 0.15
    assert args.penalty_residual == 0.20
    assert args.penalty_residual_rate == 0.05
    assert args.teacher_zero_policy_init
    assert args.teacher_selection_mode == "improve"
    assert not args.teacher_disturbances
    assert args.push_velocity_x == 0.50
    assert args.push_velocity_y == 0.40
    assert args.recovery_velocity_ema_alpha == 0.10
    assert args.recovery_forward_tolerance == 0.04
    assert args.recovery_lateral_tolerance == 0.04
    assert args.recovery_required_steps == 4
    assert not args.allow_ik_baseline_teacher
    assert not args.teacher_only


def test_pipeline_defaults_match_the_stable_ik_baseline():
    from disk_robot.teacher_student_config import ForwardTeacherStudentConfig
    from scripts.train_forward_teacher_student import (
        _resolve_acceptance_thresholds,
        _resolve_reference_spec,
        parse_args,
    )

    args = parse_args([])
    reference, source = _resolve_reference_spec(args)
    _resolve_acceptance_thresholds(args)

    assert args.command_vx == 0.08
    assert args.min_accepted_teacher_vx == 0.06
    assert args.min_accepted_vx == 0.06
    assert args.max_accepted_teacher_velocity_error == 0.03
    assert args.max_accepted_teacher_roll_pitch_rate == 0.50
    assert args.max_accepted_velocity_error == 0.03
    assert args.max_accepted_roll_pitch_rate == 0.60
    assert reference.frequency == 1.2
    assert 0.07 < reference.stride_length < 0.08
    assert source["mode"] == "command"
    assert args.teacher_learning_rate == 1e-4
    assert args.teacher_entropy_cost == 1e-3
    assert ForwardTeacherStudentConfig().velocity_sigma == 0.01
    assert ForwardTeacherStudentConfig().residual_filter_alpha == 0.15
    assert ForwardTeacherStudentConfig().penalty_residual == 0.20


def test_zero_policy_initializer_supports_brax_parameter_layouts():
    import numpy as np

    from disk_robot_mjx.pipeline import _zero_policy_output_layer

    old_layout = [
        (np.ones((3, 4)), np.ones(4)),
        (np.ones((4, 2)), np.ones(2)),
    ]
    zeroed_old = _zero_policy_output_layer(old_layout, np.zeros_like)
    assert np.all(zeroed_old[0][0] == 1.0)
    assert np.all(zeroed_old[-1][0] == 0.0)
    assert np.all(zeroed_old[-1][1] == 0.0)

    flax_layout = {
        "params": {
            "hidden_0": {"kernel": np.ones((3, 4)), "bias": np.ones(4)},
            "hidden_1": {"kernel": np.ones((4, 2)), "bias": np.ones(2)},
        }
    }
    zeroed_flax = _zero_policy_output_layer(flax_layout, np.zeros_like)
    assert np.all(zeroed_flax["params"]["hidden_0"]["kernel"] == 1.0)
    assert np.all(zeroed_flax["params"]["hidden_1"]["kernel"] == 0.0)
    assert np.all(zeroed_flax["params"]["hidden_1"]["bias"] == 0.0)


def test_t1b_preserve_selection_requires_all_baseline_tolerances():
    from scripts.train_forward_teacher_student import (
        _ppo_preserves_baseline,
        _should_select_ppo,
    )

    baseline = {
        "mean_velocity_x": 0.084,
        "failure_rate": 0.0,
        "mean_roll_pitch_rate_rms": 0.22,
        "mean_abs_velocity_y": 0.01,
        "mean_abs_yaw_rate": 0.10,
    }
    ppo = {
        "mean_velocity_x": 0.075,
        "failure_rate": 0.01,
        "mean_roll_pitch_rate_rms": 0.31,
        "mean_abs_velocity_y": 0.019,
        "mean_abs_yaw_rate": 0.149,
    }
    assert _ppo_preserves_baseline(ppo, baseline)
    assert _should_select_ppo("preserve", 0.99, 1.0, True)
    assert not _should_select_ppo("improve", 0.99, 1.0, True)

    ppo["mean_abs_yaw_rate"] = 0.151
    assert not _ppo_preserves_baseline(ppo, baseline)
    assert not _should_select_ppo("preserve", 1.01, 1.0, False)


def test_t1b_preserve_mode_cannot_start_student_distillation():
    import pytest

    from scripts.train_forward_teacher_student import main

    with pytest.raises(SystemExit, match="preserve requires --teacher-only"):
        main(["--teacher-selection-mode", "preserve"])


def test_preserve_acceptance_uses_relative_gate_not_absolute_lateral_cap():
    from scripts.train_forward_teacher_student import _teacher_gate_acceptance

    assert _teacher_gate_acceptance("preserve", "ppo", False, True, False)
    assert not _teacher_gate_acceptance("preserve", "ppo", True, False, False)
    assert not _teacher_gate_acceptance("preserve", "ik_baseline", True, True, False)


def test_brax_step_plan_reports_the_observed_rounding_quantum():
    from scripts.train_forward_teacher_student import _estimate_brax_timesteps

    inflated = _estimate_brax_timesteps(200_000, 6, 256, 32, 20)
    fitted = _estimate_brax_timesteps(200_000, 6, 256, 8, 20)

    assert inflated["step_quantum"] == 163_840
    assert inflated["estimated_effective_steps"] == 819_200
    assert fitted["estimated_effective_steps"] == 204_800


def test_t2_robust_gate_requires_nominal_preservation_and_disturbed_gain():
    from scripts.train_forward_teacher_student import (
        _ppo_improves_disturbed_baseline,
        _teacher_gate_acceptance,
    )

    baseline = {
        "reward_per_step": 1.0,
        "failure_rate": 0.20,
        "mean_velocity_error": 0.02,
        "mean_roll_pitch_rate_rms": 0.30,
        "mean_abs_velocity_y": 0.04,
        "mean_abs_yaw_rate": 0.20,
        "mean_post_push_velocity_error": 0.20,
        "mean_recovery_time": 1.0,
        "mean_disk_contacts": 0.10,
        "mean_forward_distance": 0.50,
    }
    ppo = {
        **baseline,
        "reward_per_step": 1.1,
        "failure_rate": 0.10,
        "mean_post_push_velocity_error": 0.15,
        "mean_recovery_time": 0.8,
        "mean_disk_contacts": 0.05,
        "mean_forward_distance": 0.55,
    }

    assert _ppo_improves_disturbed_baseline(ppo, baseline)
    assert _teacher_gate_acceptance("robust", "ppo", False, True, True)
    assert not _teacher_gate_acceptance("robust", "ppo", True, False, True)

    ppo["mean_forward_distance"] = 0.47
    assert not _ppo_improves_disturbed_baseline(ppo, baseline)


def test_t2_robust_mode_requires_disturbances():
    import pytest

    from scripts.train_forward_teacher_student import main

    with pytest.raises(SystemExit, match="robust requires --teacher-disturbances"):
        main(["--teacher-selection-mode", "robust", "--teacher-only"])


def test_pipeline_manual_ik_mode_keeps_explicit_parameters():
    from scripts.train_forward_teacher_student import _resolve_reference_spec, parse_args

    args = parse_args(
        ["--ik-speed-mode", "manual", "--ik-frequency", "0.9", "--ik-stride", "0.05"]
    )
    reference, source = _resolve_reference_spec(args)

    assert reference.frequency == 0.9
    assert reference.stride_length == 0.05
    assert source["mode"] == "manual"


def test_teacher_summary_separates_net_and_instantaneous_velocity_error():
    from scripts.train_forward_teacher_student import _teacher_eval_summary

    summary = _teacher_eval_summary(
        {
            "eval/avg_episode_length": 500.0,
            "eval/episode_velocity_x": 0.0837 * 500.0,
            "eval/episode_velocity_error": 0.0794 * 500.0,
        },
        command_vx=0.08,
    )

    assert abs(summary["mean_velocity_error"] - 0.0037) < 1e-8
    assert abs(summary["mean_instantaneous_velocity_error"] - 0.0794) < 1e-8


def test_terminal_evaluation_summary_is_compact(capsys):
    from scripts.train_forward_teacher_student import _print_evaluation_summary

    _print_evaluation_summary(
        "teacher_result",
        {
            "reward_per_step": 1.5,
            "mean_velocity_x": 0.08,
            "mean_forward_distance": 0.8,
            "failure_rate": 0.0,
            "mean_roll_pitch_rate_rms": 0.2,
            "mean_abs_velocity_y": 0.03,
            "mean_abs_yaw_rate": 0.1,
            "mean_disk_contacts": 0.0,
            "mean_post_push_velocity_error": 0.04,
            "mean_recovery_time": 0.4,
            "push_coverage": 1.0,
        },
        "disturbed",
    )
    output = capsys.readouterr().out

    assert "stage=teacher_result mode=disturbed" in output
    assert "post_error=0.0400" in output
    assert "recovery_s=0.400" in output
    assert "{" not in output


def test_terminal_teacher_comparison_shows_readable_deltas(capsys):
    from scripts.train_forward_teacher_student import _print_teacher_comparison

    baseline = {
        "mean_velocity_x": 0.08,
        "failure_rate": 0.1,
        "mean_roll_pitch_rate_rms": 0.3,
        "mean_post_push_velocity_error": 0.08,
        "mean_recovery_time": 0.5,
        "mean_forward_distance": 0.7,
        "mean_disk_contacts": 0.02,
    }
    ppo = {
        **baseline,
        "mean_velocity_x": 0.081,
        "failure_rate": 0.05,
        "mean_recovery_time": 0.4,
    }

    _print_teacher_comparison("disturbed", ppo, baseline, score_gain=0.03)
    output = capsys.readouterr().out

    assert "score_gain=+0.0300" in output
    assert "delta_failure=-0.050" in output
    assert "delta_recovery_s=-0.100" in output
    assert "{" not in output


def test_teacher_env_has_privileged_residual_and_student_modes():
    source = open("disk_robot_mjx/teacher_student_env.py", encoding="utf-8").read()

    assert 'role not in ("teacher", "dagger", "student")' in source
    assert 'state.info["phase"]' in source
    assert 'state.info["teacher_obs"]' not in source
    assert "teacher_action_to_student_action" in source
    assert "target_ctrl = self.stand_q + self.student_action_scale * student_action" in source
    assert "forward_velocity = world_velocity[0]" in source
    assert '"body_velocity_x": body_velocity[0]' in source
    assert 'if self.role == "student"' in source
    assert "self._blended_ik_target" in source
    assert 'state.info["push_velocity"]' in source
    assert 'state.info["motor_strength"]' in source
    assert 'state.info["control_delay"]' in source
    assert 'state.info["smoothed_world_velocity"]' in source
    assert 'state.info["recovery_streak"]' in source
    assert '"student_policy_obs"' in source
    assert "student_policy_observation_size" in source
    assert '"mean_post_push_velocity_error"' in open(
        "scripts/train_forward_teacher_student.py", encoding="utf-8"
    ).read()


def test_pipeline_produces_student_policy_and_evaluation_contract():
    source = open("scripts/train_forward_teacher_student.py", encoding="utf-8").read()

    assert 'student_path = args.out / "student_policy.npz"' in source
    assert 'args.out / "evaluation.json"' in source
    assert '"stand_source": "xml:keyframe:stand"' in source
    assert "_collect_dagger_dataset" in source
    assert "policy_params_fn" in source
    assert 'teacher_dir / "params_ppo_best"' in source
    assert 'teacher_dir / "ik_baseline_evaluation.json"' in source
    assert 'teacher_dir / "selection.json"' in source
    assert 'selected_source = "ik_baseline"' in source
    assert '"ppo_teacher_did_not_outperform_ik_baseline"' in source
    assert 'mode=teacher_only' in source
    assert '"mean_forward_distance"' in source
    assert 'make_forward_teacher_student_env(\n        "dagger"' in source
    assert 'make_forward_teacher_student_env(\n        "student"' in source


def test_standalone_student_evaluation_does_not_build_ik_reference():
    source = open("scripts/evaluate_forward_student.py", encoding="utf-8").read()

    assert "build_ik_reference" not in source
    assert 'make_forward_teacher_student_env(\n        "student"' in source
