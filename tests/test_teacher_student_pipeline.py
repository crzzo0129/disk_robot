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
    assert args.ik_frequency == 0.8
    assert args.ik_stride == 0.04
    assert args.kp == 7.5
    assert args.kd == 0.25
    assert args.startup_steps == 25
    assert args.teacher_batch_size == 256
    assert args.teacher_minibatches == 32


def test_pipeline_defaults_match_the_stable_ik_baseline():
    from disk_robot.teacher_student_config import ForwardTeacherStudentConfig
    from scripts.train_forward_teacher_student import parse_args

    args = parse_args([])

    assert args.command_vx == 0.03
    assert args.min_accepted_teacher_vx == 0.02
    assert args.min_accepted_vx == 0.02
    assert args.teacher_learning_rate == 1e-4
    assert args.teacher_entropy_cost == 1e-3
    assert ForwardTeacherStudentConfig().velocity_sigma == 0.0004


def test_teacher_env_has_privileged_residual_and_student_modes():
    source = open("disk_robot_mjx/teacher_student_env.py", encoding="utf-8").read()

    assert 'role not in ("teacher", "dagger", "student")' in source
    assert 'state.info["phase"]' in source
    assert 'state.info["teacher_obs"]' not in source
    assert "teacher_action_to_student_action" in source
    assert "target_ctrl = self.stand_q + self.student_action_scale * student_action" in source
    assert 'if self.role == "student"' in source
    assert "self._blended_ik_target" in source


def test_pipeline_produces_student_policy_and_evaluation_contract():
    source = open("scripts/train_forward_teacher_student.py", encoding="utf-8").read()

    assert 'student_path = args.out / "student_policy.npz"' in source
    assert 'args.out / "evaluation.json"' in source
    assert '"stand_source": "xml:keyframe:stand"' in source
    assert "_collect_dagger_dataset" in source
    assert "policy_params_fn" in source
    assert 'teacher_dir / "params_best"' in source
    assert 'teacher_dir / "ik_baseline_evaluation.json"' in source
    assert 'make_forward_teacher_student_env(\n        "dagger"' in source
    assert 'make_forward_teacher_student_env(\n        "student"' in source


def test_standalone_student_evaluation_does_not_build_ik_reference():
    source = open("scripts/evaluate_forward_student.py", encoding="utf-8").read()

    assert "build_ik_reference" not in source
    assert 'make_forward_teacher_student_env(\n        "student"' in source
