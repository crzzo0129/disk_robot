import json

import numpy as np
import pytest


def test_t4_entry_imports_without_jax_or_brax():
    import sys

    jax_before = sys.modules.get("jax")
    brax_before = sys.modules.get("brax")
    from scripts import dagger_forward_student

    assert sys.modules.get("jax") is jax_before
    assert sys.modules.get("brax") is brax_before
    args = dagger_forward_student.parse_args(
        ["--teacher-run", "teacher", "--bc-run", "bc"]
    )
    assert args.dagger_rounds == 2
    assert args.dagger_samples == 65_536
    assert args.student_learning_rate == 1e-4
    assert args.teacher_rollout_blend_start == 0.5
    assert args.teacher_rollout_blend_end == 0.2


def test_t4_loads_the_saved_bc_policy_and_dataset(tmp_path):
    from scripts.dagger_forward_student import _load_bc_run

    policy = tmp_path / "student_policy_bc.npz"
    np.savez(
        policy,
        obs_mean=np.zeros(2, dtype=np.float32),
        obs_std=np.ones(2, dtype=np.float32),
        weight_0=np.ones((2, 1), dtype=np.float32),
        bias_0=np.zeros(1, dtype=np.float32),
    )
    policy.with_suffix(".json").write_text(
        json.dumps({"stage": "T3_BC"}), encoding="utf-8"
    )
    np.savez(
        tmp_path / "student_bc_dataset.npz",
        observations=np.ones((3, 2), dtype=np.float32),
        actions=np.ones((3, 1), dtype=np.float32),
    )
    (tmp_path / "evaluation.json").write_text(
        json.dumps({"accepted": False}), encoding="utf-8"
    )

    _, _, artifact, observations, actions, evaluation = _load_bc_run(tmp_path)

    assert artifact.metadata["stage"] == "T3_BC"
    assert observations.shape == (3, 2)
    assert actions.shape == (3, 1)
    assert not evaluation["accepted"]


def test_t4_requires_the_t3_saved_dataset(tmp_path):
    from scripts.dagger_forward_student import _load_bc_run

    np.savez(
        tmp_path / "student_policy_bc.npz",
        obs_mean=np.zeros(1),
        obs_std=np.ones(1),
        weight_0=np.ones((1, 1)),
        bias_0=np.zeros(1),
    )
    with pytest.raises(SystemExit, match="--save-dataset"):
        _load_bc_run(tmp_path)


def test_t4_score_prefers_stable_forward_student():
    from scripts.dagger_forward_student import _student_score

    weak = {
        "reward_per_step": 0.0,
        "mean_velocity_x": 0.03,
        "mean_velocity_error": 0.05,
        "failure_rate": 0.02,
        "mean_roll_pitch_rate_rms": 1.0,
        "mean_post_push_velocity_error": 0.10,
        "mean_recovery_time": 1.4,
        "mean_disk_contacts": 0.0,
    }
    strong = {
        **weak,
        "reward_per_step": 1.0,
        "mean_velocity_x": 0.08,
        "mean_velocity_error": 0.0,
        "failure_rate": 0.0,
        "mean_roll_pitch_rate_rms": 0.25,
        "mean_post_push_velocity_error": 0.04,
        "mean_recovery_time": 0.5,
    }

    assert _student_score(strong, strong) > _student_score(weak, weak)


def test_t4_teacher_rollout_blend_anneals_across_rounds():
    from scripts.dagger_forward_student import _teacher_rollout_blend

    assert _teacher_rollout_blend(1, 3, 0.5, 0.1) == pytest.approx(0.5)
    assert _teacher_rollout_blend(2, 3, 0.5, 0.1) == pytest.approx(0.3)
    assert _teacher_rollout_blend(3, 3, 0.5, 0.1) == pytest.approx(0.1)


def test_t4_fallback_selection_does_not_prefer_stable_standing_over_walking():
    from scripts.dagger_forward_student import _student_score

    walking = {
        "reward_per_step": 0.0,
        "mean_velocity_error": 0.045,
        "failure_rate": 0.0,
        "mean_roll_pitch_rate_rms": 1.05,
        "mean_post_push_velocity_error": 0.09,
        "mean_recovery_time": 1.25,
        "mean_disk_contacts": 0.0,
    }
    standing = {
        **walking,
        "reward_per_step": 0.13,
        "mean_velocity_error": 0.066,
        "mean_roll_pitch_rate_rms": 0.97,
        "mean_recovery_time": 1.16,
    }

    assert _student_score(walking, walking) > _student_score(standing, standing)


def test_t4_uses_student_rollouts_and_frozen_teacher_labels():
    source = open("scripts/dagger_forward_student.py", encoding="utf-8").read()

    assert "_collect_dagger_dataset" in source
    assert "teacher_rollout_blend" in source
    assert 'make_forward_teacher_student_env(\n        "dagger"' in source
    assert "ppo.train(" not in source
    assert '"stage": "T4_DAGGER"' in source
    assert '"student_dagger_retention"' in source
    assert "student_policy_dagger_round_" in source
    assert 'args.out / "student_policy_dagger.npz"' in source
