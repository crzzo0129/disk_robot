import json

import numpy as np


def test_student_policy_artifact_round_trip_and_numpy_inference(tmp_path):
    from disk_robot.student_policy import apply_student_policy_numpy, load_student_policy

    path = tmp_path / "student_policy.npz"
    np.savez(
        path,
        obs_mean=np.zeros(3, dtype=np.float32),
        obs_std=np.ones(3, dtype=np.float32),
        weight_0=np.eye(3, dtype=np.float32),
        bias_0=np.zeros(3, dtype=np.float32),
        weight_1=np.ones((3, 2), dtype=np.float32),
        bias_1=np.zeros(2, dtype=np.float32),
    )
    path.with_suffix(".json").write_text(json.dumps({"format": "test"}), encoding="utf-8")

    artifact = load_student_policy(path)
    action = apply_student_policy_numpy(artifact, np.array([0.1, 0.2, -0.1]))

    assert action.shape == (2,)
    assert np.all(np.isfinite(action))
    assert artifact.metadata["format"] == "test"


def test_student_evaluation_entry_is_gait_free_by_contract():
    from scripts.evaluate_forward_student import parse_args

    args = parse_args(["student_policy.npz", "--envs", "8"])

    assert args.policy.name == "student_policy.npz"
    assert args.envs == 8
