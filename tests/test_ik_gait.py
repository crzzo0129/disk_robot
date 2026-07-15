import numpy as np


def test_foot_trajectory_is_continuous_and_lifts_only_during_swing():
    from disk_robot.ik_gait import FootTrajectoryParams, foot_offset

    params = FootTrajectoryParams(stride_length=0.04, step_height=0.025, duty=0.7)
    before_liftoff = foot_offset(params.duty - 1e-7, params)
    after_liftoff = foot_offset(params.duty + 1e-7, params)
    before_touchdown = foot_offset(1.0 - 1e-7, params)
    after_touchdown = foot_offset(1e-7, params)

    assert np.allclose(before_liftoff, after_liftoff, atol=1e-6)
    assert np.allclose(before_touchdown, after_touchdown, atol=1e-6)
    assert foot_offset(0.5 * (1.0 + params.duty), params)[2] == params.step_height
    assert foot_offset(0.5 * params.duty, params)[2] == 0.0

    eps = 1e-5
    stance_velocity = (
        foot_offset(params.duty - eps, params)[0]
        - foot_offset(params.duty - 2.0 * eps, params)[0]
    ) / eps
    swing_velocity = (
        foot_offset(params.duty + 2.0 * eps, params)[0]
        - foot_offset(params.duty + eps, params)[0]
    ) / eps
    np.testing.assert_allclose(stance_velocity, swing_velocity, rtol=2e-3, atol=2e-3)
    assert abs(foot_offset(params.duty + eps, params)[2]) < 1e-10
    assert abs(foot_offset(1.0 - eps, params)[2]) < 1e-10


def test_ik_tracks_reachable_foot_trajectory_on_target_model():
    import mujoco

    from disk_robot.ik_gait import FootSpaceIKGait, FootTrajectoryParams
    from disk_robot.model_contract import resolve_model_contract
    from disk_robot.walk_env import DEFAULT_XML

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    contract = resolve_model_contract(model)
    gait = FootSpaceIKGait(model, contract, FootTrajectoryParams())

    for t in np.linspace(0.0, 1.25, 20, endpoint=False):
        targets = gait.targets(float(t))
        assert np.all(np.isfinite(targets))
        assert np.max(gait.last_errors) < 2e-3
