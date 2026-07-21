import numpy as np


def test_rolling_sweep_defaults_cover_structure_and_com_axes():
    from scripts.sweep_rolling_variants import DEFAULT_XML, parse_args

    args = parse_args([])

    assert args.xml == DEFAULT_XML
    assert args.xml.name == "pupper_v3_disk_visual.xml"
    assert args.hip_y == [0.07, 0.09]
    assert args.leg_scale == [1.0, 0.85]
    assert args.disk_radius == [0.17, 0.20]
    assert args.com_x == [-0.03, 0.0, 0.03]
    assert args.com_z == [-0.03, 0.0, 0.03]
    assert args.initial_speed == 0.8


def test_disk_com_offset_is_relative_to_disk_geometry_center():
    import mujoco

    from scripts.sweep_rolling_variants import DEFAULT_XML, set_disk_com

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    set_disk_com(model, com_x=0.02, com_z=-0.01)
    base_id = model.body("base_link").id
    disk_id = model.geom("base_disk_collision").id

    np.testing.assert_allclose(
        model.body_ipos[base_id],
        model.geom_pos[disk_id] + np.array([0.02, 0.0, -0.01]),
    )


def test_rolling_score_penalizes_directional_bias_and_rest_drift():
    from scripts.sweep_rolling_variants import score_result

    trial = {
        "distance": 1.0,
        "raw_distance": 1.0,
        "final_speed": 0.2,
        "lateral_drift": 0.01,
        "slip_rms": 0.02,
        "axis_tilt_rms_deg": 2.0,
        "disk_contact_fraction": 1.0,
        "foot_contact_fraction": 0.0,
        "actuator_work": 0.1,
        "failed": False,
    }
    rest = {**trial, "distance": 0.0, "raw_distance": 0.0, "final_speed": 0.0}
    symmetric = score_result(trial, trial, rest)
    biased = score_result(trial, {**trial, "distance": 0.4}, {**rest, "raw_distance": 0.2})

    assert symmetric["score"] > biased["score"]


def test_initial_geometry_metrics_report_com_and_rolling_envelope():
    import mujoco

    from scripts.sweep_rolling_variants import DEFAULT_XML, initial_geometry_metrics

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    metrics = initial_geometry_metrics(model)

    assert set(metrics) == {
        "complete_com_x",
        "complete_com_z",
        "complete_com_radial_offset",
        "rolling_foot_radial_margin",
    }
    assert metrics["complete_com_radial_offset"] >= 0.0
