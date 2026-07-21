def test_viewer_defaults_to_recommended_rolling_candidate():
    from scripts.view_rolling_variant import parse_args

    args = parse_args([])

    assert args.hip_y == 0.09
    assert args.leg_scale == 0.85
    assert args.disk_radius == 0.20
    assert args.com_x == -0.005
    assert args.com_z == 0.030
    assert args.initial_speed == 0.8
    assert args.direction == "forward"


def test_launch_velocity_matches_pure_rolling_constraint():
    import mujoco

    from scripts.sweep_rolling_variants import _prepare_data
    from scripts.view_rolling_variant import build_model, parse_args, set_launch_velocity

    args = parse_args(["--direction", "reverse"])
    model = build_model(args)
    data = _prepare_data(model)
    dof, radius = set_launch_velocity(model, data, args.direction, args.initial_speed)

    assert data.qvel[dof] == -0.8
    assert data.qvel[dof + 4] == -0.8 / radius
