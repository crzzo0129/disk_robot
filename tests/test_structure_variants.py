import numpy as np


def test_disk_inertia_uses_body_y_as_the_symmetry_axis():
    from disk_robot.structure_variants import disk_inertia

    inertia = disk_inertia(mass=1.506, radius=0.20, half_thickness=0.04)

    np.testing.assert_allclose(inertia, (0.0158632, 0.03012, 0.0158632))


def test_structure_variant_updates_hip_leg_and_disk_geometry():
    import mujoco

    from disk_robot.structure_variants import StructureVariant, apply_structure_variant
    from disk_robot.walk_env import DEFAULT_XML

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    original_distal = model.body_pos[model.body("leg_front_r_3").id].copy()
    variant = StructureVariant(hip_y=0.085, leg_scale=0.85, disk_radius=0.18)

    apply_structure_variant(model, variant)

    assert model.body_pos[model.body("leg_front_r_1").id, 1] == -0.085
    assert model.body_pos[model.body("leg_front_l_1").id, 1] == 0.085
    np.testing.assert_allclose(
        model.body_pos[model.body("leg_front_r_3").id],
        0.85 * original_distal,
    )
    assert model.geom_size[model.geom("base_disk_collision").id, 0] == 0.18
    assert model.body_inertia[model.body("base_link").id, 1] > model.body_inertia[
        model.body("base_link").id, 0
    ]


def test_structure_sweep_defaults_cover_the_candidate_grid():
    from scripts.sweep_structure_variants import parse_args

    args = parse_args([])

    assert args.hip_y == [0.07, 0.085, 0.09]
    assert args.leg_scale == [1.0, 0.9, 0.85]
    assert args.disk_radius == [0.20, 0.18, 0.17]


def test_generated_candidate_xml_matches_selected_structure():
    import mujoco

    from scripts.write_structure_candidate import DEFAULT_OUTPUT

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_OUTPUT))

    assert model.body_pos[model.body("leg_front_r_1").id, 1] == -0.09
    assert model.body_pos[model.body("leg_front_l_1").id, 1] == 0.09
    assert model.geom_size[model.geom("base_disk_collision").id, 0] == 0.20
    np.testing.assert_allclose(
        model.body_inertia[model.body("base_link").id],
        (0.0158632, 0.03012, 0.0158632),
    )
