import numpy as np


def test_disk_inertia_uses_body_y_as_the_symmetry_axis():
    from disk_robot.structure_variants import disk_inertia

    inertia = disk_inertia(mass=1.506, radius=0.20, half_thickness=0.04)

    np.testing.assert_allclose(inertia, (0.0158632, 0.03012, 0.0158632))


def test_structure_variant_updates_hip_leg_and_disk_geometry():
    import mujoco

    from disk_robot.model_paths import BASE_MODEL_XML
    from disk_robot.structure_variants import StructureVariant, apply_structure_variant

    model = mujoco.MjModel.from_xml_path(str(BASE_MODEL_XML))
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
    from scripts.sweep_structure_variants import DEFAULT_XML, parse_args

    args = parse_args([])

    assert args.xml == DEFAULT_XML
    assert args.xml.name == "pupper_v3_disk_visual.xml"
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


def test_candidate_generation_defaults_to_unscaled_source():
    from scripts.write_structure_candidate import DEFAULT_OUTPUT, DEFAULT_SOURCE, parse_args

    args = parse_args([])

    assert args.source == DEFAULT_SOURCE
    assert args.source.name == "pupper_v3_disk_visual.xml"
    assert args.output == DEFAULT_OUTPUT
    assert args.output.name == "pupper_v3_disk_structure_candidate.xml"
