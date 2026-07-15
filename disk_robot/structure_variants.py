from __future__ import annotations

from dataclasses import dataclass


LEG_ROOT_NAMES = (
    "leg_front_r_1",
    "leg_front_l_1",
    "leg_back_r_1",
    "leg_back_l_1",
)


@dataclass(frozen=True)
class StructureVariant:
    hip_y: float = 0.07
    leg_scale: float = 1.0
    disk_radius: float = 0.20


def disk_inertia(mass: float, radius: float, half_thickness: float) -> tuple[float, float, float]:
    """Returns body-frame inertia for a cylinder whose symmetry axis is body Y."""

    full_thickness = 2.0 * half_thickness
    symmetry_axis = 0.5 * mass * radius**2
    transverse = mass * (3.0 * radius**2 + full_thickness**2) / 12.0
    return transverse, symmetry_axis, transverse


def apply_structure_variant(model, variant: StructureVariant) -> None:
    """Applies a provisional physics variant to an already compiled MuJoCo model.

    Link scaling changes kinematic offsets and inertial locations while keeping the
    motor-dominated masses and foot sphere radii fixed. Mesh vertices are intentionally
    untouched, so this is a dynamics screening tool rather than a replacement for CAD.
    """

    if variant.hip_y <= 0.0:
        raise ValueError("hip_y must be positive")
    if not 0.5 <= variant.leg_scale <= 1.2:
        raise ValueError("leg_scale must be in [0.5, 1.2]")
    if variant.disk_radius <= 0.0:
        raise ValueError("disk_radius must be positive")

    import mujoco

    side_sign = {
        "leg_front_r_1": -1.0,
        "leg_front_l_1": 1.0,
        "leg_back_r_1": -1.0,
        "leg_back_l_1": 1.0,
    }
    for root_name in LEG_ROOT_NAMES:
        root_id = model.body(root_name).id
        model.body_pos[root_id, 1] = side_sign[root_name] * variant.hip_y
        if variant.leg_scale == 1.0:
            continue

        middle_name = root_name[:-1] + "2"
        distal_name = root_name[:-1] + "3"
        middle_id = model.body(middle_name).id
        distal_id = model.body(distal_name).id
        model.body_pos[distal_id] *= variant.leg_scale
        for body_id in (middle_id, distal_id):
            model.body_ipos[body_id] *= variant.leg_scale
            model.body_inertia[body_id] *= variant.leg_scale**2
            geom_start = model.body_geomadr[body_id]
            geom_stop = geom_start + model.body_geomnum[body_id]
            model.geom_pos[geom_start:geom_stop] *= variant.leg_scale
            for geom_id in range(geom_start, geom_stop):
                if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mesh_id = model.geom_dataid[geom_id]
                vertex_start = model.mesh_vertadr[mesh_id]
                vertex_stop = vertex_start + model.mesh_vertnum[mesh_id]
                model.mesh_vert[vertex_start:vertex_stop] *= variant.leg_scale
        foot_site_id = model.site(f"{distal_name}_foot_site").id
        model.site_pos[foot_site_id] *= variant.leg_scale

    collision_id = model.geom("base_disk_collision").id
    visual_id = model.geom("base_disk_visual").id
    model.geom_size[collision_id, 0] = variant.disk_radius
    model.geom_size[visual_id, 0] = variant.disk_radius
    base_id = model.body("base_link").id
    inertia = disk_inertia(
        float(model.body_mass[base_id]),
        variant.disk_radius,
        float(model.geom_size[collision_id, 1]),
    )
    model.body_inertia[base_id] = inertia

    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)


__all__ = ["LEG_ROOT_NAMES", "StructureVariant", "apply_structure_variant", "disk_inertia"]
