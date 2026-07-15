"""Generate a provisional XML for the selected structure variant."""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from disk_robot.structure_variants import LEG_ROOT_NAMES, StructureVariant, disk_inertia
from disk_robot_mjx.teacher_student_env import DEFAULT_XML


DEFAULT_OUTPUT = DEFAULT_XML.with_name("pupper_v3_disk_structure_candidate.xml")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hip-y", type=float, default=0.09)
    parser.add_argument("--leg-scale", type=float, default=0.85)
    parser.add_argument("--disk-radius", type=float, default=0.20)
    return parser.parse_args(argv)


def _numbers(value):
    return [float(item) for item in value.split()]


def _scaled(value, scale):
    return " ".join(f"{item * scale:.9g}" for item in _numbers(value))


def _named_body(root, name):
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise ValueError(f"Missing body {name!r}")
    return body


def write_candidate(source: Path, output: Path, variant: StructureVariant):
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "pupper_v3_disk_structure_candidate")
    side_sign = {
        "leg_front_r_1": -1.0,
        "leg_front_l_1": 1.0,
        "leg_back_r_1": -1.0,
        "leg_back_l_1": 1.0,
    }
    scaled_mesh_names = set()
    for root_name in LEG_ROOT_NAMES:
        hip = _named_body(root, root_name)
        hip_pos = _numbers(hip.get("pos"))
        hip_pos[1] = side_sign[root_name] * variant.hip_y
        hip.set("pos", " ".join(f"{item:.9g}" for item in hip_pos))

        middle = _named_body(root, root_name[:-1] + "2")
        distal = _named_body(root, root_name[:-1] + "3")
        distal.set("pos", _scaled(distal.get("pos"), variant.leg_scale))
        for body in (middle, distal):
            inertial = body.find("inertial")
            inertial.set("pos", _scaled(inertial.get("pos"), variant.leg_scale))
            inertial.set(
                "diaginertia",
                _scaled(inertial.get("diaginertia"), variant.leg_scale**2),
            )
            for geom in body.findall("geom"):
                if geom.get("pos"):
                    geom.set("pos", _scaled(geom.get("pos"), variant.leg_scale))
                if geom.get("mesh"):
                    scaled_mesh_names.add(geom.get("mesh"))
        for site in distal.findall("site"):
            if site.get("pos"):
                site.set("pos", _scaled(site.get("pos"), variant.leg_scale))

    for mesh_name in scaled_mesh_names:
        mesh = root.find(f".//mesh[@name='{mesh_name}']")
        current = _numbers(mesh.get("scale", "1 1 1"))
        mesh.set("scale", " ".join(f"{item * variant.leg_scale:.9g}" for item in current))

    base = _named_body(root, "base_link")
    base_inertial = base.find("inertial")
    mass = float(base_inertial.get("mass"))
    collision = root.find(".//geom[@name='base_disk_collision']")
    visual = root.find(".//geom[@name='base_disk_visual']")
    collision_size = _numbers(collision.get("size"))
    collision_size[0] = variant.disk_radius
    size_value = " ".join(f"{item:.9g}" for item in collision_size)
    collision.set("size", size_value)
    visual.set("size", size_value)
    base_inertial.set(
        "diaginertia",
        " ".join(
            f"{item:.9g}"
            for item in disk_inertia(mass, variant.disk_radius, collision_size[1])
        ),
    )

    root.insert(
        0,
        ET.Comment(
            f" Provisional structure candidate: hip_y={variant.hip_y:g}, "
            f"leg_scale={variant.leg_scale:g}, disk_radius={variant.disk_radius:g}. "
            "Regenerate with scripts.write_structure_candidate. "
        ),
    )
    ET.indent(tree, space="    ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=False)


def main(argv=None):
    args = parse_args(argv)
    variant = StructureVariant(args.hip_y, args.leg_scale, args.disk_radius)
    write_candidate(args.source.resolve(), args.output.resolve(), variant)
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
