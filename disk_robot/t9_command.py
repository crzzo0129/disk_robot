from __future__ import annotations

from dataclasses import replace

from disk_robot.gait_speed import plan_forward_gait
from disk_robot.ik_reference import (
    IKReferenceSpec,
    build_ik_reference_bank,
)
from disk_robot.teacher_student_config import ForwardTeacherStudentConfig


T9_FORWARD_SPEED_ANCHORS = (0.00, 0.04, 0.06, 0.08, 0.10)


def validate_forward_speed_anchors(values):
    anchors = tuple(float(value) for value in values)
    if len(anchors) < 2:
        raise ValueError("T9 requires at least two forward-speed anchors")
    if anchors != tuple(sorted(set(anchors))):
        raise ValueError("T9 forward-speed anchors must be unique and increasing")
    if anchors[0] != 0.0:
        raise ValueError("T9 forward-speed anchors must include stop at 0.0 m/s")
    for value in anchors:
        plan_forward_gait(value)
    return anchors


def forward_reference_specs(
    values=T9_FORWARD_SPEED_ANCHORS,
    *,
    samples=256,
    step_height=0.025,
    duty=0.72,
):
    anchors = validate_forward_speed_anchors(values)
    specs = []
    for value in anchors:
        plan = plan_forward_gait(value)
        specs.append(
            IKReferenceSpec(
                samples=int(samples),
                frequency=plan.frequency,
                stride_length=plan.stride_length,
                step_height=step_height * plan.motion_scale,
                duty=float(duty),
                mode="trot",
            )
        )
    return tuple(specs)


def build_t9_reference_bank(
    xml_path,
    values=T9_FORWARD_SPEED_ANCHORS,
    *,
    samples=256,
    step_height=0.025,
    duty=0.72,
):
    anchors = validate_forward_speed_anchors(values)
    specs = forward_reference_specs(
        anchors, samples=samples, step_height=step_height, duty=duty
    )
    return build_ik_reference_bank(xml_path, anchors, specs)


def make_t9_config(
    base: ForwardTeacherStudentConfig | None = None,
    values=T9_FORWARD_SPEED_ANCHORS,
):
    anchors = validate_forward_speed_anchors(values)
    config = base or ForwardTeacherStudentConfig()
    return replace(
        config,
        command_vx=0.08 if 0.08 in anchors else anchors[-1],
        command_vx_values=anchors,
        student_phase_conditioned=True,
        student_previous_action_input=False,
        student_current_command_only=True,
        student_phase_frequency=plan_forward_gait(anchors[-1]).frequency,
    )


__all__ = [
    "T9_FORWARD_SPEED_ANCHORS",
    "build_t9_reference_bank",
    "forward_reference_specs",
    "make_t9_config",
    "validate_forward_speed_anchors",
]
