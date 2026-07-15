"""Physically simulate a robot through staged pose transitions."""
import argparse
from contextlib import nullcontext
import math
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from disk_robot.video_recorder import MujocoVideoRecorder


XML_PATH = REPO_ROOT / "assets" / "disk_quadruped_extreme_train.xml"
PUPPER_XML_PATH = REPO_ROOT / "assets" / "pupper_v3_disk_visual.xml"
DEFAULT_FROM_KEYFRAME = "walk_stand"
DEFAULT_MIDDLE_KEYFRAME = "stand"
DEFAULT_TO_KEYFRAME = "folded"

KEY_HELP = """
Keyboard controls:
  Space : pause / resume
  R     : reset simulation
"""


class PlaybackState:
    def __init__(self):
        self.paused = False
        self.switched = False
        self.push_start_time = None
        self.repeat_start_time = None
        self.repeat_phase = None
        self.repeat_phase_start = None
        self.repeat_foot_targets = None
        self.repeat_retract_ctrl = None
        self.repeat_contact_lost_time = None
        self.repeat_prepare_targets = None
        self.repeat_touchdown_targets = None
        self.repeat_count = 0
        self.rolling_angle = 0.0
        self._lock = threading.Lock()
        self._commands = []

    def push(self, command):
        with self._lock:
            self._commands.append(command)

    def drain(self):
        with self._lock:
            commands = self._commands
            self._commands = []
        return commands


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("disk", "pupper"), default="disk")
    parser.add_argument("--motion", choices=("default", "rear-push-roll"), default="default")
    parser.add_argument("--xml-path", type=Path)
    parser.add_argument("--track-body")
    parser.add_argument("--disk-geom")
    parser.add_argument("--from-keyframe")
    parser.add_argument("--middle-keyframe")
    parser.add_argument("--to-keyframe")
    parser.add_argument("--switch-time", type=float, help="Seconds to wait before starting the first transition.")
    parser.add_argument(
        "--front-fold-time",
        "--home-to-folded-time",
        dest="front_fold_time",
        type=float,
        help="Seconds to fold the front legs first.",
    )
    parser.add_argument("--rear-fold-time", type=float, help="Seconds to fold the rear legs after the front legs.")
    parser.add_argument("--folded-hold-time", type=float, help="Seconds to hold the preparatory folded pose.")
    parser.add_argument(
        "--fold-order",
        choices=("front-first", "rear-first", "simultaneous"),
        default="simultaneous",
        help="Coordination used for the standing-to-folded transition.",
    )
    parser.add_argument(
        "--folded-style",
        choices=("keyframe", "balanced"),
        default="keyframe",
        help="Use the XML folded pose or a COM-balanced rolling pose.",
    )
    parser.add_argument(
        "--rolling-pose",
        choices=("folded", "axial"),
        default="axial",
        help="Pose held after push; axial tucks the complete COM close to the disk axis.",
    )
    parser.add_argument("--walk-to-stand-time", type=float, help="Seconds to move from the first pose to the middle pose.")
    parser.add_argument("--stand-hold-time", type=float, help="Seconds to hold the middle pose.")
    parser.add_argument("--stand-to-folded-time", type=float, help="Seconds to move from the middle pose to the final pose.")
    parser.add_argument("--kp", type=float, help="Optional runtime position gain override.")
    parser.add_argument("--kd", type=float, help="Optional runtime velocity gain override.")
    parser.add_argument("--force-limit", type=float, help="Optional symmetric runtime actuator force limit.")
    parser.add_argument(
        "--push-scale",
        type=float,
        help="Fraction of the selected rear-leg push excursion to use (rear-push-roll only).",
    )
    parser.add_argument(
        "--push-style",
        choices=("ground", "tangent", "keyframe"),
        default="tangent",
        help="Plant and sweep the rear feet, swing them tangentially, or use the legacy keyframe.",
    )
    parser.add_argument(
        "--push-trigger-speed",
        type=float,
        help="Minimum forward body speed before the rear push starts; <=0 disables phase gating.",
    )
    parser.add_argument("--push-trigger-timeout", type=float, help="Maximum extra seconds to wait for the push phase.")
    parser.add_argument(
        "--repeat-pushes",
        type=int,
        default=0,
        help="Number of extra pushes after launch; use -1 to repeat indefinitely.",
    )
    parser.add_argument(
        "--repeat-controller",
        choices=("swing", "contact"),
        default="contact",
        help="Use the legacy timed leg swing or the experimental contact-driven stance controller.",
    )
    parser.add_argument("--turns-per-push", type=float, default=1.0, help="Disk revolutions between repeated pushes.")
    parser.add_argument(
        "--repeat-trigger-max-speed",
        type=float,
        default=0.40,
        help="Only start a repeated push after forward speed has fallen below this value.",
    )
    parser.add_argument(
        "--repeat-prepare-time",
        type=float,
        default=0.08,
        help="Seconds to move from rolling_folded back to the push preparation pose.",
    )
    parser.add_argument(
        "--repeat-foot-height",
        type=float,
        default=0.028,
        help="Maximum predicted rear-foot center height that permits a repeated push.",
    )
    parser.add_argument(
        "--repeat-force-limit",
        type=float,
        default=0.35,
        help="Rear-actuator force limit during repeated ground pushes.",
    )
    parser.add_argument("--repeat-seek-time", type=float, default=0.28, help="Maximum seconds used to seek ground contact.")
    parser.add_argument(
        "--repeat-stance-time",
        type=float,
        default=0.18,
        help="Seconds to hold the rear feet near fixed world positions after contact.",
    )
    parser.add_argument(
        "--repeat-foot-preload",
        type=float,
        default=0.008,
        help="Downward world-space foot offset used to maintain stance contact.",
    )
    parser.add_argument(
        "--repeat-contact-grace",
        type=float,
        default=0.016,
        help="Seconds of lost rear-foot contact tolerated before ending stance.",
    )
    parser.add_argument(
        "--repeat-retract-time",
        type=float,
        default=0.20,
        help="Seconds used to return from stance to rolling_folded.",
    )
    parser.add_argument(
        "--repeat-retract-delay",
        type=float,
        default=0.0,
        help="Seconds to coast in the unloaded stance pose before retracting.",
    )
    parser.add_argument(
        "--repeat-stance-sweep",
        type=float,
        default=0.0,
        help="Backward world-space rear-foot sweep over a full stance.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without the interactive viewer.")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps for --headless.")
    parser.add_argument("--video", type=Path, help="Write an MP4 while running the headless simulation.")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument("--status-interval", type=float, default=3, help="Seconds between status prints.")
    args = parser.parse_args(argv)

    if args.motion == "rear-push-roll":
        args.model = "pupper"
        args.xml_path = args.xml_path or PUPPER_XML_PATH
        args.track_body = args.track_body or "base_link"
        args.disk_geom = args.disk_geom or "base_disk_collision"
        args.from_keyframe = args.from_keyframe or "home"
        args.middle_keyframe = args.middle_keyframe or "rear_push"
        args.to_keyframe = args.to_keyframe or "folded"
        args.switch_time = 0.5 if args.switch_time is None else args.switch_time
        args.front_fold_time = 0.18 if args.front_fold_time is None else args.front_fold_time
        args.rear_fold_time = 0.55 if args.rear_fold_time is None else args.rear_fold_time
        # The fold excites a large backward roll. This is only a minimum hold;
        # phase gating below waits for the first sufficiently fast forward return.
        args.folded_hold_time = 0.3 if args.folded_hold_time is None else args.folded_hold_time
        if args.walk_to_stand_time is None:
            args.walk_to_stand_time = 0.45 if args.push_style == "ground" else 0.28
        args.stand_hold_time = 0.04 if args.stand_hold_time is None else args.stand_hold_time
        args.stand_to_folded_time = 0.14 if args.stand_to_folded_time is None else args.stand_to_folded_time
        args.push_scale = 1.0 if args.push_scale is None else args.push_scale
        args.push_trigger_speed = 0.1 if args.push_trigger_speed is None else args.push_trigger_speed
        args.push_trigger_timeout = 2.0 if args.push_trigger_timeout is None else args.push_trigger_timeout
        args.kp = 60.0 if args.kp is None else args.kp
        args.kd = 1.0 if args.kd is None else args.kd
        args.force_limit = 6.0 if args.force_limit is None else args.force_limit
    elif args.model == "pupper":
        args.xml_path = args.xml_path or PUPPER_XML_PATH
        args.track_body = args.track_body or "base_link"
        args.disk_geom = args.disk_geom or "base_disk_collision"
        args.from_keyframe = args.from_keyframe or "home"
        args.middle_keyframe = args.middle_keyframe or "home"
        if args.walk_to_stand_time is None:
            args.walk_to_stand_time = 0.0
    else:
        args.xml_path = args.xml_path or XML_PATH
        args.track_body = args.track_body or "disk_torso"
        args.disk_geom = args.disk_geom or "torso_disk"
        args.from_keyframe = args.from_keyframe or DEFAULT_FROM_KEYFRAME
        args.middle_keyframe = args.middle_keyframe or DEFAULT_MIDDLE_KEYFRAME
        if args.walk_to_stand_time is None:
            args.walk_to_stand_time = 2.0
    args.to_keyframe = args.to_keyframe or DEFAULT_TO_KEYFRAME
    args.switch_time = 0.5 if args.switch_time is None else args.switch_time
    args.stand_hold_time = 0.5 if args.stand_hold_time is None else args.stand_hold_time
    args.stand_to_folded_time = 2.0 if args.stand_to_folded_time is None else args.stand_to_folded_time
    if args.push_scale is not None and not 0.0 <= args.push_scale <= 2.0:
        parser.error("--push-scale must be between 0 and 2")
    if args.repeat_pushes < -1:
        parser.error("--repeat-pushes must be -1 or non-negative")
    if args.turns_per_push <= 0.0:
        parser.error("--turns-per-push must be positive")
    if args.repeat_trigger_max_speed <= 0.0:
        parser.error("--repeat-trigger-max-speed must be positive")
    if args.repeat_foot_height <= 0.0:
        parser.error("--repeat-foot-height must be positive")
    if args.repeat_force_limit <= 0.0:
        parser.error("--repeat-force-limit must be positive")
    if args.repeat_seek_time <= 0.0 or args.repeat_stance_time <= 0.0:
        parser.error("--repeat-seek-time and --repeat-stance-time must be positive")
    if args.repeat_foot_preload < 0.0:
        parser.error("--repeat-foot-preload must be non-negative")
    if args.repeat_contact_grace < 0.0 or args.repeat_retract_time <= 0.0:
        parser.error("--repeat-contact-grace must be non-negative and --repeat-retract-time must be positive")
    if args.repeat_retract_delay < 0.0:
        parser.error("--repeat-retract-delay must be non-negative")
    if args.repeat_stance_sweep < 0.0:
        parser.error("--repeat-stance-sweep must be non-negative")
    if args.video_fps <= 0 or args.video_width <= 0 or args.video_height <= 0:
        parser.error("video fps and dimensions must be positive")
    return args


def lerp_sequence(start, end, alpha, ranges=None):
    values = [float(a) * (1.0 - alpha) + float(b) * alpha for a, b in zip(start, end)]
    if ranges is None:
        return values
    return [min(max(value, float(low)), float(high)) for value, (low, high) in zip(values, ranges)]


def triangle_phase(phase):
    phase = phase % 1.0
    if phase <= 0.5:
        return phase * 2.0
    return (1.0 - phase) * 2.0


def transition_alpha(elapsed, duration):
    if elapsed < 0.0:
        return 0.0
    if duration <= 0.0:
        return 1.0
    return min(max(elapsed / duration, 0.0), 1.0)


def smootherstep(alpha):
    """Map [0, 1] to [0, 1] with zero velocity and acceleration at both ends."""
    alpha = min(max(float(alpha), 0.0), 1.0)
    return alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)


def partial_push_target(folded_ctrl, rear_push_ctrl, push_scale, ranges=None):
    return lerp_sequence(folded_ctrl, rear_push_ctrl, push_scale, ranges)


def tangential_rear_push_target(folded_ctrl, push_scale=1.0, ranges=None):
    """Move both rear feet backward with almost no vertical endpoint displacement."""
    if len(folded_ctrl) != 12:
        raise ValueError("The tangential Pupper push expects 12 actuator targets")
    full_push = list(folded_ctrl)
    # Values were selected from the compiled Pupper kinematics. Relative to the
    # folded pose, each rear foot moves about 78 mm backward and <0.1 mm vertically.
    if folded_ctrl[6] < -2.5:
        full_push[6:9] = [-2.9, 0.0, 1.94]
        full_push[9:12] = [2.9, 0.0, -1.94]
    else:
        full_push[6:9] = [-1.4, 0.0, 0.05]
        full_push[9:12] = [1.4, 0.0, -0.05]
    return partial_push_target(folded_ctrl, full_push, push_scale, ranges)


def balanced_folded_target(keyframe_folded_ctrl):
    """Fold rear-leg mass forward so the complete robot COM lies near the disk axis."""
    if len(keyframe_folded_ctrl) != 12:
        raise ValueError("The balanced Pupper fold expects 12 actuator targets")
    target = list(keyframe_folded_ctrl)
    target[6:9] = [-3.0, 0.0, 2.65]
    target[9:12] = [3.0, 0.0, -2.65]
    return target


def axial_rolling_target(folded_ctrl):
    """Compact all legs so the complete COM is close to the disk's X-Z axis position."""
    if len(folded_ctrl) != 12:
        raise ValueError("The axial Pupper rolling pose expects 12 actuator targets")
    target = list(folded_ctrl)
    target[0:3] = [-2.1452, 0.0, -1.0750]
    target[3:6] = [2.1452, 0.0, 1.0750]
    target[6:9] = [-2.2855, 0.0, -2.2500]
    target[9:12] = [2.2855, 0.0, 2.2500]
    return target


def ground_rear_push_targets(folded_ctrl, push_scale=1.0, ranges=None):
    """Return rear-foot planting and backward-sweep targets for the XML folded pose."""
    if len(folded_ctrl) != 12:
        raise ValueError("The grounded Pupper push expects 12 actuator targets")
    plant = list(folded_ctrl)
    plant[6:9] = [-0.62, 0.0, 0.48]
    plant[9:12] = [0.62, 0.0, -0.48]
    full_push = list(plant)
    full_push[6:9] = [-0.76, 0.0, -0.30]
    full_push[9:12] = [0.76, 0.0, 0.30]
    push = partial_push_target(plant, full_push, push_scale, ranges)
    if ranges is not None:
        plant = lerp_sequence(plant, plant, 1.0, ranges)
    return plant, push


def folding_target(
    from_ctrl,
    folded_ctrl,
    stage,
    alpha,
    fold_order,
    front_fold_time,
    rear_fold_time,
    ranges=None,
):
    """Coordinate the first six (front) and last six (rear) leg targets."""
    front_folded = folded_ctrl[:6] + from_ctrl[6:]
    rear_folded = from_ctrl[:6] + folded_ctrl[6:]
    eased = smootherstep(alpha)
    if fold_order == "front-first":
        start, end = (from_ctrl, front_folded) if stage == "front_fold" else (front_folded, folded_ctrl)
        return lerp_sequence(start, end, eased, ranges)
    if fold_order == "rear-first":
        start, end = (from_ctrl, rear_folded) if stage == "front_fold" else (rear_folded, folded_ctrl)
        return lerp_sequence(start, end, eased, ranges)

    elapsed = alpha * front_fold_time if stage == "front_fold" else front_fold_time + alpha * rear_fold_time
    total_time = max(front_fold_time, 0.0) + max(rear_fold_time, 0.0)
    global_alpha = 1.0 if total_time <= 0.0 else elapsed / total_time
    return lerp_sequence(from_ctrl, folded_ctrl, smootherstep(global_alpha), ranges)


def staged_target(sim_time, switch_time, walk_to_stand_time, stand_hold_time, stand_to_folded_time):
    first_elapsed = sim_time - switch_time
    first_alpha = transition_alpha(first_elapsed, walk_to_stand_time)
    if first_alpha < 1.0:
        return "walk_to_stand", first_alpha

    second_start = switch_time + max(walk_to_stand_time, 0.0) + max(stand_hold_time, 0.0)
    second_alpha = transition_alpha(sim_time - second_start, stand_to_folded_time)
    if second_alpha < 1.0:
        if sim_time < second_start:
            return "stand_hold", 0.0
        return "stand_to_folded", second_alpha
    return "folded", 1.0


def rear_push_roll_target(
    sim_time,
    switch_time,
    front_fold_time,
    rear_fold_time,
    folded_hold_time,
    folded_to_push_time,
    push_hold_time,
    push_to_folded_time,
):
    front_alpha = transition_alpha(sim_time - switch_time, front_fold_time)
    if front_alpha < 1.0:
        return "front_fold", front_alpha

    rear_start = switch_time + max(front_fold_time, 0.0)
    rear_alpha = transition_alpha(sim_time - rear_start, rear_fold_time)
    if rear_alpha < 1.0:
        return "rear_fold", rear_alpha

    folded_hold_end = rear_start + max(rear_fold_time, 0.0) + max(folded_hold_time, 0.0)
    if sim_time < folded_hold_end:
        return "folded_hold", 0.0

    push_alpha = transition_alpha(sim_time - folded_hold_end, folded_to_push_time)
    if push_alpha < 1.0:
        return "folded_to_push", push_alpha

    push_hold_end = folded_hold_end + max(folded_to_push_time, 0.0) + max(push_hold_time, 0.0)
    if sim_time < push_hold_end:
        return "push_hold", 0.0

    return_alpha = transition_alpha(sim_time - push_hold_end, push_to_folded_time)
    if return_alpha < 1.0:
        return "push_to_folded", return_alpha
    return "rolling", 1.0


def repeated_push_target(elapsed, prepare_time, push_time, push_hold_time, retract_time):
    prepare_alpha = transition_alpha(elapsed, prepare_time)
    if prepare_alpha < 1.0:
        return "repeat_prepare", prepare_alpha
    push_start = max(prepare_time, 0.0)
    push_alpha = transition_alpha(elapsed - push_start, push_time)
    if push_alpha < 1.0:
        return "repeat_push", push_alpha
    hold_end = push_start + max(push_time, 0.0) + max(push_hold_time, 0.0)
    if elapsed < hold_end:
        return "repeat_hold", 0.0
    retract_alpha = transition_alpha(elapsed - hold_end, retract_time)
    if retract_alpha < 1.0:
        return "repeat_retract", retract_alpha
    return "repeat_done", 1.0


def step_target_alpha(sim_time, switch_time, transition_time):
    return transition_alpha(sim_time - switch_time, transition_time)


def format_status_line(
    sim_time,
    stage,
    alpha,
    torso_x,
    torso_y,
    torso_z,
    contact_count,
    disk_contact_count,
    velocity_x=None,
    velocity_z=None,
    angular_velocity_y=None,
    saturation_fraction=None,
):
    line = (
        f"t={sim_time:.2f} stage={stage} alpha={alpha:.2f} "
        f"x={torso_x:.3f} y={torso_y:.3f} z={torso_z:.3f} "
        f"contacts={contact_count} disk_contacts={disk_contact_count}"
    )
    if velocity_x is not None:
        line += f" vx={velocity_x:.3f} vz={velocity_z:.3f} wy={angular_velocity_y:.3f} sat={saturation_fraction:.2f}"
    return line


def _keyframe_id(mujoco, model, name):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        raise ValueError(f"Keyframe not found: {name}")
    return key_id


def _body_id(mujoco, model, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Body not found: {name}")
    return body_id


def _geom_id(mujoco, model, name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"Geom not found: {name}")
    return geom_id


def _keyframe_ctrl(model, key_id):
    return [float(value) for value in model.key_ctrl[key_id]]


def _control_ranges(model):
    return [(float(low), float(high)) for low, high in model.actuator_ctrlrange]


def _disk_contact_count(data, disk_geom_id):
    total = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 == disk_geom_id or contact.geom2 == disk_geom_id:
            total += 1
    return total


def _body_contact_count(model, data, body_ids):
    total = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 in body_ids or body2 in body_ids:
            total += 1
    return total


def _status_line(model, data, torso_id, disk_geom_id, rear_foot_body_ids, stage, alpha):
    torso_pos = data.xpos[torso_id]
    velocity = data.cvel[torso_id]
    limits = model.actuator_forcerange[:, 1]
    saturated = sum(
        abs(float(force)) >= 0.99 * abs(float(limit))
        for force, limit in zip(data.actuator_force, limits)
        if limit > 0.0
    )
    limited_count = sum(float(limit) > 0.0 for limit in limits)
    line = format_status_line(
        sim_time=float(data.time),
        stage=stage,
        alpha=alpha,
        torso_x=float(torso_pos[0]),
        torso_y=float(torso_pos[1]),
        torso_z=float(torso_pos[2]),
        contact_count=int(data.ncon),
        disk_contact_count=_disk_contact_count(data, disk_geom_id),
        velocity_x=float(velocity[3]),
        velocity_z=float(velocity[5]),
        angular_velocity_y=float(velocity[1]),
        saturation_fraction=saturated / max(limited_count, 1),
    )
    return f"{line} rear_foot_contacts={_body_contact_count(model, data, rear_foot_body_ids)}"


def main(argv=None):
    args = parse_args(argv)

    import mujoco
    from mujoco import viewer

    model = mujoco.MjModel.from_xml_path(str(args.xml_path.resolve()))
    if args.kp is not None:
        model.actuator_gainprm[:, 0] = args.kp
        model.actuator_biasprm[:, 1] = -args.kp
    if args.kd is not None:
        model.actuator_biasprm[:, 2] = -args.kd
    if args.force_limit is not None:
        model.actuator_forcerange[:, 0] = -abs(args.force_limit)
        model.actuator_forcerange[:, 1] = abs(args.force_limit)
    data = mujoco.MjData(model)
    from_id = _keyframe_id(mujoco, model, args.from_keyframe)
    middle_id = _keyframe_id(mujoco, model, args.middle_keyframe)
    to_id = _keyframe_id(mujoco, model, args.to_keyframe)
    from_ctrl = _keyframe_ctrl(model, from_id)
    middle_ctrl = _keyframe_ctrl(model, middle_id)
    to_ctrl = _keyframe_ctrl(model, to_id)
    if args.motion == "rear-push-roll" and args.folded_style == "balanced":
        to_ctrl = balanced_folded_target(to_ctrl)
    if args.rolling_pose == "axial":
        rolling_id = _keyframe_id(mujoco, model, "rolling_folded")
        rolling_ctrl = _keyframe_ctrl(model, rolling_id)
    else:
        rolling_ctrl = to_ctrl
    ctrl_ranges = _control_ranges(model)
    plant_ctrl = None
    if args.push_style == "ground":
        plant_ctrl, push_ctrl = ground_rear_push_targets(to_ctrl, args.push_scale or 0.0, ctrl_ranges)
    elif args.push_style == "tangent":
        push_ctrl = tangential_rear_push_target(to_ctrl, args.push_scale or 0.0, ctrl_ranges)
    else:
        push_ctrl = partial_push_target(to_ctrl, middle_ctrl, args.push_scale or 0.0, ctrl_ranges)
    repeat_plant_ctrl, repeat_push_ctrl = ground_rear_push_targets(to_ctrl, args.push_scale or 0.0, ctrl_ranges)
    # Repeated pushes start from rolling_folded. Keep the front legs tucked;
    # only the rear six actuators may participate in planting and pushing.
    repeat_plant_ctrl[:6] = rolling_ctrl[:6]
    repeat_push_ctrl[:6] = rolling_ctrl[:6]
    nominal_force_ranges = model.actuator_forcerange.copy()
    probe_data = mujoco.MjData(model)
    rear_foot_site_ids = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "leg_back_r_3_foot_site"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "leg_back_l_3_foot_site"),
    )
    rear_foot_body_ids = {
        _body_id(mujoco, model, "leg_back_r_3"),
        _body_id(mujoco, model, "leg_back_l_3"),
    }
    torso_id = _body_id(mujoco, model, args.track_body)
    disk_geom_id = _geom_id(mujoco, model, args.disk_geom)
    state = PlaybackState()

    def reset_simulation():
        state.switched = False
        state.push_start_time = None
        state.repeat_start_time = None
        state.repeat_phase = None
        state.repeat_phase_start = None
        state.repeat_foot_targets = None
        state.repeat_retract_ctrl = None
        state.repeat_contact_lost_time = None
        state.repeat_prepare_targets = None
        state.repeat_touchdown_targets = None
        state.repeat_count = 0
        state.rolling_angle = 0.0
        mujoco.mj_resetDataKeyframe(model, data, from_id)
        data.ctrl[:] = from_ctrl
        mujoco.mj_forward(model, data)

    def update_target():
        stage, alpha = current_target()
        if not state.switched:
            if stage == "front_fold":
                data.ctrl[:] = folding_target(
                    from_ctrl,
                    to_ctrl,
                    stage,
                    alpha,
                    args.fold_order,
                    args.front_fold_time,
                    args.rear_fold_time,
                    ctrl_ranges,
                )
            elif stage == "rear_fold":
                data.ctrl[:] = folding_target(
                    from_ctrl,
                    to_ctrl,
                    stage,
                    alpha,
                    args.fold_order,
                    args.front_fold_time,
                    args.rear_fold_time,
                    ctrl_ranges,
                )
            elif stage == "folded_hold":
                data.ctrl[:] = to_ctrl
            elif stage == "folded_to_push":
                if plant_ctrl is None:
                    data.ctrl[:] = lerp_sequence(to_ctrl, push_ctrl, smootherstep(alpha), ctrl_ranges)
                elif alpha < 0.55:
                    data.ctrl[:] = lerp_sequence(to_ctrl, plant_ctrl, smootherstep(alpha / 0.55), ctrl_ranges)
                else:
                    sweep_alpha = (alpha - 0.55) / 0.45
                    data.ctrl[:] = lerp_sequence(plant_ctrl, push_ctrl, smootherstep(sweep_alpha), ctrl_ranges)
            elif stage == "push_hold":
                data.ctrl[:] = push_ctrl
            elif stage == "push_to_folded":
                data.ctrl[:] = lerp_sequence(push_ctrl, rolling_ctrl, smootherstep(alpha), ctrl_ranges)
            elif stage == "walk_to_stand":
                data.ctrl[:] = lerp_sequence(from_ctrl, middle_ctrl, alpha, ctrl_ranges)
            elif stage in ("stand_hold", "stand_to_folded"):
                fold_alpha = 0.0 if stage == "stand_hold" else alpha
                data.ctrl[:] = lerp_sequence(middle_ctrl, to_ctrl, fold_alpha, ctrl_ranges)
            else:
                final_ctrl = rolling_ctrl if args.motion == "rear-push-roll" else to_ctrl
                data.ctrl[:] = lerp_sequence(middle_ctrl, final_ctrl, 1.0, ctrl_ranges)
        finished_stage = "rolling" if args.motion == "rear-push-roll" else "folded"
        if not state.switched and stage == finished_stage:
            state.switched = True
            print(f"finished target transition to {args.to_keyframe} at t={data.time:.3f}", flush=True)
        return stage, alpha

    def current_target():
        if args.motion == "rear-push-roll":
            nominal_push_start = (
                args.switch_time
                + max(args.front_fold_time, 0.0)
                + max(args.rear_fold_time, 0.0)
                + max(args.folded_hold_time, 0.0)
            )
            if data.time >= nominal_push_start and state.push_start_time is None:
                velocity = data.cvel[torso_id]
                disk_grounded = _disk_contact_count(data, disk_geom_id) > 0
                phase_ready = (
                    args.push_trigger_speed <= 0.0
                    or (
                        float(velocity[3]) >= args.push_trigger_speed
                        and abs(float(velocity[5])) <= 0.08
                        and disk_grounded
                    )
                )
                timed_out = data.time >= nominal_push_start + max(args.push_trigger_timeout, 0.0)
                if phase_ready or timed_out:
                    state.push_start_time = float(data.time)
                    reason = "phase_ready" if phase_ready else "timeout"
                    print(
                        f"push_trigger={reason} t={data.time:.3f} "
                        f"vx={velocity[3]:.3f} vz={velocity[5]:.3f} disk_grounded={disk_grounded}",
                        flush=True,
                    )
                else:
                    return "folded_wait", 0.0

            effective_hold_time = args.folded_hold_time
            if state.push_start_time is not None:
                rear_fold_end = args.switch_time + max(args.front_fold_time, 0.0) + max(args.rear_fold_time, 0.0)
                effective_hold_time = max(state.push_start_time - rear_fold_end, 0.0)
            return rear_push_roll_target(
                data.time,
                args.switch_time,
                args.front_fold_time,
                args.rear_fold_time,
                effective_hold_time,
                args.walk_to_stand_time,
                args.stand_hold_time,
                args.stand_to_folded_time,
            )
        return staged_target(
            data.time,
            args.switch_time,
            args.walk_to_stand_time,
            args.stand_hold_time,
            args.stand_to_folded_time,
        )

    def step_physics():
        stage, alpha = current_target()
        if not state.paused:
            if state.switched and args.motion == "rear-push-roll" and args.repeat_pushes != 0:
                stage, alpha = update_repeated_push()
            else:
                stage, alpha = update_target()
            mujoco.mj_step(model, data)
            if state.switched and state.repeat_start_time is None:
                state.rolling_angle += max(float(data.cvel[torso_id][1]), 0.0) * model.opt.timestep
        return stage, alpha

    def update_repeated_push():
        if args.repeat_pushes >= 0 and state.repeat_count >= args.repeat_pushes:
            model.actuator_forcerange[:] = nominal_force_ranges
            data.ctrl[:] = rolling_ctrl
            return "rolling", 1.0

        trigger_angle = 2.0 * math.pi * args.turns_per_push
        if state.repeat_start_time is None:
            predicted_foot_height = target_rear_foot_height(repeat_plant_ctrl)
            phase_ready = predicted_foot_height <= args.repeat_foot_height
            forward_speed = float(data.cvel[torso_id][3])
            speed_ready = 0.0 < forward_speed <= args.repeat_trigger_max_speed
            if state.rolling_angle < trigger_angle or not phase_ready or not speed_ready:
                model.actuator_forcerange[:] = nominal_force_ranges
                data.ctrl[:] = rolling_ctrl
                return "rolling", min(state.rolling_angle / trigger_angle, 1.0)
            state.repeat_start_time = float(data.time)
            state.repeat_phase = "prepare"
            state.repeat_phase_start = float(data.time)
            state.repeat_prepare_targets = [data.site_xpos[site_id].copy() for site_id in rear_foot_site_ids]
            state.repeat_touchdown_targets = target_rear_foot_positions(repeat_plant_ctrl)
            for target in state.repeat_touchdown_targets:
                target[2] = 0.01995
            print(
                f"repeat_push_start index={state.repeat_count + 1}/"
                f"{'infinite' if args.repeat_pushes < 0 else args.repeat_pushes} "
                f"t={data.time:.3f} accumulated_angle={state.rolling_angle:.3f} "
                f"predicted_rear_foot_z={predicted_foot_height:.4f} vx={forward_speed:.3f}",
                flush=True,
            )

        model.actuator_forcerange[:] = nominal_force_ranges
        model.actuator_forcerange[6:, 0] = -args.repeat_force_limit
        model.actuator_forcerange[6:, 1] = args.repeat_force_limit

        if args.repeat_controller == "swing":
            return update_swing_repeat()

        phase_elapsed = float(data.time) - state.repeat_phase_start
        if state.repeat_phase == "prepare":
            alpha = transition_alpha(phase_elapsed, args.repeat_prepare_time)
            blend = smootherstep(alpha)
            desired_positions = [
                start * (1.0 - blend) + end * blend
                for start, end in zip(state.repeat_prepare_targets, state.repeat_touchdown_targets)
            ]
            data.ctrl[:] = solve_rear_world_targets(data.ctrl, desired_positions)
            stage = "repeat_prepare"
            if alpha >= 1.0:
                state.repeat_phase = "seek"
                state.repeat_phase_start = float(data.time)
        elif state.repeat_phase == "seek":
            rear_contacts = _body_contact_count(model, data, rear_foot_body_ids)
            disk_contacts = _disk_contact_count(data, disk_geom_id)
            if rear_contacts > 0 and disk_contacts > 0:
                state.repeat_phase = "stance"
                state.repeat_phase_start = float(data.time)
                state.repeat_foot_targets = [data.site_xpos[site_id].copy() for site_id in rear_foot_site_ids]
                state.repeat_contact_lost_time = None
                for target in state.repeat_foot_targets:
                    target[2] -= args.repeat_foot_preload
                print(
                    f"repeat_stance_start index={state.repeat_count + 1} t={data.time:.3f} "
                    f"rear_contacts={rear_contacts} disk_contacts={disk_contacts}",
                    flush=True,
                )
                data.ctrl[:] = solve_rear_stance_targets(data.ctrl, 0.0)
                stage, alpha = "repeat_stance", 0.0
            else:
                alpha = transition_alpha(phase_elapsed, args.repeat_seek_time)
                data.ctrl[:] = lerp_sequence(repeat_plant_ctrl, repeat_push_ctrl, smootherstep(alpha), ctrl_ranges)
                stage = "repeat_seek"
                if alpha >= 1.0:
                    state.repeat_phase = "retract"
                    state.repeat_phase_start = float(data.time)
                    state.repeat_retract_ctrl = list(data.ctrl)
                    print(f"repeat_contact_missed index={state.repeat_count + 1} t={data.time:.3f}", flush=True)
        elif state.repeat_phase == "stance":
            alpha = transition_alpha(phase_elapsed, args.repeat_stance_time)
            data.ctrl[:] = solve_rear_stance_targets(data.ctrl, alpha)
            stage = "repeat_stance"
            rear_contacts = _body_contact_count(model, data, rear_foot_body_ids)
            disk_contacts = _disk_contact_count(data, disk_geom_id)
            if rear_contacts > 0 and disk_contacts > 0:
                state.repeat_contact_lost_time = None
            elif state.repeat_contact_lost_time is None:
                state.repeat_contact_lost_time = float(data.time)
            contact_lost = (
                state.repeat_contact_lost_time is not None
                and float(data.time) - state.repeat_contact_lost_time >= args.repeat_contact_grace
            )
            if alpha >= 1.0 or contact_lost:
                state.repeat_phase = "coast" if args.repeat_retract_delay > 0.0 else "retract"
                state.repeat_phase_start = float(data.time)
                state.repeat_retract_ctrl = list(data.ctrl)
                reason = "timeout" if alpha >= 1.0 else "contact_lost"
                print(
                    f"repeat_stance_done index={state.repeat_count + 1} t={data.time:.3f} "
                    f"duration={phase_elapsed:.3f} reason={reason}",
                    flush=True,
                )
        elif state.repeat_phase == "coast":
            alpha = transition_alpha(phase_elapsed, args.repeat_retract_delay)
            data.ctrl[:] = state.repeat_retract_ctrl
            stage = "repeat_coast"
            if alpha >= 1.0:
                state.repeat_phase = "retract"
                state.repeat_phase_start = float(data.time)
        elif state.repeat_phase == "retract":
            alpha = transition_alpha(phase_elapsed, args.repeat_retract_time)
            data.ctrl[:] = lerp_sequence(state.repeat_retract_ctrl, rolling_ctrl, smootherstep(alpha), ctrl_ranges)
            stage = "repeat_retract"
            if alpha >= 1.0:
                state.repeat_phase = "done"
        else:
            stage, alpha = "repeat_done", 1.0
            data.ctrl[:] = rolling_ctrl
            state.repeat_count += 1
            model.actuator_forcerange[:] = nominal_force_ranges
            state.repeat_start_time = None
            state.repeat_phase = None
            state.repeat_phase_start = None
            state.repeat_foot_targets = None
            state.repeat_retract_ctrl = None
            state.repeat_contact_lost_time = None
            state.repeat_prepare_targets = None
            state.repeat_touchdown_targets = None
            state.rolling_angle = 0.0
            repeat_label = "infinite" if args.repeat_pushes < 0 else args.repeat_pushes
            print(f"repeat_push_done index={state.repeat_count}/{repeat_label} t={data.time:.3f}", flush=True)
        return stage, alpha

    def update_swing_repeat():
        elapsed = float(data.time) - state.repeat_start_time
        stage, alpha = repeated_push_target(
            elapsed,
            args.repeat_prepare_time,
            args.walk_to_stand_time,
            args.stand_hold_time,
            args.stand_to_folded_time,
        )
        if stage == "repeat_prepare":
            data.ctrl[:] = lerp_sequence(rolling_ctrl, repeat_plant_ctrl, smootherstep(alpha), ctrl_ranges)
        elif stage == "repeat_push":
            data.ctrl[:] = lerp_sequence(repeat_plant_ctrl, repeat_push_ctrl, smootherstep(alpha), ctrl_ranges)
        elif stage == "repeat_hold":
            data.ctrl[:] = repeat_push_ctrl
        elif stage == "repeat_retract":
            data.ctrl[:] = lerp_sequence(repeat_push_ctrl, rolling_ctrl, smootherstep(alpha), ctrl_ranges)
        else:
            finish_repeated_push()
        return stage, alpha

    def finish_repeated_push():
        data.ctrl[:] = rolling_ctrl
        state.repeat_count += 1
        model.actuator_forcerange[:] = nominal_force_ranges
        state.repeat_start_time = None
        state.repeat_phase = None
        state.repeat_phase_start = None
        state.repeat_foot_targets = None
        state.repeat_retract_ctrl = None
        state.repeat_contact_lost_time = None
        state.repeat_prepare_targets = None
        state.repeat_touchdown_targets = None
        state.rolling_angle = 0.0
        repeat_label = "infinite" if args.repeat_pushes < 0 else args.repeat_pushes
        print(f"repeat_push_done index={state.repeat_count}/{repeat_label} t={data.time:.3f}", flush=True)

    def solve_rear_stance_targets(seed_ctrl, stance_alpha):
        desired_positions = []
        for stance_target in state.repeat_foot_targets:
            desired = stance_target.copy()
            desired[0] -= args.repeat_stance_sweep * smootherstep(stance_alpha)
            desired_positions.append(desired)
        return solve_rear_world_targets(seed_ctrl, desired_positions)

    def solve_rear_world_targets(seed_ctrl, desired_positions):
        target = [float(value) for value in seed_ctrl]
        probe_data.qpos[:] = data.qpos
        for leg_index, (actuator_1, actuator_3, site_id) in enumerate(
            ((6, 8, rear_foot_site_ids[0]), (9, 11, rear_foot_site_ids[1]))
        ):
            desired = desired_positions[leg_index]
            joint_1 = int(model.actuator_trnid[actuator_1, 0])
            joint_3 = int(model.actuator_trnid[actuator_3, 0])
            address_1 = int(model.jnt_qposadr[joint_1])
            address_3 = int(model.jnt_qposadr[joint_3])
            q1, q3 = target[actuator_1], target[actuator_3]
            for _ in range(4):
                probe_data.qpos[address_1] = q1
                probe_data.qpos[address_3] = q3
                mujoco.mj_forward(model, probe_data)
                position = probe_data.site_xpos[site_id].copy()
                error_x = float(desired[0] - position[0])
                error_z = float(desired[2] - position[2])
                if error_x * error_x + error_z * error_z < 1e-8:
                    break
                epsilon = 1e-4
                columns = []
                for address, angle in ((address_1, q1), (address_3, q3)):
                    probe_data.qpos[address] = angle + epsilon
                    mujoco.mj_forward(model, probe_data)
                    shifted = probe_data.site_xpos[site_id]
                    columns.append(((float(shifted[0]) - float(position[0])) / epsilon, (float(shifted[2]) - float(position[2])) / epsilon))
                    probe_data.qpos[address] = angle
                determinant = columns[0][0] * columns[1][1] - columns[1][0] * columns[0][1]
                if abs(determinant) < 1e-7:
                    break
                delta_1 = (error_x * columns[1][1] - columns[1][0] * error_z) / determinant
                delta_3 = (columns[0][0] * error_z - error_x * columns[0][1]) / determinant
                q1 += min(max(delta_1, -0.15), 0.15)
                q3 += min(max(delta_3, -0.15), 0.15)
                q1 = min(max(q1, ctrl_ranges[actuator_1][0]), ctrl_ranges[actuator_1][1])
                q3 = min(max(q3, ctrl_ranges[actuator_3][0]), ctrl_ranges[actuator_3][1])
            target[actuator_1], target[actuator_3] = q1, q3
        return target

    def target_rear_foot_positions(target_ctrl):
        probe_data.qpos[:] = data.qpos
        for actuator_id, target in enumerate(target_ctrl):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            qpos_address = int(model.jnt_qposadr[joint_id])
            probe_data.qpos[qpos_address] = target
        mujoco.mj_forward(model, probe_data)
        return [probe_data.site_xpos[site_id].copy() for site_id in rear_foot_site_ids]

    def target_rear_foot_height(target_ctrl):
        return min(float(position[2]) for position in target_rear_foot_positions(target_ctrl))

    def key_callback(keycode):
        if keycode == 32:
            state.push("toggle_pause")
            return
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "r":
            state.push("reset")

    def apply_command(command):
        if command == "toggle_pause":
            state.paused = not state.paused
            print(f"paused={state.paused}", flush=True)
        elif command == "reset":
            reset_simulation()
            print("reset simulation", flush=True)

    reset_simulation()
    print(f"model={args.xml_path.resolve()}", flush=True)
    if args.kp is not None or args.kd is not None or args.force_limit is not None:
        print(
            f"runtime actuator override: kp={args.kp} kd={args.kd} force_limit={args.force_limit}",
            flush=True,
        )
    if args.motion == "rear-push-roll":
        sequence = f"{args.from_keyframe} -> {args.to_keyframe} -> {args.middle_keyframe} -> {args.to_keyframe}"
    else:
        sequence = f"{args.from_keyframe} -> {args.middle_keyframe} -> {args.to_keyframe}"
    print(f"physically simulating {sequence}", flush=True)
    if args.motion == "rear-push-roll":
        print(
            f"switch_time={args.switch_time:.3f}s "
            f"front_fold_time={args.front_fold_time:.3f}s "
            f"rear_fold_time={args.rear_fold_time:.3f}s "
            f"fold_order={args.fold_order} "
            f"folded_style={args.folded_style} "
            f"rolling_pose={args.rolling_pose} "
            f"folded_hold_time={args.folded_hold_time:.3f}s "
            f"folded_to_push_time={args.walk_to_stand_time:.3f}s "
            f"push_hold_time={args.stand_hold_time:.3f}s "
            f"push_to_folded_time={args.stand_to_folded_time:.3f}s; "
            f"push_style={args.push_style} push_scale={args.push_scale:.3f}; "
            f"push_trigger_speed={args.push_trigger_speed:.3f}m/s "
            f"push_trigger_timeout={args.push_trigger_timeout:.3f}s; "
            f"repeat_pushes={args.repeat_pushes} turns_per_push={args.turns_per_push:.3f}; "
            "the folded target stays fixed while the robot rolls.",
            flush=True,
        )
    else:
        print(
            f"switch_time={args.switch_time:.3f}s "
            f"walk_to_stand_time={args.walk_to_stand_time:.3f}s "
            f"stand_hold_time={args.stand_hold_time:.3f}s "
            f"stand_to_folded_time={args.stand_to_folded_time:.3f}s; "
            "after alpha reaches 1.0 the target stays fixed.",
            flush=True,
        )

    if args.headless or args.video is not None:
        last_status_time = -float("inf")
        stage = "walk_to_stand"
        alpha = 0.0
        video_context = (
            MujocoVideoRecorder(
                model,
                args.video,
                torso_id,
                fps=args.video_fps,
                width=args.video_width,
                height=args.video_height,
                azimuth=90,
                elevation=-8,
                distance=1.4,
            )
            if args.video is not None
            else nullcontext(None)
        )
        with video_context as recorder:
            if recorder is not None:
                recorder.capture(data)
            for _ in range(max(args.steps, 0)):
                stage, alpha = step_physics()
                if recorder is not None:
                    recorder.capture(data)
                if data.time - last_status_time >= args.status_interval:
                    print(_status_line(model, data, torso_id, disk_geom_id, rear_foot_body_ids, stage, alpha), flush=True)
                    last_status_time = data.time
            if recorder is not None:
                print(f"video={recorder.output_path} frames={recorder.frame_count}", flush=True)
        print(_status_line(model, data, torso_id, disk_geom_id, rear_foot_body_ids, stage, alpha), flush=True)
        return

    print(KEY_HELP.strip(), flush=True)
    print(f"status_interval={args.status_interval:.3f}s; close the viewer window to exit.", flush=True)

    last_status_time = -float("inf")
    with viewer.launch_passive(model, data, key_callback=key_callback) as window:
        window.cam.azimuth = 90
        window.cam.elevation = -8
        window.cam.distance = 1.4
        window.cam.lookat[:] = data.xpos[torso_id]
        while window.is_running():
            with window.lock():
                for command in state.drain():
                    apply_command(command)
                stage, alpha = step_physics()
                window.cam.lookat[:] = data.xpos[torso_id]
                if data.time - last_status_time >= args.status_interval:
                    print(
                        _status_line(model, data, torso_id, disk_geom_id, rear_foot_body_ids, stage, alpha),
                        flush=True,
                    )
                    last_status_time = data.time
            window.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
