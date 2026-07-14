"""Physically simulate a robot through staged pose transitions."""
import argparse
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--walk-to-stand-time", type=float, help="Seconds to move from the first pose to the middle pose.")
    parser.add_argument("--stand-hold-time", type=float, help="Seconds to hold the middle pose.")
    parser.add_argument("--stand-to-folded-time", type=float, help="Seconds to move from the middle pose to the final pose.")
    parser.add_argument("--kp", type=float, help="Optional runtime position gain override.")
    parser.add_argument("--kd", type=float, help="Optional runtime velocity gain override.")
    parser.add_argument("--force-limit", type=float, help="Optional symmetric runtime actuator force limit.")
    parser.add_argument("--headless", action="store_true", help="Run without the interactive viewer.")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps for --headless.")
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
        args.folded_hold_time = 0.1 if args.folded_hold_time is None else args.folded_hold_time
        args.walk_to_stand_time = 0.2 if args.walk_to_stand_time is None else args.walk_to_stand_time
        args.stand_hold_time = 0.15 if args.stand_hold_time is None else args.stand_hold_time
        args.stand_to_folded_time = 0.18 if args.stand_to_folded_time is None else args.stand_to_folded_time
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


def step_target_alpha(sim_time, switch_time, transition_time):
    return transition_alpha(sim_time - switch_time, transition_time)


def format_status_line(sim_time, stage, alpha, torso_x, torso_y, torso_z, contact_count, disk_contact_count):
    return (
        f"t={sim_time:.2f} stage={stage} alpha={alpha:.2f} "
        f"x={torso_x:.3f} y={torso_y:.3f} z={torso_z:.3f} "
        f"contacts={contact_count} disk_contacts={disk_contact_count}"
    )


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


def _status_line(data, torso_id, disk_geom_id, stage, alpha):
    torso_pos = data.xpos[torso_id]
    return format_status_line(
        sim_time=float(data.time),
        stage=stage,
        alpha=alpha,
        torso_x=float(torso_pos[0]),
        torso_y=float(torso_pos[1]),
        torso_z=float(torso_pos[2]),
        contact_count=int(data.ncon),
        disk_contact_count=_disk_contact_count(data, disk_geom_id),
    )


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
    front_fold_ctrl = to_ctrl[:6] + from_ctrl[6:]
    ctrl_ranges = _control_ranges(model)
    torso_id = _body_id(mujoco, model, args.track_body)
    disk_geom_id = _geom_id(mujoco, model, args.disk_geom)
    state = PlaybackState()

    def reset_simulation():
        state.switched = False
        mujoco.mj_resetDataKeyframe(model, data, from_id)
        data.ctrl[:] = from_ctrl
        mujoco.mj_forward(model, data)

    def update_target():
        stage, alpha = current_target()
        if not state.switched:
            if stage == "front_fold":
                data.ctrl[:] = lerp_sequence(from_ctrl, front_fold_ctrl, alpha, ctrl_ranges)
            elif stage == "rear_fold":
                data.ctrl[:] = lerp_sequence(front_fold_ctrl, to_ctrl, alpha, ctrl_ranges)
            elif stage == "folded_hold":
                data.ctrl[:] = to_ctrl
            elif stage == "folded_to_push":
                data.ctrl[:] = lerp_sequence(to_ctrl, middle_ctrl, alpha, ctrl_ranges)
            elif stage == "push_hold":
                data.ctrl[:] = middle_ctrl
            elif stage == "push_to_folded":
                data.ctrl[:] = lerp_sequence(middle_ctrl, to_ctrl, alpha, ctrl_ranges)
            elif stage == "walk_to_stand":
                data.ctrl[:] = lerp_sequence(from_ctrl, middle_ctrl, alpha, ctrl_ranges)
            elif stage in ("stand_hold", "stand_to_folded"):
                fold_alpha = 0.0 if stage == "stand_hold" else alpha
                data.ctrl[:] = lerp_sequence(middle_ctrl, to_ctrl, fold_alpha, ctrl_ranges)
            else:
                data.ctrl[:] = lerp_sequence(middle_ctrl, to_ctrl, 1.0, ctrl_ranges)
        finished_stage = "rolling" if args.motion == "rear-push-roll" else "folded"
        if not state.switched and stage == finished_stage:
            state.switched = True
            print(f"finished target transition to {args.to_keyframe} at t={data.time:.3f}", flush=True)
        return stage, alpha

    def current_target():
        if args.motion == "rear-push-roll":
            return rear_push_roll_target(
                data.time,
                args.switch_time,
                args.front_fold_time,
                args.rear_fold_time,
                args.folded_hold_time,
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
            stage, alpha = update_target()
            mujoco.mj_step(model, data)
        return stage, alpha

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
            f"folded_hold_time={args.folded_hold_time:.3f}s "
            f"folded_to_push_time={args.walk_to_stand_time:.3f}s "
            f"push_hold_time={args.stand_hold_time:.3f}s "
            f"push_to_folded_time={args.stand_to_folded_time:.3f}s; "
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

    if args.headless:
        last_status_time = -float("inf")
        stage = "walk_to_stand"
        alpha = 0.0
        for _ in range(max(args.steps, 0)):
            stage, alpha = step_physics()
            if data.time - last_status_time >= args.status_interval:
                print(_status_line(data, torso_id, disk_geom_id, stage, alpha), flush=True)
                last_status_time = data.time
        print(_status_line(data, torso_id, disk_geom_id, stage, alpha), flush=True)
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
                    print(_status_line(data, torso_id, disk_geom_id, stage, alpha), flush=True)
                    last_status_time = data.time
            window.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
