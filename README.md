                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # Disk Robot

MuJoCo model and phase-1 tools for an extreme disk-body quadruped. The robot keeps a normal quadruped layout, while the torso is a thin disk/cylinder: side view is close to a circle, front view remains narrow.

## Project Layout

- `assets/`: MJCF/XML model assets
- `scripts/`: viewer, keyboard control, pose interpolation, and diagnostics
- `tests/`: lightweight regression tests for the phase-1 model and tools
- `docs/`: current training specification, design notes, commands, and TODOs

## Quick Start

From this project root:

```powershell
python -m pip install -r requirements.txt
python -m scripts.diagnose_extreme_disk_pose --keyframe all
python -m scripts.view_extreme_disk_pose --keyframe stand
python -m scripts.view_extreme_disk_pose --keyframe folded
python -m scripts.control_extreme_disk_flex --keyframe folded
python -m scripts.interpolate_extreme_disk_pose
python -m scripts.simulate_extreme_disk_pose
python -m scripts.simulate_extreme_disk_pose --switch-time 0.5 --walk-to-stand-time 2.0 --stand-hold-time 0.5 --stand-to-folded-time 3.0
python -m scripts.simulate_extreme_disk_pose --headless --steps 1000
python -m scripts.walk_smoke --steps 100 --policy zero
python -m scripts.walk_smoke --steps 100 --policy random
```

The current phase-1 baseline is `assets/disk_quadruped_extreme.xml` with two manually calibrated keyframes:

- `stand`: temporary standard standing pose.
- `folded`: temporary folded pose for rolling preparation.

The XML keeps 12 independent position actuators. Paired front/rear flex control is implemented only in Python scripts, not as physical actuator coupling in the model.

## Current Notes

- `stand` first frame has four foot-ground contacts, but the foot spheres start with penetration into the floor contact.
- `folded` first frame places the disk torso on the ground and folds the feet upward, with foot-torso internal contacts to watch during later collision design.
- Before walking training, start from `docs/README.md` and
  `docs/omnidirectional_training_pipeline.md`.

## Walk Training Status

The active model is `assets/pupper_v3_disk_visual.xml`. The forward training path uses a
symmetric IK reference, a privileged residual PPO teacher, behavior cloning, and DAgger to
produce a student that runs without IK or gait phase inputs.

Run the complete pipeline in the offline Linux `mjx312` environment:

```bash
python -m scripts.train_forward_teacher_student --smoke --out mjx_runs/forward_ts_smoke
python -m scripts.train_forward_teacher_student --out mjx_runs/forward_ts --strict-acceptance
```

The offline instance must already have the packages in `requirements-mjx.txt` available in
the active `mjx312` environment.

The final deployable artifact is `mjx_runs/forward_ts/student_policy.npz`. See
`docs/forward_teacher_student.md` for the observation contract, stages, outputs, evaluation,
and checkpoint recovery command. The older `mjx_train_walk` entry remains available for
environment regression only; it is not the active teacher-student pipeline.

The default MuJoCo GL backend for cloud training is `egl`; use `--mujoco-gl osmesa` only on machines with a working OSMesa stack.
