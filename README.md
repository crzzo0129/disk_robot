                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # Disk Robot

MuJoCo model and phase-1 tools for an extreme disk-body quadruped. The robot keeps a normal quadruped layout, while the torso is a thin disk/cylinder: side view is close to a circle, front view remains narrow.

## Project Layout

- `assets/`: MJCF/XML model assets
- `scripts/`: viewer, keyboard control, pose interpolation, and diagnostics
- `tests/`: lightweight regression tests for the phase-1 model and tools
- `docs/`: current training specification, design notes, commands, and TODOs

## Which XML is current?

Use **`assets/pupper_v3_disk_structure_candidate.xml`** for normal simulation, training,
and evaluation. Runtime defaults are centralized in `disk_robot/model_paths.py`.

- `pupper_v3_disk_visual.xml` is the unscaled source used by structure/COM sweeps and by
  specialized rolling-keyframe prototypes that are not part of the active training model.
- `disk_quadruped_extreme.xml` and `disk_quadruped_extreme_train.xml` are retained legacy
  prototypes used only by their dedicated pose/flex tools.
- See `assets/README.md` for the complete model-file contract.

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

The legacy phase-1 pose prototype is `assets/disk_quadruped_extreme.xml` with two manually calibrated keyframes:

- `stand`: temporary standard standing pose.
- `folded`: temporary folded pose for rolling preparation.

The XML keeps 12 independent position actuators. Paired front/rear flex control is implemented only in Python scripts, not as physical actuator coupling in the model.

## Current Notes

- `stand` first frame has four foot-ground contacts, but the foot spheres start with penetration into the floor contact.
- `folded` first frame places the disk torso on the ground and folds the feet upward, with foot-torso internal contacts to watch during later collision design.
- Before walking training, start from `docs/README.md` and
  `docs/omnidirectional_training_pipeline.md`.

## Walk Training Status

The active training model is `assets/pupper_v3_disk_structure_candidate.xml`. The accepted
fixed-speed path uses a symmetric IK reference, a privileged residual PPO Teacher, and
behavior cloning to produce a Student that runs without IK, contact inputs, privileged
observations, or previous action. The Student uses a controller-owned phase clock.

The frozen accepted runs are:

```text
Teacher: mjx_runs/teacher_t2a_seed0
Student: mjx_runs/student_t8_phase_bc_no_previous_action_seed0
```

The offline instance must already have the packages in `requirements-mjx.txt` available in
the active `mjx312` environment.

The fixed-speed deployable candidate is
`mjx_runs/student_t8_phase_bc_no_previous_action_seed0/student_policy_phase_bc_no_previous_action.npz`.
See `docs/forward_teacher_student.md` for the current contract and
`docs/student_imitation_failure_debugging.md` for the T3--T8 failure diagnosis. The older
`mjx_train_walk` and DAgger entries remain available for regression/diagnostics; they are not
the accepted fixed-speed path.

Use `egl` where it is available. On the current RTX 4090 node without EGL, pass
`--mujoco-gl disable` for headless training and diagnostics.

The next gate is the read-only T8 long-horizon trajectory characterization. Run the MJX
rollout on the cloud node without a GL context:

```bash
python -m scripts.characterize_t8_trajectories --teacher-run mjx_runs/teacher_t2a_seed0 --student-run mjx_runs/student_t8_phase_bc_no_previous_action_seed0 --mujoco-gl disable
```

After copying/synchronizing the generated `trajectory_rollouts.npz`, render its paired IK,
Teacher, and T8 tracking views on a local MuJoCo machine. `--xml-path` can replace a cloud
absolute path stored in the rollout:

```powershell
python -m scripts.characterize_t8_trajectories --mode render --rollout-data mjx_runs\student_t8_phase_bc_no_previous_action_seed0\t8_trajectory_characterization\trajectory_rollouts.npz --xml-path assets\pupper_v3_disk_structure_candidate.xml
```

The saved rollout can also be analyzed without JAX or rerunning MJX:

```bash
python -m scripts.characterize_t8_trajectories --mode analyze --rollout-data mjx_runs/student_t8_phase_bc_no_previous_action_seed0/t8_trajectory_characterization/trajectory_rollouts.npz
```

If T8 is worse only over the longer horizon, separate fixed BC bias from closed-loop drift
with the non-training long-horizon feedback audit:

```bash
python -m scripts.diagnose_phase_student_feedback --teacher-run mjx_runs/teacher_t2a_seed0 --student-run mjx_runs/student_t8_phase_bc_no_previous_action_seed0 --envs 16 --steps 1500 --summary-windows 0 5 10 20 30 --mujoco-gl disable
```

To reprint the window summary from an existing report without JAX or another rollout:

```bash
python -m scripts.diagnose_phase_student_feedback --report-in mjx_runs/student_t8_phase_bc_no_previous_action_seed0/feedback_long_horizon_diagnosis.json
```

## T9 Forward-Speed Grid

T9 begins with episode-fixed forward commands at `0.00/0.04/0.06/0.08/0.10 m/s`. Run the
new Teacher smoke before any formal training:

```bash
python -m scripts.train_t9_forward_teacher --smoke --out mjx_runs/teacher_t9_vx_grid_smoke_seed0 --mujoco-gl disable
```

The preliminary Teacher is deliberately not accepted until the independent per-speed grid
gate passes. Full commands and artifact semantics are recorded in `handoff.md` section 19.2.
