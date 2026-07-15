# Disk Robot Handoff

Updated: 2026-07-15

## 1. Goal

The immediate goal is a stable fixed-speed forward Student for the disk Pupper:

1. Build a symmetric IK gait around the XML `stand` keyframe.
2. Train a privileged PPO Teacher that outputs residual joint-position corrections around IK.
3. Distill the Teacher into a gait-free Student with BC and DAgger.
4. Deploy only the Student. The Student must not need IK phase or privileged contact data.

Longer term, expand to arbitrary acceptable `[vx, vy, wz]` commands and rolling. The fixed
forward gait is only an initial anchor and must not become the final controller limitation.

## 2. Active Model And Controller

Teacher-Student training now defaults to:

```text
assets/pupper_v3_disk_structure_candidate.xml
```

Candidate structure:

```text
hip_y       = +/-0.090 m
leg_scale   = 0.85
disk_radius = 0.200 m
Kp          = 10.0
Kd          = 0.4
torque      = 3 Nm
```

The original unscaled visual model remains:

```text
assets/pupper_v3_disk_visual.xml
```

Important: `scripts/view_ik_gait.py` still defaults to the original visual XML, while the
Teacher-Student environment defaults to the structure candidate. Pass `--xml` explicitly
when viewing the candidate.

## 3. IK Speed Calibration

`disk_robot/gait_speed.py` maps a fixed forward speed command to the candidate gait.
The current training command is `0.08 m/s`, resolved to approximately:

```text
mode      = trot
frequency = 1.2 Hz
stride    = 0.0742135 m
height    = 0.025 m
duty      = 0.72
```

Local physics results:

```text
target 0.08 m/s -> actual about 0.082 m/s
target 0.10 m/s -> actual about 0.099 m/s
```

Both ran without falling, disk contact, or torque saturation. Higher speed increased body
angular motion and contact impact. Increasing cadence beyond about 1.2-1.5 Hz gave little
speed benefit; stride was the effective speed control.

Viewer command:

```powershell
python3.12 scripts\view_ik_gait.py --xml assets\pupper_v3_disk_structure_candidate.xml --training-reference --neutral-pose model --mode trot --target-speed 0.08 --height 0.025 --duty 0.72 --ramp 0.5 --kp 10 --kd 0.4 --torque-limit 3 --phase 0 --duration 0
```

## 4. Current Pipeline Semantics

- IK is handcrafted and not trained.
- Teacher PPO is the only reinforcement-learning stage.
- Teacher action is a normalized residual around the IK joint target.
- Student BC and DAgger are supervised learning, not RL.
- Student outputs a complete position action relative to `stand`, not a residual.
- Student currently observes simulated body linear velocity. Real deployment still needs
  an equivalent estimator or later observation removal/noise/distillation work.
- World-forward velocity and displacement are reward/evaluation quantities, not Student
  actor inputs.

The default fixed command is `0.08 m/s`. This first Student is not yet command-randomized.

## 5. Important Fixes Already In The Codebase

1. Candidate XML is the default Teacher-Student training model.
2. `command_vx` automatically selects the calibrated IK reference unless
   `--ik-speed-mode manual` is used.
3. `mean_velocity_error` now means `abs(mean_velocity_x - command_vx)`.
4. `mean_instantaneous_velocity_error` separately reports trot-cycle speed pulsation.
5. `velocity_sigma` was relaxed from `0.0004` to `0.01` so normal trot pulsation does not
   make the speed reward almost always zero.
6. PPO checkpoint parameters are paired only with evaluation metrics from the same step.
   The old callback used metrics from the preceding evaluation and selected the wrong params.
7. PPO and zero-residual IK are reevaluated on the same seeds before selection.
8. `--teacher-only` stops before BC/DAgger.
9. `--residual-scale-multiplier` scales Teacher correction limits without changing IK or
   Student action semantics.
10. By default, zero-residual IK cannot silently pass as a privileged PPO Teacher. If PPO is
    not selected, strict mode rejects with `ppo_teacher_did_not_outperform_ik_baseline`.

## 6. Latest Uncommitted Work

At handoff, `git status --short` shows only:

```text
M disk_robot_mjx/teacher_student_env.py
M scripts/train_forward_teacher_student.py
```

These changes close a newly identified acceptance loophole: a policy could have acceptable
world-X velocity while moving rapidly in Y or rotating in yaw.

The patch adds:

```text
mean_lateral_distance
mean_abs_velocity_y
mean_abs_yaw_rate
```

Default hard limits:

```text
mean_abs_velocity_y <= 0.03 m/s
mean_abs_yaw_rate   <= 0.25 rad/s
```

It also requires PPO not to regress baseline lateral speed by more than `0.01 m/s` or yaw
rate by more than `0.05 rad/s` during nominal preservation checks.

Local tests after this patch:

```text
71 passed
```

The lateral/yaw patch has not yet been run in cloud MJX smoke.

## 7. Cloud Experiment Results

Cloud environment:

```text
offline Linux instance
environment: mjx312
MuJoCo GL: egl
```

### Full Residual PPO Run

Run directory:

```text
mjx_runs/forward_008_v2
```

IK baseline:

```text
failure_rate                    0.0
mean_velocity_x                 0.0837425 m/s
mean_velocity_error             0.0037425 m/s
mean_instantaneous_error        0.0795349 m/s
mean_roll_pitch_rate_rms        0.220498 rad/s
mean_forward_distance           0.837425 m
```

Training reward peaked near 0.98M reported steps, then collapsed and partially recovered.
The tail plateaued around 0.82 and did not exceed the earlier 0.996 peak. This is PPO
instability/forgetting, not evidence that simply extending training will solve the task.

The old callback mismatch selected params from the wrong step. This is now fixed.

### T1 Conservative Residual Run

Settings used:

```text
teacher_steps              1.5M
learning_rate              3e-5
entropy_cost               1e-4
updates_per_batch          2
residual_scale_multiplier  0.25
teacher_only               true
```

Result: failed nominal preservation.

Fair comparison:

```text
                         IK baseline       PPO best (step 327680)
reward_per_step          1.542404          1.478387
mean_velocity_x          0.083791          0.067052
velocity error           0.003791          0.012948
instantaneous error      0.079477          0.073575
roll/pitch rate RMS      0.220237          0.228130
failure rate             0                 0
```

Interpretation:

- PPO kept the robot upright and barely changed roll/pitch motion.
- It slightly smoothed instantaneous speed.
- It reduced mean forward speed by about 20 percent.
- The failure is real, not a threshold bug.
- In the undisturbed nominal environment, zero residual is already better. PPO is being
  asked to learn a no-op, but random policy initialization and exploration keep damaging IK.

Do not distill this Teacher into a Student. Less-informed supervised Student training cannot
invent recovery behavior that the Teacher did not demonstrate.

## 8. Next Task: Implement T1b

The previous user message was `do it`, but the turn was interrupted before T1b implementation
started. T1b is therefore NOT implemented yet.

T1b is a short sanity gate, not a final Teacher:

```text
nominal state: residual approximately zero
joint target:  approximately the original IK target
```

Implement the following:

1. Start the PPO actor at or very near zero deterministic residual. Inspect the installed
   Brax `0.10+` policy-network API in the cloud before relying on initializer arguments;
   `requirements-mjx.txt` does not pin an exact Brax version.
2. Low-pass filter Teacher residual commands. A proposed update is
   `filtered = previous + alpha * (command - previous)` with configurable `alpha` around
   `0.1-0.2`. Use the filtered residual consistently in environment stepping, Teacher
   observation/history, reward penalties, and DAgger label conversion.
3. Make zero-residual regularization configurable. Start around `penalty_residual=0.2`
   instead of the current `0.02`; keep or slightly strengthen residual-rate penalty.
4. Add a nominal selection mode such as `--teacher-selection-mode preserve`. It may select
   PPO when it remains within baseline tolerances even if its scalar score is microscopically
   lower. This mode must require `--teacher-only`; it must never unlock Student distillation.
5. Use no entropy or very low entropy, learning rate around `1e-5`, one PPO update per batch,
   residual multiplier `0.25`, and only `100k-300k` steps.
6. T1b passes only if PPO remains close to baseline:

```text
mean_velocity_x          >= baseline - 0.01 m/s
roll_pitch_rate_rms      <= baseline + 0.10 rad/s
failure_rate             <= baseline + 0.02
mean_abs_velocity_y      <= baseline + 0.01 m/s
mean_abs_yaw_rate        <= baseline + 0.05 rad/s
```

T1b does not prove a useful privileged Teacher. It only proves PPO can avoid destroying IK.

## 9. After T1b: T2 Privileged Disturbance Teacher

Once T1b preserves IK, add disturbances that create a reason for nonzero residuals:

- random forward/lateral pushes;
- motor-strength variation;
- control delay;
- friction variation;
- small mass/COM variation;
- larger reset pose and joint perturbations.

Evaluate zero-residual IK and PPO on identical nominal and disturbed seeds. A valid Teacher:

1. does not materially regress nominal IK performance; and
2. significantly improves disturbed failure rate, recovery time, post-push velocity error,
   disk contact rate, and forward displacement.

Only after this dual gate passes should BC and DAgger run. T2 is where privileged phase,
contact, IK tracking error, and disturbance information can become useful.

## 10. Commands And Workflow Notes

Run local tests from `disk_robot/`:

```powershell
python -m pytest -q
```

The local bundled Python has MuJoCo but not JAX/Brax, so full MJX smoke must run in `mjx312`.

Current short Teacher experiment command, before T1b changes:

```bash
python -m scripts.train_forward_teacher_student --teacher-only --teacher-steps 1500000 --teacher-evals 11 --teacher-learning-rate 3e-5 --teacher-entropy-cost 1e-4 --teacher-updates-per-batch 2 --residual-scale-multiplier 0.25 --strict-acceptance --out mjx_runs/teacher_r025_seed0
```

Do not rerun this unchanged; it already failed and T1b must be implemented first.

Use single-line Windows and Linux commands. The user explicitly does not want PowerShell
backtick continuation syntax. The cloud cannot access the internet, so dependencies and code
must already be synchronized before training.

The repository may contain user changes. Do not revert unrelated modifications. The latest
working-tree changes are the lateral/yaw metrics described above.

