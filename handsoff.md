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

## 8. T1b Result: Passed

T1b was implemented with zero deterministic actor output initialization, residual low-pass
filtering, configurable residual penalties, and `--teacher-selection-mode preserve`.

Cloud result from `mjx_runs/teacher_t1b_seed0`:

```text
                         IK baseline       selected PPO
mean_velocity_x          0.083693          0.083194
failure_rate             0                 0
roll/pitch rate RMS      0.220249          0.216421
mean_abs_velocity_y      0.032373          0.031815
mean_abs_yaw_rate        0.157255          0.151094
instantaneous error      0.079449          0.077681
reward_per_step          1.542272          1.570872
```

PPO preserved forward speed and improved every reported stability metric. T1b therefore
passed. The old absolute lateral cap rejected the run only because the cap was `0.03 m/s`
while IK itself measured `0.03237 m/s`. Preserve-mode acceptance now uses the relative T1b
gate and reports the absolute threshold result separately.

The run also exposed Brax timestep rounding. With batch size 256, 32 minibatches, unroll 20,
and six evaluations, a requested 200k run executed 819,200 steps. The training script now
prints `stage=teacher_step_plan`. Using eight minibatches produces approximately 204,800
steps for the same request.

T1b still does not prove a useful privileged Teacher. It proves PPO can avoid destroying IK.

## 9. T2a Result: Passed

T2a was implemented with:

- one randomized forward/lateral root-velocity push per episode;
- episode-level approximate motor-strength variation;
- randomized one-control-step command delay;
- larger joint and height reset perturbations;
- four additional privileged Teacher observations: last push XY, motor-strength deviation,
  and delay state. Student observations remain unchanged;
- post-push velocity error, recovery time, disk contact, push coverage, and displacement
  metrics;
- paired nominal and disturbed IK/PPO evaluation using identical seeds;
- `--teacher-selection-mode robust`, which requires both nominal preservation and disturbed
  improvement before selecting PPO.

Friction, mass, and COM randomization are intentionally deferred to T2b because they require
per-environment MJX model randomization. Do not add them until T2a push recovery is calibrated.

After the first smoke showed one-step recovery and an almost unaffected IK baseline, the
default push was raised from `0.20/0.15` to `0.50/0.40 m/s`. Recovery now uses an EMA of
world XY velocity (`alpha=0.10`) and requires four consecutive control steps within
`0.04 m/s` forward and lateral tolerances. Raw post-push instantaneous error is still
reported separately.

The formal cloud run `mjx_runs/teacher_t2a_seed0` passed both robust gates:

```text
selected_source=ppo
selected_step=1,024,000
nominal_preserved=True
disturbed_improved=True
accepted=True
params=mjx_runs/teacher_t2a_seed0/teacher/params
```

This is the first accepted privileged disturbance Teacher and is the frozen source for T3.
Do not retrain or overwrite it while distilling the Student.

## 10. T3 Result: Offline Fit Passed, Closed-Loop Retention Failed

T3 is implemented as a separate entry point, `scripts/distill_forward_student.py`. It:

- requires an accepted PPO Teacher run and refuses rejected or IK-baseline runs;
- reconstructs the exact Teacher config and IK reference from `run_config.json`;
- loads `teacher/params` without running PPO or changing Teacher parameters;
- collects a shuffled 50/50 nominal and disturbed demonstration dataset;
- uniformly subsamples across each full rollout instead of truncating to early episode steps;
- trains only the gait-free Student by behavior cloning;
- evaluates Student separately in nominal and disturbed environments;
- reports post-push error, recovery time, push coverage, and retention relative to Teacher;
- saves `student_policy_bc.npz`, `student_policy_bc.json`, and `evaluation.json`;
- contains no DAgger collection. DAgger remains T4.

Run the cloud end-to-end smoke first:

```bash
python -m scripts.distill_forward_student --teacher-run mjx_runs/teacher_t2a_seed0 --smoke --out mjx_runs/student_t3_bc_smoke_seed0
```

Smoke is an interface and compilation check only; 20 BC updates are not enough to judge
Student quality. It should reach `stage=t3_acceptance` and save an artifact even if
`accepted=False`.

After smoke succeeds, run full T3:

```bash
python -m scripts.distill_forward_student --teacher-run mjx_runs/teacher_t2a_seed0 --out mjx_runs/student_t3_bc_seed0 --save-dataset --strict-acceptance
```

Read the compact terminal output in this order:

1. `stage=t3_teacher` confirms the frozen PPO Teacher and selected step.
2. `stage=t3_dataset_plan` confirms the nominal/disturbed sample split.
3. `stage=student_bc_result` gives nominal and disturbed Student metrics.
4. `stage=student_bc_retention` gives Student-minus-Teacher deltas.
5. `stage=t3_acceptance` is the T3 decision.

The formal T3 run `mjx_runs/student_t3_bc_seed0` produced an excellent offline fit but failed
closed-loop retention:

```text
final BC loss                         0.000006
nominal mean_velocity_x               0.0290
nominal failure_rate                  0.008
nominal roll/pitch rate RMS           1.0668
disturbed mean_velocity_x             0.0263
disturbed failure_rate                0.016
disturbed post-push velocity error    0.1028
disturbed recovery time               1.449 s
accepted                              False
```

The Student nearly exactly predicts Teacher actions on the demonstration dataset, but its own
rollout leaves that dataset and degrades. More BC updates on the same fixed data are not the
right next step. This is the intended trigger for T4 DAgger, not evidence that the accepted
Teacher should be retrained.

## 11. T4 Pure-Student DAgger Result: Failed By Phase Collapse

The first formal T4 run used pure Student rollouts. Both DAgger rounds converged to a stable
standing policy:

```text
round 1 nominal/disturbed vx    -0.0007 / -0.0005
round 2 nominal/disturbed vx    -0.0008 / -0.0009
round 2 roll/pitch rate RMS      0.0045 / 0.0135
round 2 accepted                 False
```

The DAgger supervised loss plateaued around `0.02`, versus `0.000006` for T3 BC. Once the
Student stands still, nearly identical phase-free Student observations receive different
cyclic labels as the privileged Teacher phase advances. MSE averages those conflicting labels
toward the stand pose. More pure-Student DAgger rounds will reinforce this collapse.

The old fallback score also favored standing because the environment gives standing roughly
`1.6` reward per step. T4 fallback selection now ignores reward and strongly prioritizes
velocity retention. Do not deploy `mjx_runs/student_t4_dagger_seed0/student_policy_dagger.npz`.

## 12. Current Task: T4b Phase-Preserving DAgger

`scripts/dagger_forward_student.py` now:

- loads the frozen accepted Teacher and the existing T3 Student instead of reinitializing;
- requires the T3 dataset saved by `--save-dataset`;
- rolls out in both nominal and disturbed environments with an annealed Teacher-action blend;
- asks the Teacher to label the states actually visited by the Student;
- aggregates new labels with the original BC dataset;
- evaluates every DAgger round and saves each round separately;
- selects an accepted round first, otherwise the highest-scoring round;
- uses no Teacher blend during evaluation or in the deployed Student;
- never updates or overwrites Teacher or T3 artifacts.

Run a new cloud smoke without overwriting the failed T4 run:

```bash
python -m scripts.dagger_forward_student --teacher-run mjx_runs/teacher_t2a_seed0 --bc-run mjx_runs/student_t3_bc_seed0 --smoke --out mjx_runs/student_t4b_dagger_smoke_seed0
```

If smoke reaches `stage=t4_acceptance`, formal T4b should use three rounds with blend
`0.50 -> 0.30 -> 0.10`:

```bash
python -m scripts.dagger_forward_student --teacher-run mjx_runs/teacher_t2a_seed0 --bc-run mjx_runs/student_t3_bc_seed0 --dagger-rounds 3 --teacher-rollout-blend-start 0.5 --teacher-rollout-blend-end 0.1 --out mjx_runs/student_t4b_dagger_seed0 --save-dataset --strict-acceptance
```

## 13. Commands And Workflow Notes

Run local tests from `disk_robot/`:

```powershell
python -m pytest -q
```

The local bundled Python has MuJoCo but not JAX/Brax, so full MJX smoke must run in `mjx312`.

Use single-line Windows and Linux commands. The user explicitly does not want PowerShell
backtick continuation syntax. The cloud cannot access the internet, so dependencies and code
must already be synchronized before training.

The repository may contain user changes. Do not revert unrelated modifications. The latest
local tests after the T4b phase-preserving collection change are `101 passed`.
