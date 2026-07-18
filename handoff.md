# Disk Robot Handoff

Updated: 2026-07-18

Canonical continuation file for a new Codex conversation.

## 1. Goal

The stable fixed-speed forward Student milestone has been achieved:

1. Build a symmetric IK gait around the XML `stand` keyframe.
2. Train a privileged PPO Teacher that outputs residual joint-position corrections around IK.
3. Distill the Teacher into a phase-conditioned Student with BC.
4. Deploy only the Student. The accepted Student uses a controller-owned phase clock, but
   needs no runtime IK, foot-contact input, privileged observation, or previous action.

The next goal is command-conditioned control for a joystick, beginning with a one-dimensional
`vx` range and stop/start transitions. Longer term, expand to acceptable `[vx, vy, wz]`
commands and rolling. The fixed forward gait is only an initial anchor and must not become
the final controller limitation.

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
- Student BC and diagnostic DAgger are supervised learning, not RL. The accepted T8 Student
  comes from BC; DAgger is not required for this fixed-speed artifact.
- Student outputs a complete position action relative to `stand`, not a residual.
- The deployed T8 observation is 147-dimensional: four frames of physical sensor/command
  history without previous action, plus `sin(phase), cos(phase), gait_blend`.
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

## 6. Current Local Work And Validation

The local tree contains the T7 diagnosis tools and T8 structural observation ablation. The
important implementation points are:

- `student_previous_action_input=False` produces a 147-dimensional Student observation;
- the accepted 231-dimensional Teacher contract remains unchanged;
- T8 policy/dataset artifacts have distinct names and stage metadata;
- the feedback diagnosis supports both the failed 195-dimensional T5 policy and the accepted
  147-dimensional T8 policy;
- the DAgger entry refuses to silently treat T8 artifacts as the older T5/T6 contract.

The latest local regression result after these changes is:

```text
116 passed
```

The repository contains unrelated user work. Always inspect the current tree and do not
revert or overwrite changes outside the task.

## 7. Cloud Experiment Results

Cloud environment:

```text
offline Linux instance
environment: mjx312
H200 nodes: MuJoCo GL egl when available
RTX 4090 node without EGL: pass --mujoco-gl disable
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
- trains only the phase-free T3 Student by behavior cloning;
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

## 12. T4b Phase-Preserving DAgger Smoke: Update Still Too Aggressive

The T4b smoke used `teacher_blend=0.50`, but only 20 updates at learning rate `1e-4`
reduced nominal/disturbed velocity from `0.0340/0.0303` to `0.0036/-0.0009`. The corrected
fallback selector retained round 0, so no collapsed policy was selected. Phase-preserving
collection alone is insufficient when each supervised update can move the policy this far.

## 13. T4c Conservative DAgger Result: Failed

The paired-seed T4c smoke still degraded the Student:

```text
                         round 0      round 1
nominal vx               0.0334       0.0126
disturbed vx             0.0333       0.0196
nominal failure_rate     0.062        0
disturbed recovery       0.983 s      1.465 s
```

The lower learning rate and anchor prevented an immediate collapse to exact standing, but
velocity and recovery still worsened under identical evaluation conditions. Do not run a
formal T4c or keep tuning the phase-free feed-forward Student.

## 14. T5 IK-Free Phase-Conditioned Student Result: Failed

T5 keeps the accepted Teacher parameter contract unchanged:

```text
Teacher observation             231
Student sensor history          192
Student internal clock state      3
Student policy observation      195
```

The three controller-owned values are:

```text
sin(phase), cos(phase), gait_blend
```

They require no foot-contact sensor, ground reaction force, IK, or Teacher at runtime. The
phase advances from the controller clock at the accepted reference frequency (`1.2 Hz` for
the current `vx=0.08` experiment). When the command enters the deadzone, phase freezes and
`gait_blend` ramps back toward stand. `disk_robot/phase_clock.py` provides the matching NumPy
runtime implementation for later hardware control.

The formal T5 run `mjx_runs/student_t5_phase_bc_seed0` also fit its offline dataset nearly
exactly but moved backward in closed loop:

```text
final BC loss                    0.0000045
nominal mean_velocity_x         -0.0076
disturbed mean_velocity_x       -0.0092
nominal failure_rate             0.004
disturbed failure_rate           0.004
accepted                         False
```

Adding phase did not by itself solve closed-loop imitation. Static code audit confirms that
the Student phase observation and Teacher direct-action label are generated from the same
environment state, so do not assume a simple one-control-step bug without dynamic evidence.

## 15. T5 Oracle Direct-Action Diagnosis: Passed

`scripts/diagnose_phase_student.py` performs a read-only, paired-seed comparison of:

```text
Teacher residual control
Teacher online labels executed as direct Student actions
Learned T5 phase-conditioned Student
```

Run:

```bash
python -m scripts.diagnose_phase_student --teacher-run mjx_runs/teacher_t2a_seed0 --student-run mjx_runs/student_t5_phase_bc_seed0
```

The actual diagnosis result was:

```text
teacher_residual vx     0.0817
oracle_direct vx        0.0817
learned_student vx     -0.0071
oracle preserves vx     True
```

This rules out a phase-timing mismatch, Teacher-to-Student action conversion bug, and invalid
direct-position control interface. Executing the Teacher's online direct-action labels through
the Student actuator path preserves the Teacher motion. The remaining failure is learned
Student closed-loop covariate drift.

## 16. T6 Phase-Conditioned DAgger Smoke: Failed

All previous T4 experiments used the old phase-free T3 Student. The 195-dimensional T5 Student
has not previously received DAgger data from its own closed-loop states. The DAgger entry now
auto-detects the source artifact:

```text
T3_BC          -> T4_DAGGER       192 observations
T5_PHASE_BC    -> T6_PHASE_DAGGER 195 observations
```

For T6 it:

- loads `student_policy_phase_bc.npz` and `student_phase_bc_dataset.npz`;
- reconstructs the exact T5 phase-conditioned environment config from policy metadata;
- validates that the policy, normalization statistics, saved dataset, and environment all use
  the 195-dimensional observation contract;
- rolls out the phase-conditioned Student and asks the frozen accepted Teacher for online
  labels at the same phase;
- retains conservative Teacher blending, `1e-5` learning rate, per-round anchoring, paired
  evaluation seeds, and best-round fallback selection;
- saves `student_policy_phase_dagger.npz` with stage `T6_PHASE_DAGGER`;
- requires no foot-contact input or IK at Student runtime.

The T6 technical smoke correctly loaded the 195-dimensional T5 policy:

```bash
stage=t6_source ... obs=195 phase_conditioned=True
```

Its paired results were:

```text
                         round 0      round 1
nominal vx               -0.0106      -0.0076
disturbed vx             -0.0003      -0.0053
disturbed post error      0.0948       0.1265
disturbed recovery        1.216 s      1.396 s
score                    -8.56997     -8.75342
```

The selector correctly retained round 0. Do not run formal T6. Phase-conditioned DAgger did
not restore forward motion and worsened disturbed recovery, so another BC/DAgger tuning pass
is not the current task.

The previous conservative DAgger implementation remains available for diagnostics:

- loads the frozen accepted Teacher and the existing T3 Student instead of reinitializing;
- requires the T3 dataset saved by `--save-dataset`;
- rolls out in both nominal and disturbed environments with an annealed Teacher-action blend;
- asks the Teacher to label the states actually visited by the Student;
- aggregates new labels with the original BC dataset;
- anchors each round to that round's starting Student action function;
- uses a `1e-5` default learning rate and configurable `--anchor-weight` (default `1.0`);
- evaluates every DAgger round and saves each round separately;
- uses identical paired evaluation seeds for every round so policy deltas are comparable;
- selects an accepted round first, otherwise the highest-scoring round;
- uses no Teacher blend during evaluation or in the deployed Student;
- never updates or overwrites Teacher or T3 artifacts.

## 17. T7 Result: Failure Root-Cause Audit

Do not train a new policy until the failure mechanism is measured. The read-only
`scripts/audit_phase_student_failure.py` audit covers four questions:

1. **Offline error structure:** per-joint bias/RMSE, physical target error in radians, and
   phase-binned error on the original 131,072-sample T5 dataset.
2. **Oracle approximation margin:** execute online Teacher direct actions with persistent
   episode bias and stepwise Gaussian errors at action RMS levels
   `0, 0.001, 0.002, 0.005, 0.01`.
3. **Paired closed-loop divergence:** from identical deployment-style phase-zero resets, run
   oracle and Student branches in parallel and measure the first joint/torso divergence,
   Student-vs-online-Teacher correction error, and how the Teacher label changes after the
   Student leaves the oracle trajectory.
4. **Identifiability evidence:** among nearby deployable observations from different
   environments, compare Teacher-label disagreement with deterministic Student-action
   disagreement.

The audit never updates or overwrites a policy. It writes:

```text
mjx_runs/student_t5_phase_bc_seed0/failure_audit.json
mjx_runs/student_t5_phase_bc_seed0/failure_audit_trace.npz
```

Run the technical smoke:

```bash
python -m scripts.audit_phase_student_failure --teacher-run mjx_runs/teacher_t2a_seed0 --student-run mjx_runs/student_t5_phase_bc_seed0 --smoke
```

If the smoke completes, run the full audit:

```bash
python -m scripts.audit_phase_student_failure --teacher-run mjx_runs/teacher_t2a_seed0 --student-run mjx_runs/student_t5_phase_bc_seed0
```

Interpret the result before choosing another learning algorithm:

- failure at action RMS near the Student offline RMSE means the oracle gait has inadequate
  approximation margin;
- a dominant joint or phase-localized bias points to the action representation/loss;
- very early physical divergence plus rapidly growing Teacher-label drift indicates a
  high-gain, narrow-attraction-basin target;
- materially higher Teacher disagreement than Student disagreement for nearby deployable
  observations is evidence that privileged action distillation is not identifiable.

The full audit showed that the Oracle tolerates action noise well, while the T5 Student error
amplifies from `0.00206` at step 0 to `0.01951` at step 1 and `0.11323` at step 2. Physical
joint divergence crosses `0.01 rad` at step 2. Teacher label drift remains much smaller than
the Student error. This rules out a bad reset action and points to excessive local closed-loop
gain in the Student observation/action feedback path.

`scripts/diagnose_phase_student_feedback.py` performs the next read-only diagnosis. It:

- evaluates the Student on the Oracle trajectory and its own trajectory separately;
- measures policy-output shift independently from Teacher-label drift;
- replaces one observation group at a time with its paired Oracle value;
- reports how much each replacement recovers the Oracle policy output;
- computes the local Student Jacobian gain for latest/all previous actions, joint position,
  joint velocity, IMU/body velocity, command, and phase groups.

Run:

```bash
python -m scripts.diagnose_phase_student_feedback --teacher-run mjx_runs/teacher_t2a_seed0 --student-run mjx_runs/student_t5_phase_bc_seed0
```

The script defaults to `MUJOCO_GL=disable`, 16 environments, and 12 paired steps. No H200 is
required and it does not train or modify the policy. Focus on `stage=feedback_group`: a large
positive `recovered` value identifies the observation group that causally accounts for the
Student policy shift; `jacobian_spectral > 1` for `latest_previous_action` is direct evidence
of an unstable autoregressive shortcut.

The actual feedback diagnosis established the root cause:

```text
Student oracle-manifold error                  0.002--0.006
closed-loop error, step 1 / step 2             0.016 / 0.090
previous-action-history recovered fraction     0.901
latest-previous-action recovered fraction      0.547
previous-action-history Jacobian spectral gain 59.832
latest-previous-action Jacobian spectral gain  22.668
```

The T5 Student accurately represents the Teacher on the Oracle trajectory, but it learned an
unstable autoregressive shortcut from the highly correlated previous-action history. Its own
small action error enters the next observation and is amplified each step. Teacher-label drift
is too small to explain the effect. This is not a Teacher, phase, direct-action-interface, or
reset-distribution failure.

## 18. T8 Result: Previous-Action Removal Passed

T8 is a single-variable causal ablation. It keeps the accepted Teacher observation and
parameters unchanged but removes all four 12-value previous-action blocks from the Student
policy observation:

```text
Teacher raw sensor history                  192
Teacher privileged observation             231
T8 Student deployable sensor history        144
T8 Student internal phase state               3
T8 Student policy observation               147
```

Teacher data trajectories, direct-action labels, random seeds, network hidden layers, BC
updates, nominal/disturbed split, and evaluation gates remain identical to T5. Only the
Student input contract changes. The previous-action values remain in the raw Teacher history,
so accepted Teacher parameters are still compatible; they are structurally absent from the
Student network rather than merely zeroed.

The technical smoke completed on the 4090 node. As expected, 20 BC updates were insufficient
for policy quality and were used only to validate compilation and artifact interfaces.

The formal controlled run was:

```bash
python -m scripts.distill_phase_student_no_previous_action --teacher-run mjx_runs/teacher_t2a_seed0 --save-dataset --strict-acceptance --mujoco-gl disable --out mjx_runs/student_t8_phase_bc_no_previous_action_seed0
```

Formal result:

```text
final BC loss                    0.0000164
nominal reward                   1.6194
nominal mean_velocity_x          0.0823
nominal failure_rate             0.000
nominal roll/pitch RMS           0.2120
disturbed reward                 1.5822
disturbed mean_velocity_x        0.0785
disturbed failure_rate           0.000
disturbed post-push error        0.0338
disturbed recovery               0.413 s
accepted                         True
```

Relative to the paired Teacher, nominal velocity changed by only `+0.0006 m/s`; disturbed
velocity changed by `+0.0007 m/s`, recovery improved by `0.040 s`, and disturbed distance
changed by `+0.0072 m`. T8 therefore preserves both nominal and recovery behavior.

The same feedback diagnosis was then repeated:

```bash
python -m scripts.diagnose_phase_student_feedback --teacher-run mjx_runs/teacher_t2a_seed0 --student-run mjx_runs/student_t8_phase_bc_no_previous_action_seed0
```

The T8 first-ten-step closed-loop action error stayed near `0.0025--0.0102`, policy shift near
`0.0017--0.0051`, and joint-position RMSE reached only `0.00347` at step 9. The explosive T5
feedback is gone. Joint-velocity history still explains some normal corrective behavior, but
its measured Jacobian spectral gain is only `0.182`, not an unstable previous-action loop.

This closes the causal chain:

1. T5 fitted Teacher-forced data accurately;
2. its own previous action entered the next observation;
3. the network assigned that channel extremely high local gain;
4. small deployment errors recursively amplified within two control steps;
5. deleting only that channel removed the divergence and passed acceptance.

The full reasoning, rejected hypotheses, numerical evidence, and reusable debug procedure are
recorded in `docs/student_imitation_failure_debugging.md`.

T8 artifacts use stage `T8_PHASE_BC_NO_PREVIOUS_ACTION` and policy file
`student_policy_phase_bc_no_previous_action.npz`. This is the frozen fixed-speed deployment
candidate. Do not overwrite it, and do not run DAgger merely because earlier stages planned it.

## 19. Next Task: Characterize T8, Then Build T9 Variable Forward Speed

The next product goal is joystick control, but do not jump directly to full omnidirectional
training and do not spend an open-ended tuning cycle polishing only `vx=0.08`. Use the
accepted T8 policy as a frozen regression anchor, perform one short characterization pass,
then expand only the one-dimensional forward-speed command.

### 19.1 Short T8 Characterization Gate

The current aggregate nominal result is:

```text
forward distance    0.7914 m
lateral distance    0.0444 m
drift ratio         5.62 percent
drift angle         3.21 degrees
mean abs vy         0.0302 m/s
mean abs yaw rate   0.1243 rad/s
```

This is a usable fixed-speed baseline with mild visible drift, not evidence that T8 must be
retrained. The aggregate signed lateral distance cannot distinguish a systematic one-sided
bias from different seeds drifting in opposite directions.

The immediate implementation task is a T8-compatible visualization/trajectory diagnostic,
not another policy update. It should:

1. load `student_policy_phase_bc_no_previous_action.npz` and reconstruct its 147-dimensional
   config from metadata;
2. save a single-rollout side/tracking view for gait inspection;
3. save a fixed top view or XY trajectory plot for straightness;
4. report per-environment final lateral displacement, absolute displacement, standard
   deviation, maximum drift, path angle, yaw, disk contact, failure, and action/force limits;
5. compare identical-seed IK, accepted T2a Teacher, and T8 Student trajectories;
6. run several seeds and a longer horizon to detect slowly accumulating yaw.

The current RTX 4090 environment uses `--mujoco-gl disable`, which cannot render. Prefer a
split workflow: run MJX rollout and save qpos/trajectory data on the cloud node, then render
the saved trajectory locally with MuJoCo. Do not force EGL on that node.

Decision gate:

- if IK, Teacher, and T8 have comparable drift and no collision/saturation/long-horizon
  instability, freeze T8 and proceed;
- if only T8 is materially worse than the paired Teacher, diagnose Student retention before
  expanding commands;
- if all policies show a repeatable one-sided drift, treat it as a model/IK/reward symmetry
  issue and carry the straightness metric into T9 rather than blindly tuning BC.

### 19.2 T9 One-Dimensional Command Conditioning

After the short gate, train multiple straight-line speeds before adding yaw or lateral motion.
Start with discrete anchors such as:

```text
vx = 0.00, 0.04, 0.06, 0.08, 0.10 m/s
vy = 0
wz = 0
```

First keep one command fixed for each episode so the policy learns the steady command-action
mapping. After the speed grid passes, add within-episode stop, start, acceleration, and
deceleration transitions.

T9 requirements:

1. freeze and never overwrite T2a/T8 artifacts;
2. make the IK reference and Teacher data collection genuinely command-conditioned;
3. preserve the T8 no-previous-action Student contract;
4. prefer four frames of physical sensor history plus one current command and current phase,
   rather than repeating command in every history frame;
5. evaluate every speed independently for speed error, lateral drift, failure, disk contact,
   yaw, disturbed recovery, and retention at `vx=0.08` relative to T8;
6. add command counterfactual and transition tests so a large command Jacobian is not mistaken
   for a learned command response.

Do not connect raw joystick commands to the current fixed-speed T8 artifact. Its command input
was constant during training. The controller phase clock is compatible with joystick control:
the command can regulate gait frequency/stride/blend while the clock coordinates the legs.

Only after one-dimensional `vx` and stop/start transitions pass should the task expand in this
order:

```text
forward speed range -> yaw -> backward motion -> lateral velocity -> joint [vx, vy, wz]
```

## 20. Commands And Workflow Notes

Run local tests from `disk_robot/`:

```powershell
python -m pytest -q
```

The local bundled Python has MuJoCo but not JAX/Brax, so full MJX smoke must run in `mjx312`.

Use single-line Windows and Linux commands. The user explicitly does not want PowerShell
backtick continuation syntax. The cloud cannot access the internet, so dependencies and code
must already be synchronized before training.

The repository may contain user changes. Do not revert unrelated modifications. The latest
local test result after the T8 structural ablation implementation is `116 passed`.
