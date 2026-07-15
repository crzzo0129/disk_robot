# Structure Variant Study

## Decision

The provisional walking/rolling candidate is:

```text
hip_y       = +/-0.090 m
leg_scale   = 0.85
disk_radius = 0.200 m
controller  = Kp 10.0, Kd 0.4, torque limit 3 Nm
```

It is stored as `assets/pupper_v3_disk_structure_candidate.xml` and is now the default
forward teacher-student training model. The original visual XML remains available as the
unscaled geometry reference.

## Model Correction

The disk cylinder is rotated so that its symmetry axis is body Y. Its explicit inertia had
the large symmetry-axis moment assigned to body X. The active and candidate XML now use:

```xml
diaginertia="0.0158632 0.03012 0.0158632"
```

For a cylinder with mass `m`, radius `r`, and full thickness `L`:

```text
Iyy = 0.5 * m * r^2
Ixx = Izz = m * (3*r^2 + L^2) / 12
```

## Geometry

| Quantity | Current | Candidate |
|---|---:|---:|
| Hip half-width | 0.070 m | 0.090 m |
| Foot support width at stand | 0.188 m | 0.221 m |
| Foot support length at stand | 0.200 m | 0.200 m |
| Kinematic leg scale | 1.00 | 0.85 |
| Disk radius | 0.200 m | 0.200 m |
| COM X relative to base | -0.025 m | -0.021 m |
| COM Z relative to base | -0.051 m | -0.048 m |

The provisional scaling keeps the motor-dominated link masses and foot sphere radii fixed,
scales link offsets and distal inertia locations, and scales the middle/distal visual meshes.
It is intended for screening. Final mass, COM, and inertia must come from the revised CAD.

## Sweep

The CPU sweep tested 27 combinations:

```text
hip_y       = 0.070, 0.085, 0.090 m
leg_scale   = 1.00, 0.90, 0.85
disk_radius = 0.200, 0.180, 0.170 m
```

All variants used the same warmed 256-point trot reference:

```text
frequency=0.8 Hz, stride=0.04 m, height=0.025 m, duty=0.72
Kp=10.0, Kd=0.4, torque limit=3 Nm, duration=8 s
```

| Variant | Net velocity | Roll/pitch RMS | Tracking RMSE | Saturation |
|---|---:|---:|---:|---:|
| Current dimensions | 0.0201 m/s | 1.32 deg | 0.0546 rad | 0.0% |
| Leg 0.90, hip 0.070 | 0.0253 m/s | 1.07 deg | 0.0497 rad | 0.0% |
| Leg 0.85, hip 0.070 | 0.0273 m/s | 0.91 deg | 0.0476 rad | 0.0% |
| **Leg 0.85, hip 0.090** | **0.0277 m/s** | **1.07 deg** | **0.0479 rad** | **0.0%** |

Shortening the leg produced the repeatable improvement. Widening the hip did not improve
the diagonal trot by itself because two-foot support still collapses to a diagonal line. It
is retained in the candidate for four-foot overlap, startup, slow walking, and disturbance
margin.

Disk radius had almost no effect while walking because the disk did not touch the floor. It
cannot be selected from walking score alone.

## Folded Envelope

The folded check measures each foot sphere's radial extent in the disk X-Z rolling plane.
A negative margin means that a foot protrudes beyond the disk circle.

| Structure | Folded radial margin |
|---|---:|
| Current, radius 0.200 | -11.3 mm |
| Candidate, radius 0.200 | -1.8 mm |
| Candidate, radius 0.170 | -31.8 mm |

The shorter leg nearly fits inside the existing rolling circle. Reducing the disk to
`0.17 m` would make rolling interference substantially worse, so the candidate keeps the
`0.20 m` radius. CAD should provide at least several millimeters of positive folded margin;
the remaining 1.8 mm can be removed by the folded joint target or distal-part design.

## Commands

### Calibrated Forward Speed

The candidate gait supports a calibrated `--target-speed` shortcut from `0` to
`0.10 m/s`. It keeps cadence at `1.2 Hz` and selects stride length from measured net
world displacement. This is an open-loop reference calibration, not closed-loop speed
tracking; the residual policy will still need to reject disturbances and model error.

View the candidate near `0.08 m/s`:

```powershell
python3.12 scripts\view_ik_gait.py --xml assets\pupper_v3_disk_structure_candidate.xml --training-reference --neutral-pose model --mode trot --target-speed 0.08 --height 0.025 --duty 0.72 --ramp 0.5 --kp 10 --kd 0.4 --torque-limit 3 --phase 0 --duration 0
```

View the current calibrated upper end near `0.10 m/s`:

```powershell
python3.12 scripts\view_ik_gait.py --xml assets\pupper_v3_disk_structure_candidate.xml --training-reference --neutral-pose model --mode trot --target-speed 0.10 --height 0.025 --duty 0.72 --ramp 0.5 --kp 10 --kd 0.4 --torque-limit 3 --phase 0 --duration 0
```

For experiments beyond the calibrated range, continue to specify `--frequency` and
`--stride` manually. Increasing frequency past roughly `1.2-1.5 Hz` did not improve net
speed and increased tracking error; increasing stride was the effective control.

### Structure Tools

View the candidate XML directly:

```powershell
python3.12 scripts/view_ik_gait.py --xml assets/pupper_v3_disk_structure_candidate.xml --training-reference --neutral-pose model --mode trot --frequency 0.8 --stride 0.04 --height 0.025 --duty 0.72 --ramp 0.5 --kp 10 --kd 0.4 --torque-limit 3 --phase 0 --duration 0
```

Preview a variant without generating another XML:

```powershell
python3.12 scripts/view_ik_gait.py --training-reference --hip-y 0.09 --leg-scale 0.85 --disk-radius 0.20 --neutral-pose model --mode trot --frequency 0.8 --stride 0.04 --height 0.025 --duty 0.72 --ramp 0.5 --kp 10 --kd 0.4 --torque-limit 3 --phase 0 --duration 0
```

Repeat the full sweep:

```powershell
python3.12 -m scripts.sweep_structure_variants --duration 8 --out docs/structure_sweep_results
```

Regenerate the candidate XML:

```powershell
python3.12 -m scripts.write_structure_candidate
```

The raw ranked results are in `docs/structure_sweep_results.csv` and
`docs/structure_sweep_results.json`.

## Next Gate

Do not start another long PPO run until the candidate passes all of the following:

1. Viewer shows sustained positive `dx`, not periodic body-frame velocity only.
2. Joint tracking and contact impact remain acceptable at `Kp=10`, the hardware limit.
3. Folded feet no longer exceed the disk rolling envelope after CAD adjustment.
4. Rear-push and rolling motions remain collision-feasible with the wider hips.
5. CAD-derived masses, COMs, and inertias replace the provisional scaled values.
