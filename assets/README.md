# Robot XML files

| File | Status | Purpose |
| --- | --- | --- |
| `pupper_v3_disk_structure_candidate.xml` | **Active** | Default for simulation, IK/Teacher preview, training, and evaluation. |
| `pupper_v3_disk_visual.xml` | Source / rolling prototype | Unscaled source for structure/COM sweeps. It also retains the specialized `rear_push` and `rolling_folded` keyframes used by pose-transition rolling tools. Do not use as the normal training default. |
| `disk_quadruped_extreme.xml` | Legacy | Early standalone extreme-disk pose and flex-control prototype. |
| `disk_quadruped_extreme_train.xml` | Legacy | Training-oriented variant of the early standalone prototype. |

Canonical paths live in `disk_robot/model_paths.py`. To test a non-default model, pass
`--xml` or `--xml-path` explicitly. The active candidate is generated from the source model
by `python -m scripts.write_structure_candidate`; generation must never use the candidate
as its own source because that would apply structural scaling twice.

The legacy files retain their names so old scripts, experiment metadata, and documentation
remain reproducible.
