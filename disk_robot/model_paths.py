"""Canonical MJCF paths for the disk-robot project.

Keep model selection here so training, evaluation, and viewers cannot silently drift to
different robot geometries.  Structure sweeps intentionally use ``BASE_MODEL_XML`` and
write their accepted result to ``ACTIVE_MODEL_XML``.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"

# Current Pupper-based model used by normal simulation, training, and evaluation.
ACTIVE_MODEL_XML = ASSETS_DIR / "pupper_v3_disk_structure_candidate.xml"

# Unscaled source model used only as the input to structure/COM sweeps and regeneration.
BASE_MODEL_XML = ASSETS_DIR / "pupper_v3_disk_visual.xml"

# The pose-transition rolling prototype still relies on keyframes that are not present
# in the active training candidate (notably ``rear_push`` and ``rolling_folded``).
ROLLING_PROTOTYPE_XML = BASE_MODEL_XML

# Earlier standalone prototypes.  Kept for their dedicated legacy tools and tests.
LEGACY_EXTREME_XML = ASSETS_DIR / "disk_quadruped_extreme.xml"
LEGACY_EXTREME_TRAIN_XML = ASSETS_DIR / "disk_quadruped_extreme_train.xml"


__all__ = [
    "ACTIVE_MODEL_XML",
    "ASSETS_DIR",
    "BASE_MODEL_XML",
    "LEGACY_EXTREME_TRAIN_XML",
    "LEGACY_EXTREME_XML",
    "PROJECT_ROOT",
    "ROLLING_PROTOTYPE_XML",
]
