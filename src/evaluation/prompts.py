"""Shared agricultural prompt sets for the zero-shot evaluation."""

# Prompts expected to fire ON cultivated land (positive intent)
POSITIVE = [
    "cultivated agricultural field",
    "plowed field",
    "agricultural land",
    "terraced cultivated field",
    "field with cultivation rows",
    "parallel plough rows",
    "orchard",
    "abandoned cultivated field",
]

# Control prompts describing what the BLACK hard-negatives look like;
# used to probe whether a model confuses terrain for cultivation.
NEGATIVE = [
    "rocky uncultivated slope",
    "natural erosion pattern",
]

ALL_PROMPTS = POSITIVE + NEGATIVE

# CLIP / SigLIP tile-scoring prompts (contrastive positive vs negative sets)
CLIP_POSITIVE = [
    "cultivated agricultural land",
    "plowed agricultural field",
    "field with cultivation rows",
    "terraced farmland",
]
CLIP_NEGATIVE = [
    "rocky terrain",
    "natural erosion",
    "barren slope",
    "dirt road",
]

# Detection confidence thresholds to sweep (post-hoc filtering)
BOX_THRESHOLDS = [0.15, 0.25, 0.35]
