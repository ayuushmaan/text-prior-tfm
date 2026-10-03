"""TFM3.0 text extension: coherent text fields for TabICL-v2 synthetic tables.

Built on top of automl/TFM-Playground (see ../upstream/TFM-Playground).
The text layer is backbone-agnostic: it operates on batch dicts
``{"x": Tensor, "y": Tensor, ...}`` as produced by
``TabICLPriorDataLoader``, ``PriorDumpDataLoader`` or the lightweight
nanoprior, and fuses frozen text embeddings back as extra columns.
"""

from tfm3_text.semantic_schema import DOMAINS, ground_columns
from tfm3_text.text_generator import augment_playground_batch, make_text_dataset, rank_fracs, verbalize_row
from tfm3_text.text_encoder import FrozenTfidfEncoder

__all__ = [
    "DOMAINS",
    "ground_columns",
    "verbalize_row",
    "rank_fracs",
    "make_text_dataset",
    "augment_playground_batch",
    "FrozenTfidfEncoder",
]
