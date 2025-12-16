"""
MABe Challenge Python package.
"""

from .data_loads import (
    load_meta_df,
    load_train_ann,
    get_tracking,
    get_whitelist,
    expand_ann_to_frames,
    merge_labels_tracks,
)
from .features import normalize_xy
# Other modules can be imported directly when needed to avoid circular dependencies

__all__ = [
    # Data loading
    "load_meta_df",
    "load_train_ann",
    "get_tracking",
    # Whitelist
    "get_whitelist",
    # Label expansion
    "expand_ann_to_frames",
    # Preprocessing
    "merge_labels_tracks",
    "normalize_xy",
]

