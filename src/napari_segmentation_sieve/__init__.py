try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import (
    SegmentationSieve,
    arithmetics,
    filters_widget,
    merge_labels_widget,
    morphology_widget,
    split_labels_widget,
    threshold_widget,
    watershed_widget,
)

__all__ = (
    "SegmentationSieve",
    "morphology_widget",
    "watershed_widget",
    "threshold_widget",
    "arithmetics",
    "filters_widget",
    "split_labels_widget",
    "merge_labels_widget",
)
