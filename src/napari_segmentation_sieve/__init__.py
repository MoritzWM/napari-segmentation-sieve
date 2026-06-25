try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import (
    MorphologyTools,
    SegmentationSieve,
    threshold_widget,
    watershed_widget,
)

__all__ = (
    "SegmentationSieve",
    "MorphologyTools",
    "watershed_widget",
    "threshold_widget",
)
