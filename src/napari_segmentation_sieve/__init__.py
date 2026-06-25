try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import MorphologyTools, SegmentationSieve, watershed_widget

__all__ = ("SegmentationSieve", "MorphologyTools", "watershed_widget")
