"""Concrete tokenization jobs for statically declared datasets."""

from . import blockwise as blockwise
from . import streaming as streaming

__all__ = ["blockwise", "streaming"]
