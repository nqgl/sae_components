from __future__ import annotations

"""
Compatibility shim.

The refactored, maintained implementation lives in `saeco.evaluation.filtered`.
This module stays to avoid import churn in downstream code.
"""

from .filtered import Filter, FilteredTensor

__all__ = ["Filter", "FilteredTensor"]