"""A lightweight, type hinted PyTorch implementation of the Lookahead optimizer with RAdam as its inner optimizer."""

import importlib.metadata

from .rangerlite import RangerLite

try:
    __version__ = importlib.metadata.version("rangerlite")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
__author__ = "Thorsten Buss"
__all__ = ["RangerLite", "__author__", "__version__"]
