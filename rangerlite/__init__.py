import importlib.metadata

from .rangerlite import RangerLite

try:
    __version__ = importlib.metadata.version("rangerlite")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
__all__ = ["RangerLite", "__version__"]
