from importlib.metadata import version, PackageNotFoundError

from .image_quality_callback import ImageQualityCallback

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
