from pkg_resources import DistributionNotFound, get_distribution

from .image_quality_callback import ImageQualityCallback

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
