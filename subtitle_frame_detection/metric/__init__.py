import os

from practices.metric import METRIC_BUILDER
from practices.utils.build import build_indices


__all__ = ["METRIC_BUILDER"]


def __dir__():
    return __all__


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
build_indices(f"{THIS_DIR}/subtitle_frame_detection")