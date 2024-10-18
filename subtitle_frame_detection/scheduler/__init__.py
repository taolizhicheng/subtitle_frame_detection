import os

from practices.scheduler import SCHEDULER_BUILDER
from practices.utils.build import build_indices


__all__ = ["SCHEDULER_BUILDER"]


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
build_indices(f"{THIS_DIR}/subtitle_frame_detection")