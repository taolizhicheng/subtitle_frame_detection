import os

from practices.model import MODEL_BUILDER
from practices.utils.build import build_indices


__all__ = ["MODEL_BUILDER"]


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
build_indices(f"{THIS_DIR}/subtitle_frame_detection")
