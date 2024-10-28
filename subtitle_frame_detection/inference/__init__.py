import os

from practices.inference import INFERENCE_BUILDER
from practices.utils.build import build_indices


__all__ = ["INFERENCE_BUILDER"]

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
build_indices(f"{THIS_DIR}/subtitle_frame_detection")