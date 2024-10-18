import os

from practices.dataset import DATASET_BUILDER, PREPROCESSOR_BUILDER, TRANSFORM_BUILDER
from practices.utils.build import build_indices


__all__ = ["DATASET_BUILDER", "PREPROCESSOR_BUILDER", "TRANSFORM_BUILDER"]

def __dir__():
    return __all__


THIS_DIR = os.path.abspath(os.path.dirname(__file__))


build_indices(f"{THIS_DIR}/subtitle_frame_detection")