import torch
import numpy as np

from practices.postprocessor.base.base_postprocessor import BasePostprocessor
from .. import POSTPROCESSOR_BUILDER


@POSTPROCESSOR_BUILDER.register("SubtitleFrameDetectionPostprocessor")
class SubtitleFrameDetectionPostprocessor(BasePostprocessor):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(
        self,
        data: torch.Tensor,
        output: torch.Tensor,
    ):
        coordinates = output[:, :4].cpu().numpy()
        scores = output[:, 4]
        scores = torch.sigmoid(scores)
        scores = scores.cpu().numpy()

        results = []
        for coordinate, score in zip(coordinates, scores):
            cx, cy, w, h = coordinate
            if cx < 0 or cy < 0 or w < 0 or h < 0:
                cx = cy = w = h = -100

            results.append([cx, cy, w, h, score])

        results = np.array(results)
        return results
