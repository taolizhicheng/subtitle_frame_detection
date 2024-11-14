import torch
import numpy as np

from practices.postprocessor.base.base_postprocessor import BasePostprocessor
from .. import POSTPROCESSOR_BUILDER


__all__ = [
    "SubtitleFrameDetectionPostprocessor",
    "SubtitleFrameDetectionInferencePostprocessor"
]


def __dir__():
    return __all__


@POSTPROCESSOR_BUILDER.register("SubtitleFrameDetectionPostprocessor")
class SubtitleFrameDetectionPostprocessor(BasePostprocessor):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(
        self,
        data: torch.Tensor,
        label: torch.Tensor,
        output: torch.Tensor,
    ):
        label_coord = label[:, :4]
        label_code = label[:, 4]

        coord = output[:, :4]
        codes_ = output[:, 4:]
        probs = torch.softmax(codes_, dim=-1)
        code = torch.argmax(probs, dim=-1)

        probs = probs.cpu().numpy()
        code = code.cpu().numpy()
        label = label.cpu().numpy()[:, 4].astype(np.int32)

        results = []
        for prob, c, l in zip(probs, code, label_code): 
            results.append((c, prob))
        return data, label, results


@POSTPROCESSOR_BUILDER.register("InferenceSubtitleFrameDetectionPostprocessor")
class InferenceSubtitleFrameDetectionPostprocessor(BasePostprocessor):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(
        self,
        data: torch.Tensor,
        label: torch.Tensor,
        output: torch.Tensor,
    ):
        coord = output[:, :4]
        coords = coord.cpu().numpy()

        codes_ = output[:, 4:]
        probs = torch.softmax(codes_, dim=-1)
        code = torch.argmax(probs, dim=-1)

        probs = probs.cpu().numpy()
        code = code.cpu().numpy()

        data = data.cpu().numpy()

        results = []
        for prob, c, coord in zip(probs, code, coords): 
            results.append((c, prob, coord))
        return data, label, results