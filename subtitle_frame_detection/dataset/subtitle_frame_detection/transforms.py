import torch
import numpy as np
from typing import List
from practices.dataset.base.base_transform import BaseTransform

from .. import TRANSFORM_BUILDER


__all__ = [
    "SubtitleFrameDetectionNoise",
    "SubtitleFrameDetectionNormalize",
    "SubtitleFrameDetectionConcat",
    "SubtitleFrameDetectionTranspose",
    "SubtitleFrameDetectionToTensor"
]

def __dir__():
    return __all__


@TRANSFORM_BUILDER.register("SubtitleFrameDetectionNoise")
class SubtitleFrameDetectionNoise(BaseTransform):
    def __init__(self, lower_noise: float, upper_noise: float):
        super().__init__()
        self.lower_noise = lower_noise
        self.upper_noise = upper_noise

    def __call__(self, data, label):
        if len(data) == 3:
            frame1, frame2, positions = data
        else:
            frame1, frame2 = data

        frame1 = frame1.astype(int)
        frame2 = frame2.astype(int)
        noise1 = np.random.randint(self.lower_noise, self.upper_noise + 1)
        noise2 = np.random.randint(self.lower_noise, self.upper_noise + 1)

        frame1 = frame1 + noise1
        frame2 = frame2 + noise2

        frame1 = np.clip(frame1, 0, 255)
        frame2 = np.clip(frame2, 0, 255)

        if len(data) == 3:
            return (frame1, frame2, positions), label

        return (frame1, frame2), label


@TRANSFORM_BUILDER.register("SubtitleFrameDetectionNormalize")
class SubtitleFrameDetectionNormalize(BaseTransform):
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, data, label):
        if len(data) == 3:
            frame1, frame2, positions = data
        else:
            frame1, frame2 = data

        frame1 = (frame1 - self.mean) / self.std
        frame2 = (frame2 - self.mean) / self.std

        if len(data) == 3:
            return (frame1, frame2, positions), label

        return (frame1, frame2), label


@TRANSFORM_BUILDER.register("SubtitleFrameDetectionConcat")
class SubtitleFrameDetectionConcat(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data, label):
        data = np.concatenate(data, axis=2)
        return data, label


@TRANSFORM_BUILDER.register("SubtitleFrameDetectionTranspose")
class SubtitleFrameDetectionTranspose(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data, label):
        data = data.transpose(2, 0, 1)
        return data, label


@TRANSFORM_BUILDER.register("SubtitleFrameDetectionToTensor")
class SubtitleFrameDetectionToTensor(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data, label):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        elif isinstance(label, (list, tuple)):
            label = torch.tensor(label)
        else:
            raise TypeError(f"Unsupported label type: {type(label)}")

        data = data.float()
        label = label.float()

        return data, label
