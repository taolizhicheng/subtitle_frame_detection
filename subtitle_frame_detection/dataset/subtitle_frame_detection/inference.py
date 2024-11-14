import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from practices.dataset.base.base_dataset import BaseDataset
from .. import DATASET_BUILDER

from .video import VideoFrameDataset


__all__ = ["InferenceSubtitleFrameDetectionDataset"]
def __dir__():
    return __all__


@DATASET_BUILDER.register("InferenceSubtitleFrameDetectionDataset")
class InferenceSubtitleFrameDetectionDataset(BaseDataset):
    def __init__(
        self, 
        video_path: str,
        intervals: int,
        select_range: List[float],
        image_width: int,
        image_height: int,
        **kwargs
    ):
        super().__init__(
            video_path=video_path,
            intervals=intervals,
            select_range=select_range,
            image_width=image_width,
            image_height=image_height,
            **kwargs
        )

    def build_data(
        self, 
        video_path: str,
        intervals: int,
        select_range: List[float],
        image_width: int,
        image_height: int,
    ):
        self.video_path = video_path
        self.intervals = intervals
        self.select_range = select_range
        self.image_width = image_width
        self.image_height = image_height

        self.video = VideoFrameDataset(video_path, intervals)

        x1, x2, y1, y2 = select_range
        if x1 > 1 or x2 > 1 or y1 > 1 or y2 > 1:
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
        else:
            x1 = int(x1 * self.video.width)
            x2 = int(x2 * self.video.width)
            y1 = int(y1 * self.video.height)
            y2 = int(y2 * self.video.height)

        self.images = np.zeros((len(self.video), self.image_height, self.image_width, 3), dtype=np.uint8)

        w, h = x2 - x1, y2 - y1
        scale = min(self.image_width / w, self.image_height / h)
        nh, nw = int(h * scale), int(w * scale)

        for i in range(len(self.video)):
            frame, label = self.video[i]
            frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, (nw, nh))
            self.images[i, :nh, :nw] = frame
    
    def get_data(self, index: int):
        frame1 = self.images[index]
        frame2 = self.images[index + 1]

        label = [-100, -100, -100, -100, 0]

        return (frame1, frame2), label

    def __len__(self):
        return len(self.images) - 1

    def show_data(self, index: int):
        (frame1, frame2), label = self.get_data(index)

        plt.imshow(frame1[:, :, ::-1], cmap="gray")
        plt.show()
        plt.imshow(frame2[:, :, ::-1], cmap="gray")
        plt.show()
