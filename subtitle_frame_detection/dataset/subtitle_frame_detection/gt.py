import os
import cv2
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from practices.dataset.base.base_dataset import BaseDataset
from .. import DATASET_BUILDER
from .video import VideoFrameDataset


__all__ = ["GTSubtitleFrameDetectionDataset"]
def __dir__():
    return __all__


@DATASET_BUILDER.register("GTSubtitleFrameDetectionDataset")
class GTSubtitleFrameDetectionDataset(BaseDataset):
    def __init__(
        self,
        video_dir: str,
        intervals: int,
        label_dir: str,
        roi: List[float],
        **kwargs
    ):
        super().__init__(
            video_dir=video_dir,
            intervals=intervals,
            label_dir=label_dir,
            roi=roi,
            **kwargs
        )
    
    def build_data(
        self,
        video_dir: str,
        intervals: int,
        label_dir: str,
        roi: List[float],
        **kwargs
    ):
        self.video_dir = video_dir
        self.video_paths = glob.glob(os.path.join(video_dir, "**/*.mp4"), recursive=True)
        self.video_paths = sorted(self.video_paths)
        self.intervals = intervals
        self.label_dir = label_dir
        self.label_paths = glob.glob(os.path.join(label_dir, "**/*.json"), recursive=True)
        self.label_paths = sorted(self.label_paths)
        self.video = None
        self.labels = None
        self.images = None
        self.roi = roi
        self.load_data(0)
    
    def load_data(self, index: int):
        self.load_labels(index)
        self.load_video(index)
    
    def load_video(self, index: int):
        video_path = self.video_paths[index]
        self.video = VideoFrameDataset(video_path, self.intervals)

        x1, x2, y1, y2 = self.roi
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

        self.images = np.zeros((len(self.labels), (y2 - y1), (x2 - x1), 3), dtype=np.uint8)
        for i in range(len(self.labels)):
            label = self.labels[i]
            rect = label["rect"]
            text = label["text"]
            time = label["time"]

            frame, label = self.video[time]
            frame = frame[y1:y2, x1:x2]
            self.images[i] = frame
    
    def load_labels(self, index: int):
        label_path = self.label_paths[index]

        with open(label_path, 'r') as f:
            self.labels = json.load(f)
    
    def __len__(self):
        return len(self.labels) - 1
    
    def get_data(self, index: int):
        label1 = self.labels[index]
        rect1 = label1["rect"]
        text1 = label1["text"]
        time1 = label1["time"]

        label2 = self.labels[index + 1]
        rect2 = label2["rect"]
        text2 = label2["text"]
        time2 = label2["time"]

        frame1 = self.images[index]
        frame2 = self.images[index + 1]

        x1, _, y1, _ = self.roi
        if x1 > 1 or y1 > 1:
            x1 = int(x1)
            y1 = int(y1)
        else:
            x1 = int(x1 * self.video.width)
            y1 = int(y1 * self.video.height)

        if rect2 is not None:
            sx, sy, w, h = rect2
            cx = sx + w / 2 - x1
            cy = sy + h / 2 - y1

        if text1 is None and text2 is None:
            label = [-100, -100, -100, -100, 0]
        elif text1 is None and text2 is not None:
            label = [cx, cy, w, h, 0]
        elif text1 is not None and text2 is None:
            label = [-100, -100, -100, -100, 0]
        elif text1 == text2:
            label = [cx, cy, w, h, 1]
        else:
            label = [cx, cy, w, h, 0]

        label = np.array(label, dtype=np.float32)
        
        return (frame1, frame2), label

    def show_data(self, index: int):
        (frame1, frame2), label = self.get_data(index)

        cx, cy, w, h, code = label
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        draw = frame2.copy()
        if x1 > 0:
            draw = cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)

        plt.imshow(frame1[:, :, ::-1], cmap="gray")
        plt.show()
        plt.imshow(draw[:, :, ::-1], cmap="gray")
        plt.show()
