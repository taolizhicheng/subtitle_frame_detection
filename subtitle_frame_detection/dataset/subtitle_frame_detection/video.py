import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

from practices.dataset.base.base_dataset import BaseDataset
from .. import DATASET_BUILDER


__all__ = ["VideoFrameDataset", "VideoFrameTime"]
def __dir__():
    return __all__


def skip_frames(video: cv2.VideoCapture, frames: int):
    for _ in range(frames):
        video.grab()


class VideoFrameTime:
    @staticmethod
    def string_time_to_seconds(time: str) -> float:
        time = time.strip().split(':')
        if len(time) == 2:
            minutes, seconds = time
            minutes = int(minutes)
            seconds = float(seconds)
            seconds = round(minutes * 60 + seconds, 3)
        elif len(time) == 3:
            hours, minutes, seconds = time
            hours = int(hours)
            minutes = int(minutes)
            seconds = float(seconds)
            seconds = round(hours * 3600 + minutes * 60 + seconds, 3)
        else:
            raise ValueError("Invalid time format")

        return seconds
    
    @staticmethod
    def seconds_to_string_time(time: float) -> str:
        hours = int(time // 3600)
        time %= 3600
        minutes = int(time // 60)
        seconds = time % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    @staticmethod
    def frame_index_to_seconds(frame_index: int, fps: float) -> float:
        seconds = round(frame_index / fps, 3)
        return seconds
    
    @staticmethod
    def seconds_to_frame_index(seconds: float, fps: float) -> int:
        frame_index = int(round(seconds * fps))
        return frame_index

    @staticmethod
    def string_time_to_frame_index(time: str, fps: float) -> int:
        seconds = VideoFrameTime.string_time_to_seconds(time)
        frame_index = VideoFrameTime.seconds_to_frame_index(seconds, fps)
        return frame_index
    
    @staticmethod
    def frame_index_to_string_time(frame_index: int, fps: float) -> str:
        seconds = VideoFrameTime.frame_index_to_seconds(frame_index, fps)
        time = VideoFrameTime.seconds_to_string_time(seconds)
        return time

    @staticmethod
    def from_string_time(time: str, fps: float):
        return VideoFrameTime(time, fps)

    @staticmethod
    def from_seconds(seconds: float, fps: float):
        return VideoFrameTime(seconds, fps)

    @staticmethod
    def from_frame_index(frame_index: int, fps: float):
        seconds = VideoFrameTime.frame_index_to_seconds(frame_index, fps)
        return VideoFrameTime(seconds, fps)

    def __init__(
        self, 
        time: Union[float, str],
        fps: float
    ):
        if isinstance(time, str):
            self._seconds = self.string_time_to_seconds(time)
        elif isinstance(time, float):
            self._seconds = time
        else:
            try:
                time = float(time)
                self._seconds = time
            except:
                raise ValueError(f"Invalid time format: {time}")
    
        self._fps = fps
        self._frame_index = int(round(self._seconds * self._fps))
        self._time = self.seconds_to_string_time(self._seconds)
    
    def __str__(self):
        info = f"Time: {self.time}, Frame: {self.frame_index}"
        return info
    
    def __repr__(self):
        return str(self)
    
    @property
    def time(self):
        return self._time

    @property
    def frame_index(self):
        return self._frame_index

    @property
    def seconds(self):
        return self._seconds
    
    @property
    def fps(self):
        return self._fps


@DATASET_BUILDER.register("VideoFrameDataset")
class VideoFrameDataset(BaseDataset):
    def __init__(
        self, 
        video_path      : str,
        intervals       : int = 1,
        start_time      : str = None,
        end_time        : str = None,
        **kwargs
    ):
        super().__init__(
            video_path = video_path,
            intervals = intervals,
            start_time = start_time,
            end_time = end_time,
            **kwargs
        )

    def __len__(self):
        start_frame_index = self._start_time.frame_index
        end_frame_index = self._end_time.frame_index
        total_frames = (end_frame_index - start_frame_index) // self._intervals
        return total_frames
        
    def __iter__(self):
        self._set_start_frame_index(self._start_time.frame_index)
        for index in range(len(self)):
            ret, image = self._video.read()
            if not ret:
                raise ValueError("Frame not read!")
            
            self._cache_frame = image
            yield image,

            self._current_frame_index += self._intervals
            skip_frames(self.video, self._intervals - 1)
    
    def __next__(self):
        frame_index = self._current_frame_index + self._intervals
        if frame_index >= self._end_time.frame_index:
            raise StopIteration
        
        self._set_start_frame_index(frame_index)
        ret, image = self._video.read()
        if not ret:
            raise ValueError("Frame not read!")
        
        self._cache_frame = image
        return image, 

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._video.release()

    def build_data(
        self, 
        video_path: str,
        intervals: int,
        start_time: str,
        end_time: str
    ):
        self._video_path = video_path
        self._video = cv2.VideoCapture(self._video_path)
        if not self._video.isOpened():
            raise ValueError("Failed to open video file: {self._video_path}")

        self._fps = self._video.get(cv2.CAP_PROP_FPS)
        self._width = int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frames = int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

        if start_time is None:
            start_time = "00:00:00.000"
        if end_time is None:
            end_time = VideoFrameTime.frame_index_to_seconds(self._frames, self._fps)
            end_time = VideoFrameTime.seconds_to_string_time(end_time)
        
        self._start_time = VideoFrameTime.from_string_time(start_time, self._fps)
        self._end_time = VideoFrameTime.from_string_time(end_time, self._fps)

        self._intervals = intervals
        self._current_frame_index = 0
        self._set_start_frame_index(self._start_time.frame_index)
        self._cache_frame = None
    
    def get_data(self, index):
        if isinstance(index, str):
            seconds = VideoFrameTime.string_time_to_seconds(index)
            index = VideoFrameTime(seconds, self._fps)
        elif isinstance(index, float):
            index = VideoFrameTime(index, self._fps).frame_index

        if isinstance(index, VideoFrameTime):
            real_index = index.frame_index
            if (real_index - self._start_time.frame_index) % self._intervals != 0:
                warnings.warn("Index not aligned with intervals, rounding to nearest frame!")
            index = (real_index - self._start_time.frame_index) // self._intervals
        elif isinstance(index, int):
            pass
        else:
            raise IndexError("Invalid index type!")

        if index >= len(self) or index < 0:
            raise IndexError("Index out of range!")

        start_frame_index = self._start_time.frame_index
        frame_index = start_frame_index + index * self._intervals

        if frame_index == self._current_frame_index and self._cache_frame is not None:
            return self._cache_frame,
        
        self._set_start_frame_index(frame_index)

        ret, image = self._video.read()
        if not ret:
            raise ValueError("Frame not read!")
        
        self._cache_frame = image
        # self._current_frame_index += 1
        # if not ret:
        #     raise ValueError("Frame not read!")

        return image,

    def show_data(self, index):
        data, = self.get_data(index)
        plt.imshow(data[:, :, ::-1], cmap="gray")
        plt.show()

    def _set_start_frame_index(self, frame_index):
        if self._current_frame_index >= frame_index:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            skip_frames(self.video, frame_index)
        else:
            skip_frames(self.video, frame_index - self._current_frame_index - 1)
        
        self._current_frame_index = frame_index

    def set_time(self, start_time: str, end_time: str):
        if isinstance(start_time, str):
            start_time = VideoFrameTime(start_time, self._fps)
        if isinstance(end_time, str):
            end_time = VideoFrameTime(end_time, self._fps)

        if start_time.frame_index >= end_time.frame_index:
            raise ValueError("Start time must be earlier than end time!")

        self._start_time = start_time
        self._end_time = end_time
        self._set_start_frame_index(self._start_time.frame_index)

    def set_start_time(self, start_time: str):
        if isinstance(start_time, str):
            start_time = VideoFrameTime(start_time, self._fps)

        if start_time.frame_index >= self._end_time.frame_index:
            raise ValueError("Start time must be earlier than end time!")

        self._start_time = start_time
        self._set_start_frame_index(self._start_time.frame_index)
    
    def set_end_time(self, end_time: str):
        if isinstance(end_time, str):
            end_time = VideoFrameTime(end_time, self._fps)

        if end_time.frame_index <= self._start_time.frame_index:
            raise ValueError("End time must be later than start time!")

        self._end_time = end_time

    def set_intervals(self, intervals: int):
        self._intervals = intervals
    
    def batch(self, batch_size: int):
        data = np.zeros((batch_size, self._height, self._width, 3), dtype=np.uint8)
        for i in range(batch_size):
            try:
                data[i] = next(self)
            except StopIteration:
                data = data[:i]
                break
        
        return data

    def reset(self):
        self._set_start_frame_index(self._start_time.frame_index)

    @property
    def video_path(self):
        return self._video_path
    
    @property
    def video(self):
        return self._video
    
    @property
    def fps(self):
        return self._fps
    
    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height
    
    @property
    def frames(self):
        return self._frames
    
    @property
    def start_time(self):
        return self._start_time
    
    @property
    def end_time(self):
        return self._end_time
    
    @property
    def intervals(self):
        return self._intervals

    @property
    def current_time(self):
        current_time = VideoFrameTime.from_frame_index(self._current_frame_index, self._fps)
        return current_time

    @property
    def current_frame_index(self):
        """
        @brief: 当前指向的帧索引，如果当前帧索引为2，则表示当前指向的是第3帧，第3帧还没有被读取
        """
        return self._current_frame_index
