import cv2
import numpy as np
from practices.dataset.base.base_preprocessor import BasePreprocessor
from .. import PREPROCESSOR_BUILDER


__all__ = ["SubtitleFrameDetectionPreprocessor"]
def __dir__():
    return __all__


def get_position_map(image_width: int, image_height: int):
    x_positions = np.zeros((image_height, image_width), np.float32)
    y_positions = np.zeros((image_height, image_width), np.float32)
    for i in range(image_width):
        x_positions[:, i] = i / image_width
    for i in range(image_height):
        y_positions[i, :] = i / image_height
        
    positions = np.zeros((image_height, image_width, 2))
    positions[:, :, 0] = x_positions
    positions[:, :, 1] = y_positions
    return positions


@PREPROCESSOR_BUILDER.register("SubtitleFrameDetectionPreprocessor")
class SubtitleFrameDetectionPreprocessor(BasePreprocessor):
    def __init__(
        self, 
        image_width: int,
        image_height: int,
        add_positions: bool = True,
    ):
        super().__init__(
            image_width=image_width,
            image_height=image_height,
            add_positions=add_positions,
        )
        self.add_positions = add_positions
        self.image_width = image_width
        self.image_height = image_height
        self.positions = get_position_map(image_width, image_height)

    def __call__(self, data, label):
        # keep ratio resize
        frame1, frame2 = data
        h, w = frame1.shape[:2]
        scale = min(self.image_width / w, self.image_height / h)
        nh, nw = int(h * scale), int(w * scale)
        frame1 = cv2.resize(frame1, (nw, nh))
        frame2 = cv2.resize(frame2, (nw, nh))

        if label[0] > 0:
            label[:4] *= scale

        # padding to (self.image_width, self.image_height)
        new_frame1 = np.zeros((self.image_height, self.image_width, frame1.shape[2]), dtype=np.uint8)
        new_frame2 = np.zeros((self.image_height, self.image_width, frame2.shape[2]), dtype=np.uint8)
        new_frame1[:nh, :nw] = frame1
        new_frame2[:nh, :nw] = frame2
        frame1 = new_frame1
        frame2 = new_frame2

        positions = self.positions.copy()

        if self.add_positions:
            return (frame1, frame2, positions), label

        return (frame1, frame2), label
