import os
import cv2
import glob
import json
import tqdm
import chardet
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import ImageFont, ImageDraw, Image


from practices.dataset.base.base_dataset import BaseDataset
from .. import DATASET_BUILDER
from . import SubtitlePairCode
from .video import VideoFrameDataset, VideoFrameTime


__all__ = ["SimulatedSubtitleFrameDetectionDataset"]
def __dir__():
    return __all__


def get_encoding(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        return encoding


def add_text_to_image(image, text, font, position):
    """
    @brief      在图片上添加文字

    @param      image : numpy.ndarray, 图片
    @param      text  : str, 文字
    @param      font  : str, 字体
    @param      position : tuple, 文字中心位置
    """
    if text is None:
        return image.copy(), None, None, None
    
    h, w = image.shape[:2]

    image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(image)

    start_x = 20  # 留有20像素的边距
    start_y = 0
    start_points = []
    char_bboxes = []
    real_text = ''
    for char in text:
        x1, y1, x2, y2 = draw.textbbox((start_x, start_y), char, font=font)
        if x2 >= w:
            break
        start_points.append((start_x, start_y))
        char_bboxes.append((x1, y1, x2, y2))
        real_text += char
        start_x = x2 + 1
    
    start_points = np.array(start_points)
    char_bboxes = np.array(char_bboxes)
    whole_box = np.array([char_bboxes[:, 0].min(), char_bboxes[:, 1].min(), char_bboxes[:, 2].max(), char_bboxes[:, 3].max()])

    cx, cy = position
    bbox_cx, bbox_cy = (whole_box[0] + whole_box[2]) // 2, (whole_box[1] + whole_box[3]) // 2
    offset = np.array([cx - bbox_cx, cy - bbox_cy, cx - bbox_cx, cy - bbox_cy])

    start_points = start_points + offset[:2]
    char_bboxes = char_bboxes + offset
    whole_box = whole_box + offset
    text = real_text

    for i, char in enumerate(text):
        draw.text(start_points[i], char, font=font, fill=(255, 255, 255))

    x1, y1, x2, y2 = whole_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    whole_box = np.array([cx, cy, w, h])
    
    return np.array(image), text, whole_box, char_bboxes


@DATASET_BUILDER.register("SimulatedSubtitleFrameDetectionDataset")
class SimulatedSubtitleFrameDetectionDataset(BaseDataset):
    """
    @brief: 连续帧字幕数据集
    @details: 
    该类实现了一个连续帧字幕数据集。主要功能包括:
    1. 从视频目录中读取视频文件
    2. 按指定间隔采样连续的视频帧对
    3. 从文本目录中读取字幕文本
    4. 使用不同字体将字幕随机添加到视频帧上
    5. 生成5种不同的字幕对类型:
       - 两帧都无字幕
       - 第一帧无字幕,第二帧有字幕  
       - 第一帧有字幕,第二帧无字幕
       - 两帧都有不同字幕
       - 两帧有相同字幕
    6. 返回帧对图像和对应的标签, 标签为5个元素的列表, 表示第二帧的字幕位置和字幕是否相同, 格式为[cx, cy, w, h, is_same]

    该数据集可用于训练字幕检测、识别等相关模型。

    @param  video_dir: 视频文件目录
    @param  intervals: 采样帧对的间隔
    @param  select_range: 视频采样范围
    @param  image_size: 输出图像大小
    @param  text_dir: 字幕文本目录  
    @param  text_length_range: 字幕长度范围
    @param  font_dir: 字体文件目录
    @param  font_size_range: 字体大小范围

    @example:
    >>> dataset = SimulatedSubtitleFrameDetectionDataset(
    ...     video_dir='path/to/videos',
    ...     intervals=5,
    ...     select_range=[0.2, 0.8],
    ...     image_size=[720, 1280],
    ...     text_dir='path/to/subtitles',
    ...     text_length_range=(5, 20),
    ...     font_dir='path/to/fonts',
    ...     font_size_range=(24, 48)
    ... )
    >>> frame_pair, label = dataset[0]
    >>> print(f"Frame pair shape: {frame_pair[0].shape}, {frame_pair[1].shape}")
    Frame pair shape: (720, 1280, 3), (720, 1280, 3)
    >>> print(f"Label: {label}")
    Label: [200, 100, 300, 200, 1]  # 示例标签，实际值可能不同
    >>> config = {
    ...     "name": "SimulatedSubtitleFrameDetectionDataset",
    ...     "args": {
    ...         "video_dir": "path/to/videos",
    ...         "intervals": 5,
    ...         "select_range": [0.2, 0.8],
    ...         "image_size": [720, 1280],
    ...         "text_dir": "path/to/subtitles",
    ...         "text_length_range": (5, 20),
    ...         "font_dir": "path/to/fonts",
    ...         "font_size_range": (24, 48)
    ...     },
    ...     "preprocessor": {
    ...         "name": "SubtitleFrameDetectionPreprocessor",
    ...         "args": {
    ...             "image_width": 1280,
    ...             "image_height": 720,
    ...             "add_position": True,
    ...         }
    ...     },
    ...     "transforms": [
    ...         {
    ...             "name": "RandomNoise",
    ...             "args": {
    ...                 "lower_noise": -20,
    ...                 "upper_noise": 20
    ...             }
    ...         },
    ...         {
    ...             "name": "Normalize",
    ...             "args": {
    ...                 "mean": [0, 0, 0],
    ...                 "std": [255, 255, 255]
    ...             }
    ...         },
    ...         {
    ...             "name": "Concat",
    ...             "args": {}
    ...         }
    ...     ]
    ... }
    >>> dataset = DATASET_BUILDER.build(config)
    >>> data, label = dataset[0]
    >>> print(f"Data shape: {data.shape}")
    Data shape: (720, 1280, 8)
    """
    def __init__(
        self, 
        video_dir: str,
        intervals: int,
        select_range: List[float],
        image_size: List[int],
        text_dir: str,
        text_length_range: Tuple[int, int],
        font_dir: str,
        font_size_range: Tuple[int, int],
        **kwargs
    ):
        super().__init__(
            video_dir=video_dir,
            intervals=intervals,
            select_range=select_range,
            image_size=image_size,
            text_dir=text_dir,
            text_length_range=text_length_range,
            font_dir=font_dir,
            font_size_range=font_size_range,
            **kwargs
        )
    
    def build_data(
        self,
        video_dir: str,
        intervals: int,
        select_range: List[float],
        image_size: List[int],
        text_dir: str,
        text_length_range: Tuple[int, int],
        font_dir: str,
        font_size_range: Tuple[int, int],
        **kwargs
    ):
        self.video_dir = video_dir
        self.video_paths = glob.glob(os.path.join(video_dir, "**/*.mp4"), recursive=True)
        self.intervals = intervals
        self.select_range = select_range
        self.image_size = image_size
        self.text_dir = text_dir
        self.text_paths = glob.glob(os.path.join(text_dir, "**/*.txt"), recursive=True)
        self.text_length_range = text_length_range
        self.font_dir = font_dir
        self.font_paths = glob.glob(os.path.join(font_dir, "**/*.ttf"), recursive=True)
        self.font_size_range = font_size_range

        self.images = None
        self.texts = None
        self.font = None

        self.load_images(0)
        self.load_texts(0)
        self.load_font(0)
        self.load_data()

    def load_images(self, index: int):
        self.images = np.zeros((25000, self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        selected_paths = np.random.choice(self.video_paths, 10)

        current_index = 0

        for video_path in tqdm.tqdm(selected_paths):
            video = VideoFrameDataset(video_path, self.intervals)
            video_height, video_width = video.height, video.width

            select_start_x = int(video_width * self.select_range[0])
            select_end_x = int(video_width * self.select_range[1])
            select_start_y = int(video_height * self.select_range[2])
            select_end_y = int(video_height * self.select_range[3])

            start_x = np.random.randint(select_start_x, select_end_x - self.image_size[0] + 1)
            start_y = np.random.randint(select_start_y, select_end_y - self.image_size[1] + 1)

            num_frames = len(video)
            num_selected_frames = 2000
            start_index = np.random.randint(0, num_frames - num_selected_frames)
            end_index = start_index + num_selected_frames

            for i in range(start_index, end_index):
                frame, label = video[i]
                frame = frame[start_y:start_y+self.image_size[1], start_x:start_x+self.image_size[0]]
                self.images[current_index] = frame
                current_index += 1
        
        for i in range(current_index, 25000):
            index = np.random.randint(0, 20000)
            self.images[i] = self.images[index]
    
    def load_texts(self, index: int):
        text_path = self.text_paths[index]
        self.texts = []
        encoding = get_encoding(text_path)
        with open(text_path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                text = ""
                for char in line:
                    char_id = ord(char)
                    if char_id >= 0x4E00 and char_id <= 0x9FA5:
                        text += char
                    elif char_id >= 0x20 and char_id <= 0x7E:
                        text += char
                if not text:
                    continue

                if len(text) < self.text_length_range[0]:
                    continue

                if len(text) > self.text_length_range[1]:
                    new_range = np.random.randint(self.text_length_range[0], self.text_length_range[1])
                    start = np.random.randint(0, len(text) - new_range)
                    text = text[start:start+new_range]

                self.texts.append(text)
        
        if len(self.texts) < 100:
            raise ValueError("Too few texts")
    
    def load_font(self, index: int):
        font_path = self.font_paths[index]
        font_size = np.random.randint(self.font_size_range[0], self.font_size_range[1])
        self.font = ImageFont.truetype(font_path, font_size)

    def load_single_data(self, i: int):
        font_index = np.random.randint(0, len(self.font_paths))
        self.load_font(font_index)

        frame1 = self.images[i]
        frame2 = self.images[i + 1]
        frame1 = frame1.astype(np.float32) / 255.0 * 250
        frame2 = frame2.astype(np.float32) / 255.0 * 250
        frame1 = frame1.astype(np.uint8)
        frame2 = frame2.astype(np.uint8)

        code = np.random.randint(0, 5)
        text1 = np.random.choice(self.texts)
        text2 = np.random.choice(self.texts)

        if code == SubtitlePairCode.NF1_NF2:
            text1 = text2 = None
        elif code == SubtitlePairCode.NF1_F2:
            text1 = None
        elif code == SubtitlePairCode.F1_NF2:
            text2 = None
        elif code == SubtitlePairCode.F1_F2_DIFF:
            if text1 == text2:
                while text1 == text2:
                    text2 = np.random.choice(self.texts)
        elif code == SubtitlePairCode.F1_F2_SAME:
            text2 = text1
        else:
            raise ValueError("Invalid code")

        cx, cy = self.image_size[0] // 2, self.image_size[1] // 2
        cx = np.random.randint(cx - 10, cx + 10)
        cy = np.random.randint(cy - 80, cy + 80)

        frame1, text1, bbox1, char_bboxes1 = add_text_to_image(frame1, text1, self.font, (cx, cy))
        frame2, text2, bbox2, char_bboxes2 = add_text_to_image(frame2, text2, self.font, (cx, cy))

        # if bbox1 is not None and bbox2 is not None:
        #     if np.random.rand() < 0.5:
        #         cx1, cy1, w1, h1 = bbox1
        #         cx2, cy2, w2, h2 = bbox2
        #         max_y = max(cy1 + h1 // 2, cy2 + h2 // 2)
        #         if max_y < self.image_size[1] - 100:
        #             y = np.random.randint(max_y + 20, self.image_size[1])
        #             frame1[y:] = 0
        #             frame2[y:] = 0
        # elif bbox1 is not None and bbox2 is None:
        #     if np.random.rand() < 0.5:
        #         cx1, cy1, w1, h1 = bbox1
        #         max_y = cy1 + h1 // 2
        #         if max_y < self.image_size[1] - 100:
        #             y = np.random.randint(max_y + 20, self.image_size[1])
        #             frame1[y:] = 0
        #             frame2[y:] = 0
        # elif bbox1 is None and bbox2 is not None:
        #     if np.random.rand() < 0.5:
        #         cx2, cy2, w2, h2 = bbox2
        #         max_y = cy2 + h2 // 2
        #         if max_y < self.image_size[1] - 100:
        #             y = np.random.randint(max_y + 20, self.image_size[1])
        #             frame1[y:] = 0
        #             frame2[y:] = 0
        # else:
        #     if np.random.rand() < 0.5:
        #         y = np.random.randint(self.image_size[1] - 200, self.image_size[1])
        #         frame1[y:] = 0
        #         frame2[y:] = 0

        if bbox2 is not None:
            label = [*bbox2, code]
        else:
            label = [-100, -100, -100, -100, code]

        frame = np.concatenate([frame1, frame2], axis=-1)
        label = np.array(label, dtype=np.float32)

        self.data[i] = frame
        self.labels[i] = label
        self.text_pairs.append((text1, text2))

    def load_data(self):
        self.data = np.zeros((len(self.images) - 1, self.image_size[1], self.image_size[0], 6), dtype=np.uint8)
        self.labels = np.zeros((len(self.images) - 1, 5), dtype=np.float32)
        self.text_pairs = []

        progress_bar = tqdm.tqdm(range(len(self.data)), desc="Loading data")
        for i in progress_bar:
            self.load_single_data(i)
            progress_bar.update(1)
        progress_bar.close()

    def random_load(self):
        image_index = np.random.randint(0, len(self.video_paths))
        text_index = np.random.randint(0, len(self.text_paths))
        font_index = np.random.randint(0, len(self.font_paths))

        self.load_images(image_index)
        while True:
            try:
                self.load_texts(text_index)
                break
            except Exception as e:
                text_index = np.random.randint(0, len(self.text_paths))
        
        self.load_font(font_index)
        self.load_data()

    def __len__(self):
        return len(self.data)
    
    def get_data(self, index: int):
        frame1_frame2 = self.data[index]
        frame1 = frame1_frame2[:, :, :3]
        frame2 = frame1_frame2[:, :, 3:]
        label = self.labels[index]

        return (frame1, frame2), label

    def show_data(self, index: int):
        (frame1, frame2), label = self.get_data(index)
        cx, cy, w, h, code = label
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        draw = frame2.copy()
        if x1 > 0:
            draw = cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if code == SubtitlePairCode.NF1_NF2:
            print("帧1和帧2都没有字幕")
        elif code == SubtitlePairCode.NF1_F2:
            print("帧1没有字幕，帧2有字幕")
        elif code == SubtitlePairCode.F1_NF2:
            print("帧1有字幕，帧2没有字幕")
        elif code == SubtitlePairCode.F1_F2_DIFF:
            print("帧1和帧2都有字幕，但字幕不同")
        elif code == SubtitlePairCode.F1_F2_SAME:
            print("帧1和帧2都有字幕，且字幕相同")
        else:
            raise ValueError("Invalid code")

        plt.imshow(frame1[:, :, ::-1], cmap="gray")
        plt.show()
        plt.imshow(draw[:, :, ::-1], cmap="gray")
        plt.show()
