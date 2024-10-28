import cv2
import os
import torch

from practices.inference.base.base_inference import BaseInference
from practices.utils.load_config import save_config

from .. import INFERENCE_BUILDER
from ...dataset.subtitle_frame_detection import SubtitlePairCode
from ...dataset.subtitle_frame_detection.video import VideoFrameTime


__all__ = ["SubtitleFrameDetectionInference"]
def __dir__():
    return __all__


CODE_MAP = {
    SubtitlePairCode.NF1_NF2: "第1帧无字幕，第2帧无字幕",
    SubtitlePairCode.NF1_F2: "第1帧无字幕，第2帧有字幕",
    SubtitlePairCode.F1_NF2: "第1帧有字幕，第2帧无字幕",
    SubtitlePairCode.F1_F2_DIFF: "第1帧有字幕，第2帧有字幕，且字幕不同",
    SubtitlePairCode.F1_F2_SAME: "第1帧有字幕，第2帧有字幕，且字幕相同",
}


@INFERENCE_BUILDER.register("SubtitleFrameDetectionInference")
class SubtitleFrameDetectionInference(BaseInference):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def inference(self, **kwargs):
        dataset_args = kwargs.get("DATASET", {})
        dataset = self.build_dataset(**dataset_args)
        dataloader = self.build_dataloader(dataset)
        outputs = []

        self.model.eval()
        with torch.no_grad():
            for data, label in dataloader:
                data = data.to(self.model.device)
                label = label.to(self.model.device)
                
                output = self.model(data)
                data, label, output = self.postprocessor(data, label, output)
                outputs.extend(output)
        
        results = self.post_inference(outputs, dataset=dataset, **kwargs)
        return results

    def post_inference(self, outputs, dataset, **kwargs):
        dataset_args = kwargs.get("DATASET")
        args = dataset_args.get("ARGS", {})
        intervals = args.get("intervals", 5)
        
        hyper_args = kwargs.get("HYPER")
        others_args = hyper_args.get("OTHERS", {})
        fps = others_args.get("fps", 25)

        if not outputs:
            return []

        start_time = end_time = None
        index = None

        results = []
        for i in range(len(outputs)):
            frame_idx = i * intervals
            t = VideoFrameTime.frame_index_to_string_time(frame_idx, fps)

            code, prob = outputs[i]

            if code == SubtitlePairCode.NF1_NF2:
                start_time = end_time = None
                index = i
                continue
            elif code == SubtitlePairCode.NF1_F2:
                start_time = end_time = t
                index = i
            elif code == SubtitlePairCode.F1_NF2:
                if start_time is None:
                    self.logger.warning(f"start_time is None at frame {frame_idx} at case {CODE_MAP[code]}")
                    start_time = end_time = None
                    index = None
                    continue
                if index is None:
                    self.logger.warning(f"index is None at frame {frame_idx} at case {CODE_MAP[code]}")
                    start_time = end_time = None
                    index = None
                    continue

                end_time = t
                results.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "index": index,
                })
                start_time = end_time = None
                index = None
            elif code == SubtitlePairCode.F1_F2_DIFF:
                if start_time is None:
                    self.logger.warning(f"start_time is None at frame {frame_idx} at case {CODE_MAP[code]}")
                    start_time = end_time = None
                    index = None
                    continue

                if index is None:
                    self.logger.warning(f"index is None at frame {frame_idx} at case {CODE_MAP[code]}")
                    start_time = end_time = None
                    index = None
                    continue

                end_time = t
                results.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "index": index,
                })
                start_time = end_time = t
                index = None
            elif code == SubtitlePairCode.F1_F2_SAME:
                end_time = t
                continue
        
        if start_time and end_time:
            results.append({
                "start_time": start_time,
                "end_time": end_time,
                "index": index,
            })

        for i, result in enumerate(results):
            start_time = result["start_time"]
            end_time = result["end_time"]
            index = result["index"]

            (frame1, frame2), label = dataset.get_data(index)
            result["frame"] = frame2
            result["index"] = index * intervals
        
        return results
