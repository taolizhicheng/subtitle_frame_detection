import cv2
import os
import tqdm
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
        self.dataset = None
        super().__init__(**kwargs)

    def inference(self, **kwargs):
        dataset_args = kwargs.get("DATASET", {})
        self.dataset = self.build_dataset(**dataset_args)
        dataloader = self.build_dataloader(self.dataset)
        outputs = []

        progress_bar = tqdm.tqdm(dataloader, desc="Inference", total=len(dataloader))
        self.model.eval()
        with torch.no_grad():
            for data, label in dataloader:
                data = data.to(self.model.device)
                label = label.to(self.model.device)
                
                output = self.model(data)
                data, label, output = self.postprocessor(data, label, output)
                outputs.extend(output)

                progress_bar.update(1)
        
        progress_bar.close()
        
        results = self.post_inference(outputs, dataset=self.dataset, **kwargs)
        return results, outputs

    def _build_graph(self, outputs):
        graph = {
            "FRAME0_NOTEXT": {},
            "FRAME0_TEXT1": {},
            "FRAME0_TEXT2": {},
        }
        for i, output in enumerate(outputs):
            code, prob = output
            graph[f"FRAME{i}_NOTEXT"][f"FRAME{i+1}_NOTEXT"] = prob[0]
            graph[f"FRAME{i}_NOTEXT"][f"FRAME{i+1}_TEXT1"] = prob[1]
            graph[f"FRAME{i}_NOTEXT"][f"FRAME{i+1}_TEXT2"] = prob[1]
            graph[f"FRAME{i}_TEXT1"][f"FRAME{i+1}_NOTEXT"] = prob[2]
            graph[f"FRAME{i}_TEXT1"][f"FRAME{i+1}_TEXT1"] = prob[4]
            graph[f"FRAME{i}_TEXT1"][f"FRAME{i+1}_TEXT2"] = prob[3]
            graph[f"FRAME{i}_TEXT2"][f"FRAME{i+1}_NOTEXT"] = prob[2]
            graph[f"FRAME{i}_TEXT2"][f"FRAME{i+1}_TEXT1"] = prob[3]
            graph[f"FRAME{i}_TEXT2"][f"FRAME{i+1}_TEXT2"] = prob[4]

            graph[f"FRAME{i+1}_NOTEXT"] = {}
            graph[f"FRAME{i+1}_TEXT1"] = {}
            graph[f"FRAME{i+1}_TEXT2"] = {}

        graph[f"FRAME{len(outputs)}_NOTEXT"] = {"END": 0}
        graph[f"FRAME{len(outputs)}_TEXT1"] = {"END": 0}
        graph[f"FRAME{len(outputs)}_TEXT2"] = {"END": 0}

        return graph

    def _build_path(self, graph, outputs):
        node1 = f"FRAME0_NOTEXT"
        node2 = f"FRAME0_TEXT1"
        node3 = f"FRAME0_TEXT2"
        distances = {node: float('infinity') for node in graph}
        distances[node1] = 0
        distances[node2] = 0
        distances[node3] = 0

        path = {node: [] for node in graph}
        path[node1] = [node1]
        path[node2] = [node2]
        path[node3] = [node3]


        for i in range(len(outputs)):
            node1 = f"FRAME{i}_NOTEXT"
            node2 = f"FRAME{i}_TEXT1"
            node3 = f"FRAME{i}_TEXT2"

            node1_next = f"FRAME{i+1}_NOTEXT"
            node2_next = f"FRAME{i+1}_TEXT1"
            node3_next = f"FRAME{i+1}_TEXT2"

            node1_node1_next_score = distances[node1] + graph[node1][node1_next]
            node2_node1_next_score = distances[node2] + graph[node2][node1_next]
            node3_node1_next_score = distances[node3] + graph[node3][node1_next]

            node1_node2_next_score = distances[node1] + graph[node1][node2_next]
            node2_node2_next_score = distances[node2] + graph[node2][node2_next]
            node3_node2_next_score = distances[node3] + graph[node3][node2_next]

            node1_node3_next_score = distances[node1] + graph[node1][node3_next]
            node2_node3_next_score = distances[node2] + graph[node2][node3_next]
            node3_node3_next_score = distances[node3] + graph[node3][node3_next]

            distances[node1_next] = max(node1_node1_next_score, node2_node1_next_score, node3_node1_next_score)
            distances[node2_next] = max(node1_node2_next_score, node2_node2_next_score, node3_node2_next_score)
            distances[node3_next] = max(node1_node3_next_score, node2_node3_next_score, node3_node3_next_score)

            if distances[node1_next] == node1_node1_next_score:
                path[node1_next] = path[node1] + [node1_next]
            elif distances[node1_next] == node2_node1_next_score:
                path[node1_next] = path[node2] + [node1_next]
            else:
                path[node1_next] = path[node3] + [node1_next]

            if distances[node2_next] == node1_node2_next_score:
                path[node2_next] = path[node1] + [node2_next]
            elif distances[node2_next] == node2_node2_next_score:
                path[node2_next] = path[node2] + [node2_next]
            else:
                path[node2_next] = path[node3] + [node2_next]

            if distances[node3_next] == node1_node3_next_score:
                path[node3_next] = path[node1] + [node3_next]
            elif distances[node3_next] == node2_node3_next_score:
                path[node3_next] = path[node2] + [node3_next]
            else:
                path[node3_next] = path[node3] + [node3_next]

        node_end = f"END"
        distances[node_end] = max(distances[node1_next], distances[node2_next], distances[node3_next])
        if distances[node_end] == distances[node1_next]:
            path[node_end] = path[node1_next] + [node_end]
        elif distances[node_end] == distances[node2_next]:
            path[node_end] = path[node2_next] + [node_end]
        else:
            path[node_end] = path[node3_next] + [node_end]

        return path, distances

    def correct_codes(self, outputs):
        graph = self._build_graph(outputs)
        path, distances = self._build_path(graph, outputs)

        new_outputs = []
        
        for i in range(len(path["END"]) - 2):
            code, prob = outputs[i]

            pred = path["END"][i].split("_")[1]
            post = path["END"][i + 1].split("_")[1]

            if pred == "NOTEXT" and post == "NOTEXT":
                new_code = SubtitlePairCode.NF1_NF2
            elif pred == "NOTEXT" and post == "TEXT1":
                new_code = SubtitlePairCode.NF1_F2
            elif pred == "NOTEXT" and post == "TEXT2":
                new_code = SubtitlePairCode.NF1_F2
            elif pred == "TEXT1" and post == "NOTEXT":
                new_code = SubtitlePairCode.F1_NF2
            elif pred == "TEXT1" and post == "TEXT1":
                new_code = SubtitlePairCode.F1_F2_SAME
            elif pred == "TEXT1" and post == "TEXT2":
                new_code = SubtitlePairCode.F1_F2_DIFF
            elif pred == "TEXT2" and post == "NOTEXT":
                new_code = SubtitlePairCode.F1_NF2
            elif pred == "TEXT2" and post == "TEXT1":
                new_code = SubtitlePairCode.F1_F2_DIFF
            elif pred == "TEXT2" and post == "TEXT2":
                new_code = SubtitlePairCode.F1_F2_SAME

            new_outputs.append((new_code, prob))

        return new_outputs


    def post_inference(self, outputs, dataset, **kwargs):
        # better not to correct codes
        # outputs = self.correct_codes(outputs)

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

            code, prob, coord = outputs[i]

            if code == SubtitlePairCode.NF1_NF2:
                if start_time is not None and end_time is not None and index is not None:
                    self.logger.warning(f"start_time, end_time, index is not None at frame {frame_idx} at case {CODE_MAP[code]}")
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
                    "coord": coord,
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
                    "coord": coord,
                })
                start_time = end_time = t
                index = i
            elif code == SubtitlePairCode.F1_F2_SAME:
                if start_time is None:
                    self.logger.warning(f"start_time is None at frame {frame_idx} at case {CODE_MAP[code]}")

                end_time = t
                continue
        
        if start_time and end_time:
            results.append({
                "start_time": start_time,
                "end_time": end_time,
                "index": index,
                "coord": coord,
            })

        for i, result in enumerate(results):
            start_time = result["start_time"]
            end_time = result["end_time"]
            index = result["index"]

            (frame1, frame2), label = dataset.get_data(index)
            result["prev_frame"] = frame1
            result["frame"] = frame2
            result["index"] = index * intervals
        
        return results
