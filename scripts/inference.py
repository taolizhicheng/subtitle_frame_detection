import os
import cv2
import argparse
from practices.utils.load_config import load_config, save_config


from subtitle_frame_detection.inference import INFERENCE_BUILDER

MODULE_DIR = os.environ["MODULE_DIR"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()
    config = load_config(args.config)
    inference = INFERENCE_BUILDER.build("SubtitleFrameDetectionInference", config)

    video_path = args.video_path
    output_dir = args.output_dir

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path {video_path} does not exist")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config["DATASET"]["ARGS"]["video_path"] = video_path
    results, _ = inference.inference(**config)

    for i, result in enumerate(results):
        start_time = result["start_time"]
        end_time = result["end_time"]
        index = result["index"]
        frame = result["frame"]

        image_path = os.path.join(output_dir, f"{i:04d}.png")
        cv2.imwrite(image_path, frame)

        meta_path = os.path.join(output_dir, f"{i:04d}.json")
        save_config(meta_path, {"start_time": start_time, "end_time": end_time, "index": index})


if __name__ == "__main__":
    main()
