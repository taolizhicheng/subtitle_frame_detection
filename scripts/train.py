import os
import argparse
from practices.utils.load_config import load_config


from subtitle_frame_detection.trainer import TRAINER_BUILDER

MODULE_DIR = os.environ["MODULE_DIR"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, required=True)
    return parser.parse_args()

def main(): 
    args = get_args()
    config = load_config(args.config)
    trainer = TRAINER_BUILDER.build("SubtitleFrameDetectionTrainer", config)
    trainer.test_dataset.load_data(1)
    trainer.train()


if __name__ == "__main__":
    main()
