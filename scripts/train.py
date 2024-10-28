import os
from practices.utils.load_config import load_config


from subtitle_frame_detection.trainer import TRAINER_BUILDER

MODULE_DIR = os.environ["MODULE_DIR"]


def main(): 
    config = load_config(f"{MODULE_DIR}/configs/subtitle_frame_detection/train/base.yaml")
    trainer = TRAINER_BUILDER.build("SubtitleFrameDetectionTrainer", config)
    trainer.test_dataset.load_data(1)
    trainer.train()


if __name__ == "__main__":
    main()
