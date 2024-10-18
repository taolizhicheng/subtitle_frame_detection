from practices.trainer.base.base_trainer import BaseTrainer
from .. import TRAINER_BUILDER


@TRAINER_BUILDER.register("SubtitleFrameDetectionTrainer")
class SubtitleFrameDetectionTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_train_epoch(self):
        self.logger.info("Random Load Train Dataset...")
        self.train_dataset.random_load()
        self.logger.info("Train Dataset Loaded!")
        
        super().before_train_epoch()

