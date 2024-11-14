from practices.trainer.base.base_trainer import BaseTrainer
from .. import TRAINER_BUILDER


@TRAINER_BUILDER.register("SubtitleFrameDetectionTrainer")
class SubtitleFrameDetectionTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_train_epoch(self, epoch: int):
        self.logger.info("Random Load Train Dataset...")
        if epoch % 5 == 0 and epoch != 0:
            self.train_dataset.random_load()
        self.logger.info("Train Dataset Loaded!")
        
        super().before_train_epoch(epoch)

