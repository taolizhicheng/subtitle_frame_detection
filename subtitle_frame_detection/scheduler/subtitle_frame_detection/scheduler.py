from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from practices.scheduler import SCHEDULER_BUILDER
from practices.scheduler.base.base_scheduler import BaseScheduler
from practices.optimizer.base.base_optimizer import BaseOptimizer


@SCHEDULER_BUILDER.register("SubtitleFrameDetectionScheduler")
class SubtitleFrameDetectionScheduler(BaseScheduler):
    def __init__(
        self, 
        optimizer: BaseOptimizer,
        warmup_steps: int, 
        cosine_decay_steps: int,
        cosine_eta_min: float = 0.0,
        **kwargs
    ):
        super().__init__(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            cosine_decay_steps=cosine_decay_steps,
            cosine_eta_min=cosine_eta_min,
            **kwargs
        )

        self.warmup_steps = warmup_steps
        self.cosine_decay_steps = cosine_decay_steps
        self.cosine_eta_min = cosine_eta_min

        # 定义 warmup scheduler
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.lr_warmup)

        # 定义衰减 scheduler
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_decay_steps, eta_min=cosine_eta_min)

    def lr_warmup(self, epoch):
        if epoch < self.warmup_steps:
            return float(epoch + 1) / float(self.warmup_steps)
        return 1.0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            self.warmup_scheduler.step()
        elif self.warmup_steps <= self.current_step <= self.warmup_steps + self.cosine_decay_steps:
            self.cosine_scheduler.step()
        else:
            pass


