import torch

from practices.loss.base.base_loss import BaseLoss

from .. import LOSS_BUILDER


__all__ = [
    "SubtitleFrameDetectionLoss"
]
def __dir__():
    return __all__


@LOSS_BUILDER.register("SubtitleFrameDetectionLoss")
class SubtitleFrameDetectionLoss(BaseLoss):
    def __init__(
        self, 
        coord_loss_weight: float,
        code_loss_weight: float,
    ):
        super().__init__()
        self.coord_loss = torch.nn.SmoothL1Loss(reduction='mean')
        self.code_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.coord_loss_weight = coord_loss_weight
        self.code_loss_weight = code_loss_weight

    def forward(self, output, label):
        coord = output[:, :4]
        code = output[:, 4:]

        label_coord = label[:, :4]
        label_code = label[:, 4]
        label_code = label_code.to(torch.long)

        coord_loss = self.coord_loss(coord, label_coord)
        code_loss = self.code_loss(code, label_code)
        return coord_loss * self.coord_loss_weight + code_loss * self.code_loss_weight
