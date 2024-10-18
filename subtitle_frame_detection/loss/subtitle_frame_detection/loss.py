import torch

from practices.loss.base.base_loss import BaseLoss

from .. import LOSS_BUILDER


@LOSS_BUILDER.register("SubtitleFrameDetectionLoss")
class SubtitleFrameDetectionLoss(BaseLoss):
    def __init__(
        self, 
        coord_loss_weight: float = 0.1,
        code_loss_weight: float = 1.0
    ):
        super().__init__()
        self.coord_loss = torch.nn.SmoothL1Loss(reduction='mean')
        self.code_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.coord_loss_weight = coord_loss_weight
        self.code_loss_weight = code_loss_weight

    def forward(self, output, label):
        pred_coord = output[:, :4]
        pred_code = output[:, 4]
        label_coord = label[:, :4]
        label_code = label[:, 4]

        coord_loss = self.coord_loss(pred_coord, label_coord) * self.coord_loss_weight
        code_loss = self.code_loss(pred_code, label_code) * self.code_loss_weight
        loss = coord_loss + code_loss
        return loss
