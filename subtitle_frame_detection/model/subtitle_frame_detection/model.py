import torch
import torch.nn as nn

from practices.model.base.base_model import BaseModel
from practices.model.backbone.resnet import ResNet, BasicBlock, Bottleneck

from .. import MODEL_BUILDER


__all__ = [
    "SubtitleFrameDetectionModel"
]
def __dir__():
    return __all__


@MODEL_BUILDER.register("SubtitleFrameDetectionModel")
class SubtitleFrameDetectionModel(BaseModel):
    def __init__(
        self, 
        device,
        model_path=None,
        input_channels=8,
        block = "BasicBlock",
        layers = [2, 2, 2, 2],
        num_classes=1000,
        stage_with_dcn=(False, False, False, False),
        fallback_on_stride=False,
        with_modulated_dcn=False    
    ):
        super().__init__(
            device=device,
            model_path=model_path,
            input_channels=input_channels,
            block=block,
            layers=layers,
            num_classes=num_classes,
            stage_with_dcn=stage_with_dcn,
            fallback_on_stride=fallback_on_stride,
            with_modulated_dcn=with_modulated_dcn
        )
    
    def _get_block(self, block):
        if block == "BasicBlock":
            return BasicBlock
        elif block == "Bottleneck":
            return Bottleneck
        else:
            raise ValueError(f"Invalid block type: {block}")

    def forward(self, data):
        x1, x2, x3, x4, _ = self.backbone(data)
        x1 = self.avgpool1(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)
        x1 = nn.functional.relu(x1)
        x2 = self.avgpool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc2(x2)
        x2 = nn.functional.relu(x2)
        x3 = self.avgpool3(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc3(x3)
        x3 = nn.functional.relu(x3)
        x4 = self.avgpool4(x4)
        x4 = x4.view(x4.size(0), -1)
        x4 = self.fc4(x4)
        x4 = nn.functional.relu(x4)

        x = x1 + x2 + x3 + x4
        coord = self.coord_fc(x)
        code = self.code_fc(x)

        x = torch.cat([coord, code], dim=1)
        return x

    def build_model(
        self,
        input_channels,
        block,
        layers,
        num_classes,
        stage_with_dcn,
        fallback_on_stride,
        with_modulated_dcn
    ):
        self.backbone = ResNet(
            input_channels = input_channels,
            block = self._get_block(block),
            layers = layers,
            num_classes=num_classes,
            stage_with_dcn=stage_with_dcn,
            fallback_on_stride=fallback_on_stride,
            with_modulated_dcn=with_modulated_dcn
        )
        self.avgpool1 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(64, 512)
        self.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = torch.nn.Linear(128, 512)
        self.avgpool3 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = torch.nn.Linear(256, 512)
        self.avgpool4 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc4 = torch.nn.Linear(512, 512)
        self.coord_fc = torch.nn.Linear(512, 4)
        self.code_fc = torch.nn.Linear(512, 5)