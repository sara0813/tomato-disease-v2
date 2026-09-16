"""비교용 기준 모델.

- baseline_cnn   1차 실험에서 쓰던 단순 CNN (비교 기준)
- mobilenetv2    경량 후보 기준선 (torchvision, ImageNet 사전학습)
- efficientnetb0 / densenet121  대형 사전학습 모델, 참고 기준으로만 유지

1차 실험 결과(내부 테스트 Accuracy): EfficientNetB0 86.21 / MobileNetV2 84.42 /
Baseline CNN 83.32 / DenseNet121 80.58. 외부 데이터에서는 9.46~30.89로 급락했다.

torchvision 사전학습 모델은 224x224 입력을 기준으로 하므로 config.IMG_SIZE를
쓴다(Tiny CNN 계열의 config.TINY_IMG_SIZE=128x128과 다름에 유의).
"""

import torch
from torch import nn
from torchvision import models


def build_baseline_cnn(input_shape=(3, 224, 224), num_classes: int = 10) -> nn.Module:
    """V1의 baseline_cnn과 동일한 구조 (Conv-Pool x3 + Dense), PyTorch로 재구현."""
    in_channels = input_shape[0]
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )


def _replace_classifier_head(model: nn.Module, in_features: int, num_classes: int, attr_path: list[str]) -> nn.Module:
    """torchvision 모델의 마지막 분류층만 우리 클래스 수(10개)에 맞게 교체한다."""
    parent = model
    for name in attr_path[:-1]:
        parent = getattr(parent, name)
    setattr(parent, attr_path[-1], nn.Linear(in_features, num_classes))
    return model


def build_mobilenetv2(input_shape=(3, 224, 224), num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    return _replace_classifier_head(model, in_features, num_classes, ["classifier", "1"])


def build_efficientnetb0(input_shape=(3, 224, 224), num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    return _replace_classifier_head(model, in_features, num_classes, ["classifier", "1"])


def build_densenet121(input_shape=(3, 224, 224), num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    return _replace_classifier_head(model, in_features, num_classes, ["classifier"])
