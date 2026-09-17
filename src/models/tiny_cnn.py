"""직접 설계한 경량 CNN (1단계).

설계 원칙 (PDF 6장)
  입력 해상도  필요 범위 내에서 축소      → 연산량·메모리 감소 (128x128 사용)
  필터·층 수   단계별 최소화              → 파라미터 수 감소
  합성곱       Depthwise Separable 검토   → FLOPs 감소
  분류부       Global Average Pooling     → 완전연결층 경량화
  규제         적정 Dropout               → 과적합 억제

변형 3종 (입력 128x128x3, 기본 num_classes=10)
  tiny_cnn_a  최소형    conv 3블록 (32→64→128), 일반 conv        약 9.5만 파라미터
  tiny_cnn_b  중간형    conv 4블록 (32→64→128→128), 일반 conv    약 24만 파라미터
  tiny_cnn_c  Depthwise B와 같은 4블록 깊이, stem 이후 Depthwise
              Separable Conv 적용                                약 3.1만 파라미터

A는 "얼마나 줄여도 버티는가"의 하한선, B는 "일반 conv로 층을 더 쌓으면 얼마나
나아지는가", C는 "B와 같은 깊이인데 Depthwise Separable을 쓰면 파라미터는
B의 1/8 수준으로 줄면서 성능은 얼마나 유지되는가"를 보기 위한 설계다.

세 모델 모두 build 함수는 구조만 만든다 (compile/optimizer는 train_model.py에서
모든 모델에 동일하게 적용해 학습 설정 일관성을 유지한다).
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv3x3 -> BatchNorm -> ReLU -> MaxPool2x2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise3x3 -> Pointwise1x1 -> BatchNorm -> ReLU -> MaxPool2x2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),  # depthwise: 채널별로 독립적인 3x3 필터
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # pointwise: 채널 혼합
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyCNNHead(nn.Module):
    """GAP -> Dropout -> Dense(num_classes). 세 모델이 공유하는 분류부."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class TinyCNN_A(nn.Module):
    """최소형: conv 3블록 (32 -> 64 -> 128), 일반 conv."""

    def __init__(self, input_shape=(3, 128, 128), num_classes: int = 10):
        super().__init__()
        in_channels = input_shape[0]
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.head = TinyCNNHead(128, num_classes, dropout=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TinyCNN_B(nn.Module):
    """중간형: conv 4블록 (32 -> 64 -> 128 -> 128), 일반 conv."""

    def __init__(self, input_shape=(3, 128, 128), num_classes: int = 10):
        super().__init__()
        in_channels = input_shape[0]
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        self.head = TinyCNNHead(128, num_classes, dropout=0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TinyCNN_C(nn.Module):
    """Depthwise: B와 같은 4블록 깊이, stem 이후 Depthwise Separable Conv 적용."""

    def __init__(self, input_shape=(3, 128, 128), num_classes: int = 10):
        super().__init__()
        in_channels = input_shape[0]
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),  # stem은 일반 conv로 저수준 특징 추출
            DepthwiseSeparableBlock(32, 64),
            DepthwiseSeparableBlock(64, 128),
            DepthwiseSeparableBlock(128, 128),
        )
        self.head = TinyCNNHead(128, num_classes, dropout=0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _init_weights(module: nn.Module) -> None:
    """He normal 초기화 (ReLU 계열 활성화에 적합)."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_tiny_cnn_a(input_shape=(3, 128, 128), num_classes: int = 10) -> TinyCNN_A:
    model = TinyCNN_A(input_shape, num_classes)
    model.apply(_init_weights)
    return model


def build_tiny_cnn_b(input_shape=(3, 128, 128), num_classes: int = 10) -> TinyCNN_B:
    model = TinyCNN_B(input_shape, num_classes)
    model.apply(_init_weights)
    return model


def build_tiny_cnn_c(input_shape=(3, 128, 128), num_classes: int = 10) -> TinyCNN_C:
    model = TinyCNN_C(input_shape, num_classes)
    model.apply(_init_weights)
    return model
