"""모델 레지스트리.

모든 학습·평가 스크립트는 build_model(name)으로만 모델을 만든다.
모델을 추가할 때 여기 한 곳만 수정하면 된다.

입력 해상도가 모델군마다 다르다는 점에 유의:
  Tiny CNN 계열       config.TINY_IMG_SIZE (128x128)
  reference 모델 계열  config.IMG_SIZE (224x224, ImageNet 사전학습 기준)
"""

from models.reference import (
    build_baseline_cnn,
    build_densenet121,
    build_efficientnetb0,
    build_mobilenetv2,
)
from models.tiny_cnn import build_tiny_cnn_a, build_tiny_cnn_b, build_tiny_cnn_c

MODEL_BUILDERS = {
    "tiny_cnn_a": build_tiny_cnn_a,
    "tiny_cnn_b": build_tiny_cnn_b,
    "tiny_cnn_c": build_tiny_cnn_c,
    "baseline_cnn": build_baseline_cnn,
    "mobilenetv2": build_mobilenetv2,
    "efficientnetb0": build_efficientnetb0,
    "densenet121": build_densenet121,
}

# 모델별 입력 해상도 (channels, height, width)
TINY_INPUT_SHAPE = (3, 128, 128)
REFERENCE_INPUT_SHAPE = (3, 224, 224)

TINY_MODEL_NAMES = {"tiny_cnn_a", "tiny_cnn_b", "tiny_cnn_c"}


def input_shape_for(name: str) -> tuple[int, int, int]:
    return TINY_INPUT_SHAPE if name in TINY_MODEL_NAMES else REFERENCE_INPUT_SHAPE


def build_model(name: str, num_classes: int = 10, **kwargs):
    if name not in MODEL_BUILDERS:
        raise KeyError(f"등록되지 않은 모델: {name} (사용 가능: {list(MODEL_BUILDERS)})")
    return MODEL_BUILDERS[name](input_shape=input_shape_for(name), num_classes=num_classes, **kwargs)
