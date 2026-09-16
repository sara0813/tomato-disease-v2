"""모델 레지스트리.

모든 학습·평가 스크립트는 build_model(name)으로만 모델을 만든다.
모델을 추가할 때 여기 한 곳만 수정하면 된다.
"""

# TODO: tiny_cnn / reference의 build 함수를 import 해서 아래 표를 채운다
MODEL_BUILDERS = {
    # "tiny_cnn_a": build_tiny_cnn_a,
    # "tiny_cnn_b": build_tiny_cnn_b,
    # "tiny_cnn_c": build_tiny_cnn_c,
    # "baseline_cnn": build_baseline_cnn,
    # "mobilenetv2": build_mobilenetv2,
    # "efficientnetb0": build_efficientnetb0,
    # "densenet121": build_densenet121,
}


def build_model(name: str, input_shape, num_classes: int):
    if name not in MODEL_BUILDERS:
        raise KeyError(f"등록되지 않은 모델: {name} (사용 가능: {list(MODEL_BUILDERS)})")
    return MODEL_BUILDERS[name](input_shape=input_shape, num_classes=num_classes)
