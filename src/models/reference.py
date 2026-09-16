"""비교용 기준 모델.

- baseline_cnn   1차 실험에서 쓰던 단순 CNN (비교 기준)
- mobilenetv2    경량 후보 기준선
- efficientnetb0 / densenet121  대형 사전학습 모델, 참고 기준으로만 유지

1차 실험 결과(내부 테스트 Accuracy): EfficientNetB0 86.21 / MobileNetV2 84.42 /
Baseline CNN 83.32 / DenseNet121 80.58. 외부 데이터에서는 9.46~30.89로 급락했다.
"""

# TODO: 구현
# def build_baseline_cnn(input_shape, num_classes)
# def build_mobilenetv2(input_shape, num_classes)
# def build_efficientnetb0(input_shape, num_classes)
# def build_densenet121(input_shape, num_classes)
