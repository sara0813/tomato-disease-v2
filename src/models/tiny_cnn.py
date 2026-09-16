"""직접 설계한 경량 CNN (1단계).

설계 원칙 (PDF 6장)
  입력 해상도  필요 범위 내에서 축소      → 연산량·메모리 감소
  필터·층 수   단계별 최소화              → 파라미터 수 감소
  합성곱       Depthwise Separable 검토   → FLOPs 감소
  분류부       Global Average Pooling     → 완전연결층 경량화
  규제         적정 Dropout               → 과적합 억제

변형 3종
  tiny_cnn_a  최소형    - 가장 작은 채널 구성
  tiny_cnn_b  중간형    - 층/채널을 한 단계 늘린 구성
  tiny_cnn_c  Depthwise - B와 같은 깊이에 Depthwise Separable Conv 적용
"""

# TODO: 구현
# def build_tiny_cnn_a(input_shape, num_classes)
# def build_tiny_cnn_b(input_shape, num_classes)
# def build_tiny_cnn_c(input_shape, num_classes)
