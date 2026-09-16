"""tf.data 데이터 로더.

학습·평가 스크립트가 공통으로 쓰는 데이터 파이프라인.
클래스 순서는 class_info.CLASS_NAMES로 고정해서 내부/외부 평가 라벨이 어긋나지 않게 한다.

주의: 이 프로젝트에서 학습 데이터 증강은 사용하지 않는다(교수님 피드백 03).
      이미지 변형은 학습이 아니라 강건성 평가(src/corruption)에만 쓴다.
"""

# TODO: 구현
# def make_dataset(directory, img_size, batch_size, shuffle=False) -> tf.data.Dataset
# def count_per_class(directory) -> dict[str, int]
