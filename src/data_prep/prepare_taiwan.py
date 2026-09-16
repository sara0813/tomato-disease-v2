"""5단계 준비: Taiwan 데이터셋을 외부 평가셋으로 변환한다.

data/raw/taiwan/{Train,Test}/<taiwan class>/  →  data/external/taiwan/<plantvillage class>/

- Taiwan으로 학습하지 않으므로 Train/Test를 모두 평가에 사용한다.
- class_info.TAIWAN_CLASS_MAPPING에 없는 클래스(Black mold 등)는 제외한다.
- 모델 출력 순서와 맞추기 위해 10개 클래스 폴더를 모두 만들어 둔다(비어 있어도 됨).
"""

# TODO: 구현
