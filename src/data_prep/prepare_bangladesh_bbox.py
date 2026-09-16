"""5단계 준비: Bangladesh YOLO 데이터셋을 bbox 크롭 분류셋으로 변환한다.

data/raw/bangladesh/{train,valid,test}/{images,labels}/
    →  data/external/bangladesh_bbox/<plantvillage class>/

- labels/*.txt(YOLO 형식: class cx cy w h, 정규화 좌표)를 읽어 잎 영역만 크롭한다.
- class_info.BANGLADESH_CLASS_ID_MAPPING이 None인 id(Black Spot)는 건너뛴다.
- 한 이미지에 여러 bbox가 있으면 각각 별도 샘플로 저장한다.
"""

# TODO: 구현
