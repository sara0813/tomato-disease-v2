"""Corruption 변형 정의 (4단계).

실제 촬영 환경에서 생기는 열화를 테스트 이미지에만 적용한다.
각 조건은 약(1)·중(2)·강(3) 3단계 강도를 가진다.

조건 목록
  brightness  밝기 변화
  shadow      그림자
  reflection  빛 반사
  blur        블러
  noise       노이즈
  occlusion   부분 가림

각 함수 시그니처: fn(image: np.ndarray, level: int) -> np.ndarray
"""

CORRUPTION_TYPES = [
    "brightness",
    "shadow",
    "reflection",
    "blur",
    "noise",
    "occlusion",
]

LEVELS = [1, 2, 3]

# TODO: 조건별 변형 함수 구현 + CORRUPTION_FUNCS 레지스트리
