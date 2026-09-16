"""프로젝트 전역 설정.

경로, 이미지/학습 하이퍼파라미터, 실험 대상 모델 목록을 한곳에서 관리한다.
모든 스크립트는 이 파일의 상수를 import 해서 사용한다.
"""

from pathlib import Path

# ============================================================
# Project root
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Dataset paths
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"

# 원본 데이터 (건드리지 않음)
RAW_DIR = DATA_DIR / "raw"
RAW_PLANTVILLAGE_DIR = RAW_DIR / "plantvillage"
RAW_TAIWAN_DIR = RAW_DIR / "taiwan"
RAW_BANGLADESH_DIR = RAW_DIR / "bangladesh"

# 학습용으로 가공한 PlantVillage 분할
PROCESSED_DIR = DATA_DIR / "processed"
PLANTVILLAGE_DIR = PROCESSED_DIR / "plantvillage"

TRAIN_DIR = PLANTVILLAGE_DIR / "train"
VAL_DIR = PLANTVILLAGE_DIR / "val"
TEST_DIR = PLANTVILLAGE_DIR / "test"

# 외부 평가 데이터 (5단계)
EXTERNAL_DIR = DATA_DIR / "external"
EXTERNAL_DIRS = {
    "taiwan": EXTERNAL_DIR / "taiwan",
    "bangladesh_bbox": EXTERNAL_DIR / "bangladesh_bbox",
}

# Corruption 평가용 변형 테스트셋 (4단계)
CORRUPTED_DIR = DATA_DIR / "corrupted"


# ============================================================
# Model paths
# ============================================================
MODEL_DIR = PROJECT_ROOT / "models"


def model_path(model_name: str) -> Path:
    """학습된 모델 가중치 저장 경로."""
    return MODEL_DIR / f"{model_name}.keras"


# ============================================================
# Result paths
# ============================================================
RESULT_DIR = PROJECT_ROOT / "results"

COMMON_RESULT_DIR = RESULT_DIR / "_common"      # class_names.json 등 공용 산출물
INTERNAL_RESULT_DIR = RESULT_DIR / "internal"    # 2단계: PlantVillage 기본 성능
EFFICIENCY_RESULT_DIR = RESULT_DIR / "efficiency"  # 3단계: 크기·속도·FLOPs
CORRUPTION_RESULT_DIR = RESULT_DIR / "corruption"  # 4단계: 강건성
EXTERNAL_RESULT_DIR = RESULT_DIR / "external"    # 5단계: 외부 일반화
SUMMARY_RESULT_DIR = RESULT_DIR / "summary"      # 6단계: 종합 비교·최종 선정
FIGURE_DIR = RESULT_DIR / "figures"

CLASS_NAMES_PATH = COMMON_RESULT_DIR / "class_names.json"


# ============================================================
# Image / training settings
# ============================================================
IMG_SIZE = (224, 224)      # 대형 비교 모델 기준 입력 크기
TINY_IMG_SIZE = (128, 128)  # 경량 모델 기본 입력 크기 (RQ3: 해상도 축소 효과)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

NUM_CLASSES = 10

# 데이터 분할 비율 (PlantVillage 원본 → train/val/test)
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}


# ============================================================
# 실험 대상 모델
# ============================================================
# 이번 연구의 주 비교 대상 (직접 설계한 경량 모델 + 경량 기준선)
TINY_MODELS = ["tiny_cnn_a", "tiny_cnn_b", "tiny_cnn_c"]
LIGHT_BASELINES = ["baseline_cnn", "mobilenetv2"]

# 참고 기준으로만 유지하는 대형 사전학습 모델
REFERENCE_MODELS = ["efficientnetb0", "densenet121"]

ALL_MODELS = TINY_MODELS + LIGHT_BASELINES + REFERENCE_MODELS


# ============================================================
# Directory creation helper
# ============================================================
def ensure_dirs() -> None:
    """산출물 디렉터리를 생성한다 (이미 있으면 통과)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for path in (
        COMMON_RESULT_DIR,
        INTERNAL_RESULT_DIR,
        EFFICIENCY_RESULT_DIR,
        CORRUPTION_RESULT_DIR,
        EXTERNAL_RESULT_DIR,
        SUMMARY_RESULT_DIR,
        FIGURE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

    for path in EXTERNAL_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
