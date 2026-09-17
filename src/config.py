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
RAW_PLANTDOC_DIR = RAW_DIR / "plantdoc"

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
    "plantdoc": EXTERNAL_DIR / "plantdoc",
}

# Corruption 평가용 변형 테스트셋 (4단계)
CORRUPTED_DIR = DATA_DIR / "corrupted"


# ============================================================
# Model paths
# ============================================================
MODEL_DIR = PROJECT_ROOT / "models"


def model_path(model_name: str, seed: int = None) -> Path:
    """학습된 모델 가중치 저장 경로 (PyTorch state_dict).

    models/<model_name>/seed<seed>.pt 형태로 seed별로 분리 저장한다.
    다른 seed로 재학습해도 기존 seed 결과를 덮어쓰지 않기 위함 (반복실험용).
    """
    if seed is None:
        seed = SEED
    return MODEL_DIR / model_name / f"seed{seed}.pt"


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


def internal_result_dir(model_name: str, seed: int = None) -> Path:
    """2단계(내부 성능) 산출물 경로: results/internal/<model>/seed<seed>/"""
    if seed is None:
        seed = SEED
    return INTERNAL_RESULT_DIR / model_name / f"seed{seed}"


def corruption_result_dir(model_name: str, seed: int = None) -> Path:
    """4단계(강건성) 산출물 경로: results/corruption/<model>/seed<seed>/"""
    if seed is None:
        seed = SEED
    return CORRUPTION_RESULT_DIR / model_name / f"seed{seed}"


def external_result_dir(dataset_key: str, model_name: str, seed: int = None) -> Path:
    """5단계(외부 일반화) 산출물 경로: results/external/<dataset>/<model>/seed<seed>/"""
    if seed is None:
        seed = SEED
    return EXTERNAL_RESULT_DIR / dataset_key / model_name / f"seed{seed}"


# ============================================================
# Image / training settings
# ============================================================
IMG_SIZE = (224, 224)      # 대형 비교 모델 기준 입력 크기
TINY_IMG_SIZE = (128, 128)  # 경량 모델 기본 입력 크기 (RQ3: 해상도 축소 효과)
BATCH_SIZE = 32
EPOCHS = 20                 # early stopping 상한선 (보통 이보다 먼저 멈춤)
EARLY_STOPPING_PATIENCE = 4  # val loss가 이 횟수 연속 개선 안 되면 중단
SEED = 42

# 반복실험(평균·표준편차 보고)용 seed 목록. 현재는 SEED(42)로만 학습이 완료된
# 상태이고, 나머지 두 seed는 아직 실행 전이다. model_path()/*_result_dir()가
# seed별로 경로를 분리해주므로 나중에 이 목록을 돌며 학습해도 seed42 결과는
# 덮어쓰이지 않는다.
SEEDS = [42, 123, 2026]

NUM_CLASSES = 10

# 데이터 분할 비율 (PlantVillage 원본 → train/val/test)
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}


# ============================================================
# 실험 대상 모델
# ============================================================
# V2에서 실제로 학습시키는 모델은 이 3개뿐이다 (직접 설계한 경량 CNN).
TINY_MODELS = ["tiny_cnn_a", "tiny_cnn_b", "tiny_cnn_c"]

# baseline_cnn / mobilenetv2 / efficientnetb0 / densenet121은 V2에서 재학습하지
# 않는다. 1차 실험(V1, docs/의 PDF)에서 이미 이 구조들의 성능을 답했으므로,
# 그 결과를 참고 기준선으로 인용만 한다. models/reference.py의 build 함수들은
# 나중에 필요하면(예: 재현 확인) 쓸 수 있도록 남겨두되, 기본 학습 대상은 아니다.
CITED_REFERENCE_MODELS = ["baseline_cnn", "mobilenetv2", "efficientnetb0", "densenet121"]

ALL_MODELS = TINY_MODELS + CITED_REFERENCE_MODELS


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
