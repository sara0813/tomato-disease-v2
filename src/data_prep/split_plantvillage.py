"""1단계 준비: PlantVillage 원본을 train/val/test로 분할한다.

data/raw/plantvillage/<class>/  →  data/processed/plantvillage/{train,val,test}/<class>/

원본은 전혀 건드리지 않고(읽기 전용) 새 폴더에 복사만 한다.
비율은 config.SPLIT_RATIO(70/15/15), 시드는 config.SEED(42)를 따르며,
클래스별로 독립적으로 섞어서 나누기 때문에 자동으로 층화 분할(stratified split)이 된다.
"""

import random
import shutil
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_NAMES  # noqa: E402
from config import (  # noqa: E402
    RAW_PLANTVILLAGE_DIR,
    SEED,
    SPLIT_RATIO,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from utils.io import IMAGE_EXTENSIONS, iter_images, save_json  # noqa: E402


def reset_output_dirs() -> None:
    """이전 실행 결과를 지우고 클래스별 폴더를 새로 만든다."""
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for class_name in CLASS_NAMES:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)


def split_files(files: list[Path], seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    """파일 목록을 섞은 뒤 SPLIT_RATIO 비율로 train/val/test로 자른다."""
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * SPLIT_RATIO["train"])
    n_val = int(n * SPLIT_RATIO["val"])

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train : n_train + n_val]
    test_files = shuffled[n_train + n_val :]  # 나머지는 전부 test (반올림 오차 흡수)
    return train_files, val_files, test_files


def main() -> None:
    if not RAW_PLANTVILLAGE_DIR.exists():
        raise FileNotFoundError(f"원본 데이터가 없습니다: {RAW_PLANTVILLAGE_DIR}")

    reset_output_dirs()

    stats = {}
    print(f"{'class':45s} {'total':>6s} {'train':>6s} {'val':>6s} {'test':>6s}")

    for class_name in CLASS_NAMES:
        class_dir = RAW_PLANTVILLAGE_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"클래스 폴더가 없습니다: {class_dir}")

        files = list(iter_images(class_dir))
        train_files, val_files, test_files = split_files(files, SEED)

        for split_name, split_files_, dest_root in (
            ("train", train_files, TRAIN_DIR),
            ("val", val_files, VAL_DIR),
            ("test", test_files, TEST_DIR),
        ):
            dest_dir = dest_root / class_name
            for src_path in split_files_:
                shutil.copy2(src_path, dest_dir / src_path.name)

        stats[class_name] = {
            "total": len(files),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }
        print(
            f"{class_name:45s} {len(files):6d} {len(train_files):6d} "
            f"{len(val_files):6d} {len(test_files):6d}"
        )

    totals = {
        split: sum(v[split] for v in stats.values()) for split in ("total", "train", "val", "test")
    }
    print(
        f"{'TOTAL':45s} {totals['total']:6d} {totals['train']:6d} "
        f"{totals['val']:6d} {totals['test']:6d}"
    )

    from config import COMMON_RESULT_DIR

    save_json(stats, COMMON_RESULT_DIR / "plantvillage_split_stats.json")
    print(f"\n분할 통계 저장: {COMMON_RESULT_DIR / 'plantvillage_split_stats.json'}")
    print(f"지원 확장자: {sorted(IMAGE_EXTENSIONS)}")


if __name__ == "__main__":
    main()
