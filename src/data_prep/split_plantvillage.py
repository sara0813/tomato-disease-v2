"""1단계 준비: PlantVillage 원본을 train/val/test로 분할한다.

data/raw/plantvillage/<class>/  →  data/processed/plantvillage/{train,val,test}/<class>/

원본은 전혀 건드리지 않고(읽기 전용) 새 폴더에 복사만 한다.
비율은 config.SPLIT_RATIO(70/15/15), 시드는 config.SEED(42)를 따르며,
클래스별로 독립적으로 섞어서 나누기 때문에 자동으로 층화 분할(stratified split)이 된다.

그룹 단위 분할 (leakage 방지)
    PlantVillage에는 같은 잎을 연속 촬영한 완전 동일 이미지(perceptual hash 동일)가
    소수 섞여 있다(check_leakage.py 진단: test-train 3장, val-train 4장 완전 동일).
    파일 단위로 무작위 분할하면 이런 "쌍둥이 이미지"가 train과 test에 각각 들어가
    내부 테스트 정확도를 실제보다 부풀릴 수 있다. 이를 막기 위해 클래스별로
    dHash가 동일한 이미지들을 먼저 그룹으로 묶고(build_hash_groups), 그룹 전체를
    통째로 하나의 split에만 배정한다(split_groups) — 같은 그룹이 train/val/test에
    걸쳐 나뉘는 일이 없다. 그룹을 통째로 배정하다 보니 클래스별 실제 비율이
    70/15/15에서 아주 약간 벗어날 수 있는데, 그 오차는 stats에 기록한다.
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
from utils.phash import dhash  # noqa: E402


def reset_output_dirs() -> None:
    """이전 실행 결과를 지우고 클래스별 폴더를 새로 만든다."""
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for class_name in CLASS_NAMES:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)


def build_hash_groups(files: list[Path]) -> list[list[Path]]:
    """dHash가 완전히 같은 파일들을 하나의 그룹으로 묶는다 (중복 없는 파일은 그룹 크기 1)."""
    buckets: dict[int, list[Path]] = {}
    for f in files:
        h = int(dhash(f))
        buckets.setdefault(h, []).append(f)
    return list(buckets.values())


def split_groups(
    groups: list[list[Path]], seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    """그룹을 섞은 뒤, 목표 비율(SPLIT_RATIO)에 가장 덜 채워진 split에 그룹째로 배정한다.

    같은 그룹(동일 dHash)은 항상 같은 split에만 들어가므로 train/val/test 간
    완전 동일 이미지가 나뉘어 들어가는 일이 없다.
    """
    shuffled_groups = list(groups)
    random.Random(seed).shuffle(shuffled_groups)

    total = sum(len(g) for g in groups)
    targets = {name: total * ratio for name, ratio in SPLIT_RATIO.items()}
    counts = {"train": 0, "val": 0, "test": 0}
    assigned: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    for group in shuffled_groups:
        # 목표 대비 채워진 비율이 가장 낮은 split에 그룹 전체를 배정
        split_name = min(counts, key=lambda name: counts[name] / targets[name] if targets[name] > 0 else float("inf"))
        assigned[split_name].extend(group)
        counts[split_name] += len(group)

    return assigned["train"], assigned["val"], assigned["test"]


def main() -> None:
    if not RAW_PLANTVILLAGE_DIR.exists():
        raise FileNotFoundError(f"원본 데이터가 없습니다: {RAW_PLANTVILLAGE_DIR}")

    reset_output_dirs()

    stats = {}
    total_dup_groups = 0
    total_dup_images = 0
    print(f"{'class':45s} {'total':>6s} {'train':>6s} {'val':>6s} {'test':>6s} {'dup_groups':>10s}")

    for class_name in CLASS_NAMES:
        class_dir = RAW_PLANTVILLAGE_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"클래스 폴더가 없습니다: {class_dir}")

        files = list(iter_images(class_dir))
        groups = build_hash_groups(files)
        dup_groups = [g for g in groups if len(g) > 1]

        train_files, val_files, test_files = split_groups(groups, SEED)

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
            "n_hash_groups": len(groups),
            "n_duplicate_groups": len(dup_groups),
            "n_images_in_duplicate_groups": sum(len(g) for g in dup_groups),
        }
        total_dup_groups += len(dup_groups)
        total_dup_images += sum(len(g) for g in dup_groups)
        print(
            f"{class_name:45s} {len(files):6d} {len(train_files):6d} "
            f"{len(val_files):6d} {len(test_files):6d} {len(dup_groups):10d}"
        )

    totals = {
        split: sum(v[split] for v in stats.values()) for split in ("total", "train", "val", "test")
    }
    print(
        f"{'TOTAL':45s} {totals['total']:6d} {totals['train']:6d} "
        f"{totals['val']:6d} {totals['test']:6d} {total_dup_groups:10d}"
    )
    print(
        f"\n완전 동일(dHash 동일) 그룹: {total_dup_groups}개 "
        f"({total_dup_images}장) — 그룹 전체를 하나의 split에만 배정해 leakage를 방지했다."
    )

    from config import COMMON_RESULT_DIR

    save_json(stats, COMMON_RESULT_DIR / "plantvillage_split_stats.json")
    print(f"\n분할 통계 저장: {COMMON_RESULT_DIR / 'plantvillage_split_stats.json'}")
    print(f"지원 확장자: {sorted(IMAGE_EXTENSIONS)}")


if __name__ == "__main__":
    main()
