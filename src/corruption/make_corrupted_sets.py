"""4단계 준비: PlantVillage 테스트셋에 corruption을 적용해 변형 테스트셋을 만든다.

data/processed/plantvillage/test/  →  data/corrupted/<type>/level<n>/<class>/

모든 모델이 동일한 이미지로 평가받도록 한 번만 생성해 두고 재사용한다.
원본 test셋은 건드리지 않고 새 폴더에 변형 이미지를 저장한다.
"""

import shutil
import sys
import zlib
from pathlib import Path

from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_NAMES  # noqa: E402
from config import CORRUPTED_DIR, TEST_DIR  # noqa: E402
from corruption.transforms import CORRUPTION_TYPES, LEVELS, apply_corruption  # noqa: E402
from utils.io import iter_images, save_json  # noqa: E402


def reset_output_dir() -> None:
    if CORRUPTED_DIR.exists():
        shutil.rmtree(CORRUPTED_DIR)
    for corruption_type in CORRUPTION_TYPES:
        for level in LEVELS:
            for class_name in CLASS_NAMES:
                (CORRUPTED_DIR / corruption_type / f"level{level}" / class_name).mkdir(
                    parents=True, exist_ok=True
                )


def image_seed_from_name(name: str) -> int:
    """파일명 기반 정수 시드. 같은 파일이면 항상 같은 시드 -> 재현 가능."""
    return zlib.crc32(name.encode("utf-8"))


def main() -> None:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"테스트셋이 없습니다. 먼저 split_plantvillage.py를 실행하세요: {TEST_DIR}")

    reset_output_dir()

    test_images = []
    for class_name in CLASS_NAMES:
        class_dir = TEST_DIR / class_name
        for img_path in iter_images(class_dir):
            test_images.append((class_name, img_path))

    total_variants = len(test_images) * len(CORRUPTION_TYPES) * len(LEVELS)
    print(f"원본 테스트 이미지: {len(test_images)}장")
    print(f"생성할 변형 이미지: {total_variants}장 ({len(CORRUPTION_TYPES)}조건 x {len(LEVELS)}단계)")

    counts = {}
    done = 0
    report_every = 5000

    for class_name, img_path in test_images:
        seed = image_seed_from_name(img_path.name)
        with Image.open(img_path) as im:
            im = im.convert("RGB")

            for corruption_type in CORRUPTION_TYPES:
                for level in LEVELS:
                    corrupted = apply_corruption(im, corruption_type, level, image_seed=seed)
                    dest_dir = CORRUPTED_DIR / corruption_type / f"level{level}" / class_name
                    corrupted.save(dest_dir / img_path.name, "JPEG", quality=95)

                    key = f"{corruption_type}/level{level}"
                    counts[key] = counts.get(key, 0) + 1
                    done += 1

        if done % report_every < len(CORRUPTION_TYPES) * len(LEVELS):
            print(f"  진행: {done}/{total_variants}")

    print(f"완료: {done}/{total_variants}")

    from config import COMMON_RESULT_DIR

    save_json(
        {"source_test_images": len(test_images), "variants_per_condition_level": counts},
        COMMON_RESULT_DIR / "corruption_generation_stats.json",
    )


if __name__ == "__main__":
    main()
