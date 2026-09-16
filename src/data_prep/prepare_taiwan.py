"""5단계 준비: Taiwan 데이터셋을 외부 평가셋으로 변환한다.

data/raw/taiwan/{Train,Test}/<taiwan class>/  →  data/external/taiwan/<plantvillage class>/

- Taiwan으로 학습하지 않으므로 원본의 Train/Test 구분은 우리에게 의미가 없다.
  두 폴더를 전부 합쳐서 평가 샘플 수를 최대화한 외부 평가셋 하나로 만든다.
- class_info.TAIWAN_CLASS_MAPPING에 없는 클래스(Black mold, Gray spot,
  powdery mildew)는 PlantVillage와 대응되지 않으므로 제외한다.
- 모델 출력 순서(10개 클래스)와 맞추기 위해 10개 클래스 폴더를 모두 만들어 둔다
  (Taiwan에 없는 클래스는 빈 폴더로 남는다).
- 원본은 전혀 건드리지 않고 새 폴더에 복사만 한다.
"""

import shutil
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_NAMES, TAIWAN_CLASS_MAPPING  # noqa: E402
from config import EXTERNAL_DIRS, RAW_TAIWAN_DIR  # noqa: E402
from utils.io import iter_images, save_json  # noqa: E402

OUT_DIR = EXTERNAL_DIRS["taiwan"]
SOURCE_SPLITS = ["Train", "Test"]


def reset_output_dir() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for class_name in CLASS_NAMES:
        (OUT_DIR / class_name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not RAW_TAIWAN_DIR.exists():
        raise FileNotFoundError(f"원본 데이터가 없습니다: {RAW_TAIWAN_DIR}")

    reset_output_dir()

    stats: dict[str, dict] = {}
    skipped_classes: set[str] = set()

    for split_name in SOURCE_SPLITS:
        split_dir = RAW_TAIWAN_DIR / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"분할 폴더가 없습니다: {split_dir}")

        for taiwan_class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            taiwan_class_name = taiwan_class_dir.name
            target_class = TAIWAN_CLASS_MAPPING.get(taiwan_class_name)

            if target_class is None:
                skipped_classes.add(taiwan_class_name)
                continue

            dest_dir = OUT_DIR / target_class
            entry = stats.setdefault(
                target_class, {"taiwan_source_classes": set(), "count": 0}
            )
            entry["taiwan_source_classes"].add(taiwan_class_name)

            for src_path in iter_images(taiwan_class_dir):
                # 같은 PlantVillage 클래스에 Train/Test 여러 소스가 합쳐질 수 있으므로
                # split_name + 원본 클래스명을 파일명 앞에 붙여 충돌을 막는다.
                dest_name = f"{split_name}_{taiwan_class_name}_{src_path.name}"
                shutil.copy2(src_path, dest_dir / dest_name)
                entry["count"] += 1

    print(f"{'plantvillage class':45s} {'taiwan source':30s} {'count':>6s}")
    total = 0
    summary = {}
    for class_name in CLASS_NAMES:
        entry = stats.get(class_name)
        count = entry["count"] if entry else 0
        sources = ", ".join(sorted(entry["taiwan_source_classes"])) if entry else "-"
        print(f"{class_name:45s} {sources:30s} {count:6d}")
        total += count
        summary[class_name] = count
    print(f"{'TOTAL':45s} {'':30s} {total:6d}")

    if skipped_classes:
        print(f"\n제외된 Taiwan 클래스 (대응 클래스 없음): {sorted(skipped_classes)}")

    from config import COMMON_RESULT_DIR

    save_json(
        {"per_class_count": summary, "skipped_taiwan_classes": sorted(skipped_classes)},
        COMMON_RESULT_DIR / "taiwan_external_stats.json",
    )


if __name__ == "__main__":
    main()
