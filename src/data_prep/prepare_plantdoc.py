"""5단계 준비: PlantDoc 데이터셋을 외부 평가셋으로 변환한다.

data/raw/plantdoc/{train,test}/<plantdoc class>/  →  data/external/plantdoc/<plantvillage class>/

- PlantDoc으로 학습하지 않으므로 원본의 train/test 구분은 우리에게 의미가 없다.
  두 폴더를 전부 합쳐서 평가 샘플 수를 최대화한다.
- class_info.PLANTDOC_CLASS_MAPPING에 없는 클래스(spider mites, 원본 2장뿐)는 제외.
- PlantDoc은 Google/Bing 이미지 검색으로 수집한 데이터라 진단표·여러 이미지를 합친
  비교 콜라주·삽화처럼 "잎 사진이 아닌" 이미지가 섞여 있다. 클래스별 콘택트시트를
  만들어 수작업으로 검수했고, 확인된 문제 이미지는 PLANTDOC_EXCLUDE_FILES로 제외한다.
- 원본은 전혀 건드리지 않고 새 폴더에 복사만 한다.
"""

import shutil
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_NAMES, PLANTDOC_CLASS_MAPPING  # noqa: E402
from config import EXTERNAL_DIRS, RAW_PLANTDOC_DIR  # noqa: E402
from utils.io import iter_images, save_json  # noqa: E402

OUT_DIR = EXTERNAL_DIRS["plantdoc"]
SOURCE_SPLITS = ["train", "test"]

# 콘택트시트 수작업 검수로 확인한 "잎 사진이 아닌" 이미지 (진단표/비교 콜라주/삽화/문서).
# "<split>/<plantdoc 폴더>/<파일명>" 형식, 폴더명은 프로젝트 규칙대로 공백 유지(원본 그대로).
PLANTDOC_EXCLUDE_FILES = {
    # Tomato leaf mosaic virus
    r"train\Tomato leaf mosaic virus\Tomato-brown-rugose-fruit-virus-ToBRFV-infected-tomato-Solanum-lycopersicum-plants_Q320.jpg",
    r"train\Tomato leaf mosaic virus\tomato-fern-leaf-a-symptom-of-tmv-cmv-or-pepmv-virus-on-tomato-plants-h6ey8w.jpg",
    r"test\Tomato leaf mosaic virus\page_2.jpg",
    # Tomato leaf (healthy)
    r"train\Tomato leaf\9bfac1dea7f401e3b379dab33f535bb4.jpg",
    r"train\Tomato leaf\L.peruv-assort.jpg",
    r"test\Tomato leaf\2013-08-20-06.jpg",
    # Tomato leaf yellow virus
    r"train\Tomato leaf yellow virus\IU-TYLCV_img_6.jpg",
    r"train\Tomato leaf yellow virus\tomato-yellow-leaf-curl-disease.jpg",
    r"train\Tomato leaf yellow virus\tomato-yellow-leaf-curl-virus-pdf-78kb.jpg",
    r"train\Tomato leaf yellow virus\tomato_leaf_roll.JPG.jpg",
    r"train\Tomato leaf yellow virus\tylcv-seminar-1-638.jpg",
    r"test\Tomato leaf yellow virus\glyphosate.jpg",
    r"test\Tomato leaf yellow virus\tylcv-seminar-1-638.jpg",
    # Tomato mold leaf
    r"train\Tomato mold leaf\Leaf+Mold+lower+leaf+surface+spores+%28conidia%29.jpg",
    r"train\Tomato mold leaf\Leaf+Mold+upper+leaf+surface.jpg",
    r"train\Tomato mold leaf\top-leaf-mold-Fulva-fulvea-WN.jpg",
    r"test\Tomato mold leaf\50%20Leafmold%20Top.jpg",
    # Tomato Early blight leaf
    r"train\Tomato Early blight leaf\tomato_blight_early.jpg",
    r"test\Tomato Early blight leaf\039b47d574bc4bb8a14259a1cd96a741.jpg",
    # Tomato Septoria leaf spot
    r"train\Tomato Septoria leaf spot\3-septoria-leaf-blight-n.jpg",
    r"train\Tomato Septoria leaf spot\46-Septoria-leaf-spot.jpg",
    r"train\Tomato Septoria leaf spot\slide21-n.jpg",
    r"train\Tomato Septoria leaf spot\Tomato+Problems+Septoria+Leaf+Spot.jpg",
    r"train\Tomato Septoria leaf spot\tomato-diseases-1-638.jpg",
    # Tomato leaf bacterial spot
    r"train\Tomato leaf bacterial spot\a3fb24f84fe70e5d37352d13106531d6.jpg",
    r"train\Tomato leaf bacterial spot\bac-canker.jpg",
    r"train\Tomato leaf bacterial spot\bac-spot.jpg",
    r"train\Tomato leaf bacterial spot\bac.jpg",
    r"train\Tomato leaf bacterial spot\bacterial-spot-of-tomato-6-638.jpg",
    r"train\Tomato leaf bacterial spot\bacterial-spot-of-tomato-7-638.jpg",
    r"train\Tomato leaf bacterial spot\Bacterial-spot-on-tomato-leaves.jpg",
    r"train\Tomato leaf bacterial spot\BacterialSpot06.jpg",
    r"train\Tomato leaf bacterial spot\BacterialSpot07.jpg",
    r"test\Tomato leaf bacterial spot\%2320+Bacterial+Spot+and+Speck.jpg",
    # Tomato leaf late blight
    r"train\Tomato leaf late blight\Late_Blight_2.jpg",
}


def reset_output_dir() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for class_name in CLASS_NAMES:
        (OUT_DIR / class_name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not RAW_PLANTDOC_DIR.exists():
        raise FileNotFoundError(f"원본 데이터가 없습니다: {RAW_PLANTDOC_DIR}")

    reset_output_dir()

    stats: dict[str, dict] = {}
    skipped_classes: set[str] = set()
    excluded_count = 0

    for split_name in SOURCE_SPLITS:
        split_dir = RAW_PLANTDOC_DIR / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"분할 폴더가 없습니다: {split_dir}")

        for plantdoc_class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            plantdoc_class_name = plantdoc_class_dir.name
            target_class = PLANTDOC_CLASS_MAPPING.get(plantdoc_class_name)

            if target_class is None:
                skipped_classes.add(plantdoc_class_name)
                continue

            dest_dir = OUT_DIR / target_class
            entry = stats.setdefault(
                target_class, {"plantdoc_source_classes": set(), "count": 0, "excluded": 0}
            )
            entry["plantdoc_source_classes"].add(plantdoc_class_name)

            for src_path in iter_images(plantdoc_class_dir):
                rel_key = f"{split_name}\\{plantdoc_class_name}\\{src_path.name}"
                if rel_key in PLANTDOC_EXCLUDE_FILES:
                    entry["excluded"] += 1
                    excluded_count += 1
                    continue

                dest_name = f"{split_name}_{plantdoc_class_name}_{src_path.name}"
                shutil.copy2(src_path, dest_dir / dest_name)
                entry["count"] += 1

    print(f"{'plantvillage class':45s} {'plantdoc source':38s} {'count':>6s} {'excluded':>9s}")
    total = 0
    summary = {}
    for class_name in CLASS_NAMES:
        entry = stats.get(class_name)
        count = entry["count"] if entry else 0
        excluded = entry["excluded"] if entry else 0
        sources = ", ".join(sorted(entry["plantdoc_source_classes"])) if entry else "-"
        print(f"{class_name:45s} {sources:38s} {count:6d} {excluded:9d}")
        total += count
        summary[class_name] = count
    print(f"{'TOTAL':45s} {'':38s} {total:6d} {excluded_count:9d}")

    if skipped_classes:
        print(f"\n제외된 PlantDoc 클래스 (대응 클래스 없음/샘플 부족): {sorted(skipped_classes)}")

    from config import COMMON_RESULT_DIR

    save_json(
        {
            "per_class_count": summary,
            "excluded_bad_images": excluded_count,
            "skipped_plantdoc_classes": sorted(skipped_classes),
        },
        COMMON_RESULT_DIR / "plantdoc_external_stats.json",
    )


if __name__ == "__main__":
    main()
