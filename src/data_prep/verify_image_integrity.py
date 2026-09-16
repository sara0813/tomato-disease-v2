"""전처리 데이터 전체의 이미지 무결성을 검사한다.

data/processed/plantvillage/{train,val,test}, data/external/taiwan,
data/external/bangladesh_bbox 를 전부 훑으면서 각 이미지가:
  1) 0바이트가 아닌지
  2) PIL이 손상 없이 열 수 있는지 (Image.verify())
  3) 실제로 픽셀까지 디코딩되는지 (Image.load())
를 확인한다. 문제가 있으면 학습/평가 중간에 죽는 대신 지금 미리 잡는다.

원본은 읽기만 하고 수정하지 않는다.
"""

import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import EXTERNAL_DIRS, TEST_DIR, TRAIN_DIR, VAL_DIR  # noqa: E402
from utils.io import iter_images, save_json  # noqa: E402

TARGET_DIRS = {
    "plantvillage_train": TRAIN_DIR,
    "plantvillage_val": VAL_DIR,
    "plantvillage_test": TEST_DIR,
    "taiwan_external": EXTERNAL_DIRS["taiwan"],
    "bangladesh_bbox_external": EXTERNAL_DIRS["bangladesh_bbox"],
}


def check_image(path: Path) -> str | None:
    """문제 없으면 None, 문제 있으면 에러 메시지를 반환."""
    if path.stat().st_size == 0:
        return "0바이트 파일"

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            if img.mode not in ("RGB", "L"):
                return f"예상 밖 색상 모드: {img.mode}"
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return f"{type(e).__name__}: {e}"

    return None


def main() -> None:
    report = {}
    total_checked = 0
    total_bad = 0

    for dataset_name, root_dir in TARGET_DIRS.items():
        if not root_dir.exists():
            print(f"[스킵] {dataset_name}: 폴더 없음 ({root_dir})")
            continue

        bad_files = []
        count = 0
        for img_path in iter_images(root_dir):
            count += 1
            error = check_image(img_path)
            if error is not None:
                bad_files.append({"path": str(img_path), "error": error})

        report[dataset_name] = {"checked": count, "bad_count": len(bad_files), "bad_files": bad_files}
        total_checked += count
        total_bad += len(bad_files)

        status = "OK" if not bad_files else f"문제 {len(bad_files)}건"
        print(f"{dataset_name:28s} {count:6d}장 검사 -> {status}")
        for bf in bad_files[:10]:
            print(f"    - {bf['path']}: {bf['error']}")
        if len(bad_files) > 10:
            print(f"    ... 외 {len(bad_files) - 10}건 더 (JSON 리포트 참고)")

    print(f"\n총 {total_checked}장 검사, 문제 {total_bad}건")

    from config import COMMON_RESULT_DIR

    save_json(report, COMMON_RESULT_DIR / "image_integrity_report.json")
    print(f"상세 리포트 저장: {COMMON_RESULT_DIR / 'image_integrity_report.json'}")


if __name__ == "__main__":
    main()
