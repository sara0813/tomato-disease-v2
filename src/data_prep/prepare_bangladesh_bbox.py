"""5단계 준비: Bangladesh YOLO 데이터셋을 bbox 크롭 분류셋으로 변환한다.

data/raw/bangladesh/{train,valid,test}/{images,labels}/
    →  data/external/bangladesh_bbox/<plantvillage class>/

- Bangladesh로 학습하지 않으므로 원본의 train/valid/test 구분은 우리에게
  의미가 없다. 세 폴더를 전부 합쳐서 평가 샘플 수를 최대화한다.
- labels/*.txt는 YOLO 형식(각 줄: class cx cy w h, 0~1로 정규화된 좌표)이다.
  한 이미지에 bbox가 여러 개면 각각 별도 크롭 이미지로 저장한다.
- class_info.BANGLADESH_CLASS_ID_MAPPING이 None인 id(1 = Black Spot)는
  PlantVillage와 대응되지 않으므로 건너뛴다.
- 빈 라벨 파일(검출된 bbox 없음)은 건너뛴다.
- 원본은 전혀 건드리지 않고, bbox로 잘라낸 새 이미지를 만들어 저장한다.

주의: 이 데이터셋의 bbox는 "잎 전체"가 아니라 "병반 하나"를 표시한다.
      그대로 자르면 1101장 중 843장(77%)이 128x128보다 작아서(최소 4x3px)
      리사이즈 시 심하게 뭉개진다. 그래서 bbox를 그대로 쓰지 않고
      MARGIN_RATIO(40%) 여백을 두르고, MIN_CROP_SIZE(96px) 미만이면
      중심을 유지한 채 그 크기까지 확장해서 자른다(이미지 경계는 넘지 않게
      필요하면 창을 안쪽으로 밀어서 clamp).
"""

import shutil
import sys
from pathlib import Path

from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import BANGLADESH_CLASS_ID_MAPPING, CLASS_NAMES  # noqa: E402
from config import EXTERNAL_DIRS, RAW_BANGLADESH_DIR  # noqa: E402
from utils.io import save_json  # noqa: E402

OUT_DIR = EXTERNAL_DIRS["bangladesh_bbox"]
SOURCE_SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# bbox가 병반 하나만 표시해서 매우 작은 경우가 많다.
# 여백을 둘러서 주변 잎 맥락을 포함하고, 그래도 작으면 최소 크기까지 확장한다.
MARGIN_RATIO = 0.4  # 원래 bbox 너비/높이의 40%씩 양쪽에 추가
MIN_CROP_SIZE = 96  # 이보다 작으면 중심을 유지한 채 이 크기까지 확장 (px)


def reset_output_dir() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for class_name in CLASS_NAMES:
        (OUT_DIR / class_name).mkdir(parents=True, exist_ok=True)


def find_image_path(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _clamp_window(center: float, size: float, img_dim: int) -> tuple[int, int]:
    """center를 기준으로 size 크기의 구간을 잡되, 이미지 경계를 벗어나면
    크기는 유지한 채 안쪽으로 밀어서 clamp한다 (이미지 자체가 더 작으면 [0, img_dim])."""
    size = min(size, img_dim)
    left = center - size / 2
    right = center + size / 2

    if left < 0:
        right -= left
        left = 0
    if right > img_dim:
        left -= right - img_dim
        right = img_dim

    left = max(0, left)
    right = min(img_dim, right)
    return int(round(left)), int(round(right))


def yolo_box_to_pixels(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """정규화된 YOLO bbox(중심좌표+너비높이) -> 여백/최소크기를 적용한 픽셀 좌표(left, top, right, bottom).

    1) 원래 bbox 크기에 MARGIN_RATIO만큼 여백을 둔다 (잎 맥락 포함).
    2) 그래도 MIN_CROP_SIZE보다 작으면 중심을 유지한 채 그 크기까지 확장한다.
    3) 이미지 경계를 넘으면 크기는 유지하고 창을 안쪽으로 밀어서 clamp한다.
    """
    box_w_px = w * img_w
    box_h_px = h * img_h
    center_x = cx * img_w
    center_y = cy * img_h

    target_w = max(box_w_px * (1 + 2 * MARGIN_RATIO), MIN_CROP_SIZE)
    target_h = max(box_h_px * (1 + 2 * MARGIN_RATIO), MIN_CROP_SIZE)

    left, right = _clamp_window(center_x, target_w, img_w)
    top, bottom = _clamp_window(center_y, target_h, img_h)

    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return left, top, right, bottom


def main() -> None:
    if not RAW_BANGLADESH_DIR.exists():
        raise FileNotFoundError(f"원본 데이터가 없습니다: {RAW_BANGLADESH_DIR}")

    reset_output_dir()

    counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    skipped_class_id_count = 0
    empty_label_count = 0
    missing_image_count = 0
    bad_box_count = 0

    for split_name in SOURCE_SPLITS:
        split_dir = RAW_BANGLADESH_DIR / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not labels_dir.exists():
            raise FileNotFoundError(f"라벨 폴더가 없습니다: {labels_dir}")

        for label_path in sorted(labels_dir.glob("*.txt")):
            lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                empty_label_count += 1
                continue

            image_path = find_image_path(images_dir, label_path.stem)
            if image_path is None:
                missing_image_count += 1
                continue

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size

                for box_idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        bad_box_count += 1
                        continue

                    class_id = int(parts[0])
                    cx, cy, w, h = (float(v) for v in parts[1:])

                    target_class = BANGLADESH_CLASS_ID_MAPPING.get(class_id)
                    if target_class is None:
                        skipped_class_id_count += 1
                        continue

                    left, top, right, bottom = yolo_box_to_pixels(cx, cy, w, h, img_w, img_h)
                    crop = img.crop((left, top, right, bottom))

                    dest_name = f"{split_name}_{label_path.stem}_box{box_idx}.jpg"
                    crop.save(OUT_DIR / target_class / dest_name, "JPEG", quality=95)
                    counts[target_class] += 1

    print(f"{'plantvillage class':45s} {'count':>6s}")
    total = 0
    for class_name in CLASS_NAMES:
        print(f"{class_name:45s} {counts[class_name]:6d}")
        total += counts[class_name]
    print(f"{'TOTAL':45s} {total:6d}")

    print(f"\n제외된 bbox (대응 클래스 없음, 주로 Black Spot): {skipped_class_id_count}")
    print(f"빈 라벨 파일(스킵): {empty_label_count}")
    print(f"이미지 파일 못 찾음(스킵): {missing_image_count}")
    print(f"형식 오류 bbox(스킵): {bad_box_count}")

    from config import COMMON_RESULT_DIR

    save_json(
        {
            "per_class_count": counts,
            "skipped_class_id_boxes": skipped_class_id_count,
            "empty_label_files": empty_label_count,
            "missing_images": missing_image_count,
            "bad_format_boxes": bad_box_count,
        },
        COMMON_RESULT_DIR / "bangladesh_bbox_external_stats.json",
    )


if __name__ == "__main__":
    main()
