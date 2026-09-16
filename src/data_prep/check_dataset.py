"""데이터 점검: 원본·분할·외부 데이터의 클래스별 장수와 불균형을 확인한다.

주의: 이 스크립트는 불균형을 기록만 한다. 클래스 가중치나 오버샘플링으로
보정하지 않는다 — PlantVillage의 자연스러운 클래스 분포를 그대로 학습에
쓰는 것이 이번 실험 설계다. 다만 나중에 클래스별 recall/F1을 해석할 때
"이 클래스는 원래 학습 데이터가 적었다"는 걸 알아야 하므로 미리 정리해 둔다.

산출물:
  results/_common/dataset_stats.csv   (long format: dataset, class, count)
  results/_common/dataset_stats.md    (사람이 읽기 좋은 표 + 불균형 비율)
"""

import csv
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_NAMES  # noqa: E402
from config import (  # noqa: E402
    COMMON_RESULT_DIR,
    EXTERNAL_DIRS,
    RAW_PLANTVILLAGE_DIR,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from utils.io import iter_images  # noqa: E402

DATASETS = {
    "raw_plantvillage": RAW_PLANTVILLAGE_DIR,
    "plantvillage_train": TRAIN_DIR,
    "plantvillage_val": VAL_DIR,
    "plantvillage_test": TEST_DIR,
    "taiwan_external": EXTERNAL_DIRS["taiwan"],
    "bangladesh_bbox_external": EXTERNAL_DIRS["bangladesh_bbox"],
}


def count_per_class(root_dir: Path) -> dict[str, int]:
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = root_dir / class_name
        counts[class_name] = sum(1 for _ in iter_images(class_dir)) if class_dir.exists() else 0
    return counts


def imbalance_ratio(counts: dict[str, int]) -> float | None:
    """max/min (0인 클래스는 제외하고 계산; 존재하는 클래스가 0개면 None)."""
    nonzero = [c for c in counts.values() if c > 0]
    if not nonzero:
        return None
    return max(nonzero) / min(nonzero)


def main() -> None:
    all_counts = {}
    for dataset_name, root_dir in DATASETS.items():
        if not root_dir.exists():
            print(f"[스킵] {dataset_name}: 폴더 없음 ({root_dir})")
            continue
        all_counts[dataset_name] = count_per_class(root_dir)

    # ---- long format CSV ----
    csv_path = COMMON_RESULT_DIR / "dataset_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "class", "count"])
        for dataset_name, counts in all_counts.items():
            for class_name, count in counts.items():
                writer.writerow([dataset_name, class_name, count])

    # ---- 콘솔 표 + 불균형 비율 ----
    lines = []
    header = f"{'class':45s} " + " ".join(f"{name:>18s}" for name in all_counts)
    print(header)
    lines.append("| class | " + " | ".join(all_counts.keys()) + " |")
    lines.append("|---" * (len(all_counts) + 1) + "|")

    for class_name in CLASS_NAMES:
        row = [all_counts[d].get(class_name, 0) for d in all_counts]
        print(f"{class_name:45s} " + " ".join(f"{v:18d}" for v in row))
        lines.append(f"| {class_name} | " + " | ".join(str(v) for v in row) + " |")

    print("\n=== 불균형 비율 (max/min, 존재하는 클래스 기준) ===")
    lines.append("\n### 불균형 비율 (max/min, 존재하는 클래스 기준)\n")
    lines.append("| dataset | max/min ratio | max class | min class |")
    lines.append("|---|---|---|---|")

    for dataset_name, counts in all_counts.items():
        nonzero = {k: v for k, v in counts.items() if v > 0}
        if not nonzero:
            continue
        ratio = imbalance_ratio(counts)
        max_class = max(nonzero, key=nonzero.get)
        min_class = min(nonzero, key=nonzero.get)
        print(
            f"{dataset_name:28s} ratio={ratio:6.2f}  "
            f"max={max_class}({nonzero[max_class]})  min={min_class}({nonzero[min_class]})"
        )
        lines.append(
            f"| {dataset_name} | {ratio:.2f} | {max_class} ({nonzero[max_class]}) | "
            f"{min_class} ({nonzero[min_class]}) |"
        )

    md_path = COMMON_RESULT_DIR / "dataset_stats.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 데이터셋 클래스별 분포 및 불균형\n\n")
        f.write(
            "이 리포트는 기록용이다. 클래스 가중치/오버샘플링으로 불균형을 보정하지 "
            "않고, PlantVillage의 자연스러운 분포를 그대로 학습에 사용한다.\n\n"
        )
        f.write("\n".join(lines))
        f.write("\n")

    print(f"\n저장: {csv_path}")
    print(f"저장: {md_path}")


if __name__ == "__main__":
    main()
