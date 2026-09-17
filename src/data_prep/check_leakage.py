"""PlantVillage train/val/test 간 근접 중복(leakage) 이미지를 검사한다.

PlantVillage는 같은 잎을 여러 각도/여러 장 찍은 유사 이미지가 섞여 있다고
알려진 데이터셋이다. 완전 랜덤 분할을 하면 거의 동일한 사진이 train과
test로 나뉘어 들어갈 수 있고, 그러면 내부 테스트 정확도가 "새로운 이미지를
맞히는 능력"이 아니라 "본 적 있는 이미지의 쌍둥이를 맞히는 능력"으로
부풀려질 위험이 있다.

방법: difference hash(dHash, 64비트)로 지각적 해시를 구하고, Hamming distance가
가까운 이미지 쌍을 근접 중복으로 판정한다. train↔test, train↔val을 검사한다
(둘 다 PlantVillage 내부 분할이라 같은 원본 풀에서 나왔을 수 있음).

이 스크립트는 보정하지 않고 진단만 한다 — 결과를 보고 재분할 여부를 결정한다.

(진단 결과: 완전 동일 이미지가 소수 발견되어(test-train 3장, val-train 4장)
split_plantvillage.py를 그룹 단위 분할로 개선함. 재학습 후 이 스크립트를 다시
돌려 완전 동일 개수가 0이 되는지 확인한다.)
"""

import sys
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_NAMES  # noqa: E402
from config import COMMON_RESULT_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR  # noqa: E402
from utils.io import iter_images, save_json  # noqa: E402
from utils.phash import POPCOUNT_TABLE_256, dhash  # noqa: E402

NEAR_DUP_THRESHOLD = 5  # Hamming distance <= 5 이면 근접 중복으로 판정 (0=완전 동일)


def hash_all(root_dir: Path, label: str) -> tuple[list[np.uint64], list[str], list[str]]:
    """폴더 아래 모든 이미지의 해시, 상대경로, 클래스명을 반환."""
    hashes, paths, classes = [], [], []
    t0 = time.time()
    for class_name in CLASS_NAMES:
        class_dir = root_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in iter_images(class_dir):
            hashes.append(dhash(img_path))
            paths.append(str(img_path))
            classes.append(class_name)
    print(f"[{label}] {len(hashes)}장 해시 완료 ({time.time() - t0:.1f}s)")
    return hashes, paths, classes


def min_hamming_distances(query_hashes: np.ndarray, ref_hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """query 각각에 대해 ref 전체와의 최소 Hamming distance와 해당 ref 인덱스를 구한다."""
    n_ref = len(ref_hashes)
    min_dist = np.empty(len(query_hashes), dtype=np.uint16)
    min_idx = np.empty(len(query_hashes), dtype=np.int64)

    for i, q in enumerate(query_hashes):
        xored = np.bitwise_xor(ref_hashes, np.uint64(q))
        byte_view = xored.view(np.uint8).reshape(n_ref, 8)
        dists = POPCOUNT_TABLE_256[byte_view].sum(axis=1)
        j = int(np.argmin(dists))
        min_dist[i] = dists[j]
        min_idx[i] = j

    return min_dist, min_idx


def check_pair(name_a: str, hashes_a, paths_a, classes_a, name_b: str, hashes_b, paths_b, classes_b) -> dict:
    print(f"검사 중: {name_a} <-> {name_b} ({len(hashes_a)} x {len(hashes_b)})")
    t0 = time.time()

    arr_b = np.array(hashes_b, dtype=np.uint64)
    dists, idxs = min_hamming_distances(np.array(hashes_a, dtype=np.uint64), arr_b)

    print(f"  완료 ({time.time() - t0:.1f}s)")

    near_dup_mask = dists <= NEAR_DUP_THRESHOLD
    exact_mask = dists == 0

    examples = []
    for i in np.where(near_dup_mask)[0][:20]:
        j = idxs[i]
        examples.append(
            {
                "distance": int(dists[i]),
                f"{name_a}_path": paths_a[i],
                f"{name_a}_class": classes_a[i],
                f"{name_b}_path": paths_b[j],
                f"{name_b}_class": classes_b[j],
            }
        )

    # 클래스별 근접중복 개수
    by_class = {}
    for i in np.where(near_dup_mask)[0]:
        by_class[classes_a[i]] = by_class.get(classes_a[i], 0) + 1

    return {
        "pair": f"{name_a}<->{name_b}",
        "n_a": len(hashes_a),
        "n_b": len(hashes_b),
        "threshold": NEAR_DUP_THRESHOLD,
        "exact_duplicate_count": int(exact_mask.sum()),
        "near_duplicate_count": int(near_dup_mask.sum()),
        "near_duplicate_ratio_of_a": float(near_dup_mask.mean()),
        "near_duplicate_by_class": by_class,
        "examples": examples,
    }


def main() -> None:
    train_hashes, train_paths, train_classes = hash_all(TRAIN_DIR, "train")
    val_hashes, val_paths, val_classes = hash_all(VAL_DIR, "val")
    test_hashes, test_paths, test_classes = hash_all(TEST_DIR, "test")

    results = {}
    results["test_vs_train"] = check_pair(
        "test", test_hashes, test_paths, test_classes, "train", train_hashes, train_paths, train_classes
    )
    results["val_vs_train"] = check_pair(
        "val", val_hashes, val_paths, val_classes, "train", train_hashes, train_paths, train_classes
    )

    print("\n=== 요약 (Hamming distance <= {} 를 근접 중복으로 판정) ===".format(NEAR_DUP_THRESHOLD))
    for key, r in results.items():
        pct = r["near_duplicate_ratio_of_a"] * 100
        print(
            f"{r['pair']:16s} 완전동일={r['exact_duplicate_count']:5d}  "
            f"근접중복={r['near_duplicate_count']:5d}/{r['n_a']} ({pct:.2f}%)"
        )

    save_json(results, COMMON_RESULT_DIR / "leakage_check_report.json")
    print(f"\n상세 리포트 저장: {COMMON_RESULT_DIR / 'leakage_check_report.json'}")


if __name__ == "__main__":
    main()
