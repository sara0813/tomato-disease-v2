"""지각적 해시(difference hash) 유틸리티.

near-duplicate 진단(check_leakage.py)과 그룹 단위 분할(split_plantvillage.py)에서
공통으로 사용한다.
"""

from pathlib import Path

import numpy as np
from PIL import Image

HASH_SIZE = 8  # 8x8 = 64비트
POPCOUNT_TABLE_256 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def dhash(image_path: Path) -> np.uint64:
    """difference hash: 인접 픽셀 밝기 비교 기반 64비트 지각적 해시."""
    with Image.open(image_path) as img:
        img = img.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
        pixels = np.asarray(img, dtype=np.int16)

    diff = pixels[:, 1:] > pixels[:, :-1]  # (HASH_SIZE, HASH_SIZE) bool
    packed = np.packbits(diff.flatten())  # 8 bytes
    return np.frombuffer(packed.tobytes(), dtype=">u8")[0]  # big-endian uint64
