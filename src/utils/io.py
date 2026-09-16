"""결과 저장/로드 헬퍼.

모든 평가 스크립트가 같은 형식(JSON + CSV)으로 결과를 남기도록 통일한다.
"""

import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_images(directory: Path):
    """디렉터리 아래 이미지 파일을 재귀적으로 순회한다."""
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
