"""Corruption 변형 정의 (4단계).

실제 촬영 환경에서 생기는 열화를 테스트 이미지에만 적용한다(학습 증강 아님).
각 조건은 약(1)·중(2)·강(3) 3단계 강도를 가지며, 모든 함수는 PIL Image를
받아 PIL Image를 반환한다.

그림자/반사/가림처럼 위치가 필요한 조건은 image_seed로 위치를 결정한다.
같은 seed면 항상 같은 위치가 나오므로(재현성), 호출부에서 파일명 기반 정수를
넘겨 이미지마다 자연스럽게 다른 위치가 나오게 한다.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

CORRUPTION_TYPES = [
    "brightness",
    "shadow",
    "reflection",
    "blur",
    "noise",
    "occlusion",
]

LEVELS = [1, 2, 3]


# ============================================================
# 1. 밝기: 곱셈 계수로 어둡게 (그늘/흐린 날 재현)
# ============================================================
_BRIGHTNESS_FACTOR = {1: 0.7, 2: 0.5, 3: 0.3}


def apply_brightness(image: Image.Image, level: int, image_seed: int = 0) -> Image.Image:
    factor = _BRIGHTNESS_FACTOR[level]
    return ImageEnhance.Brightness(image).enhance(factor)


# ============================================================
# 2. 그림자: 반투명 검은 타원 오버레이
# ============================================================
_SHADOW_PARAMS = {
    1: {"area_ratio": 0.20, "alpha": 0.30},
    2: {"area_ratio": 0.35, "alpha": 0.50},
    3: {"area_ratio": 0.50, "alpha": 0.70},
}


def _random_ellipse_box(img_w: int, img_h: int, area_ratio: float, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """이미지 면적의 area_ratio만큼을 덮는 타원의 bounding box를 무작위 위치에 잡는다."""
    ew = int(img_w * np.sqrt(area_ratio))
    eh = int(img_h * np.sqrt(area_ratio))
    ew, eh = max(ew, 1), max(eh, 1)

    max_x = max(img_w - ew, 0)
    max_y = max(img_h - eh, 0)
    x0 = int(rng.integers(0, max_x + 1))
    y0 = int(rng.integers(0, max_y + 1))
    return x0, y0, x0 + ew, y0 + eh


def apply_shadow(image: Image.Image, level: int, image_seed: int = 0) -> Image.Image:
    params = _SHADOW_PARAMS[level]
    rng = np.random.default_rng(image_seed + level * 1000)

    overlay = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(overlay)
    box = _random_ellipse_box(*image.size, params["area_ratio"], rng)
    draw.ellipse(box, fill=int(255 * params["alpha"]))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(image.size) * 0.02))

    black = Image.new("RGB", image.size, (0, 0, 0))
    return Image.composite(black, image, overlay)


# ============================================================
# 3. 반사: 반투명 흰색 타원 오버레이 (플래시/햇빛 글레어)
# ============================================================
_REFLECTION_PARAMS = {
    1: {"area_ratio": 0.10, "alpha": 0.30},
    2: {"area_ratio": 0.18, "alpha": 0.50},
    3: {"area_ratio": 0.28, "alpha": 0.70},
}


def apply_reflection(image: Image.Image, level: int, image_seed: int = 0) -> Image.Image:
    params = _REFLECTION_PARAMS[level]
    rng = np.random.default_rng(image_seed + level * 2000)

    overlay = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(overlay)
    box = _random_ellipse_box(*image.size, params["area_ratio"], rng)
    draw.ellipse(box, fill=int(255 * params["alpha"]))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(image.size) * 0.03))

    white = Image.new("RGB", image.size, (255, 255, 255))
    return Image.composite(white, image, overlay)


# ============================================================
# 4. 블러: Gaussian Blur
# ============================================================
_BLUR_RADIUS = {1: 2, 2: 4, 3: 7}


def apply_blur(image: Image.Image, level: int, image_seed: int = 0) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=_BLUR_RADIUS[level]))


# ============================================================
# 5. 노이즈: Gaussian noise
# ============================================================
_NOISE_STD = {1: 10, 2: 25, 3: 45}


def apply_noise(image: Image.Image, level: int, image_seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(image_seed + level * 3000)
    arr = np.asarray(image).astype(np.float32)
    noise = rng.normal(0, _NOISE_STD[level], arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


# ============================================================
# 6. 가림: 회색 사각형 패치
# ============================================================
_OCCLUSION_PARAMS = {1: 0.15, 2: 0.25, 3: 0.40}


def apply_occlusion(image: Image.Image, level: int, image_seed: int = 0) -> Image.Image:
    area_ratio = _OCCLUSION_PARAMS[level]
    rng = np.random.default_rng(image_seed + level * 4000)

    img_w, img_h = image.size
    pw = int(img_w * np.sqrt(area_ratio))
    ph = int(img_h * np.sqrt(area_ratio))
    pw, ph = max(pw, 1), max(ph, 1)

    max_x = max(img_w - pw, 0)
    max_y = max(img_h - ph, 0)
    x0 = int(rng.integers(0, max_x + 1))
    y0 = int(rng.integers(0, max_y + 1))

    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle([x0, y0, x0 + pw, y0 + ph], fill=(128, 128, 128))
    return result


CORRUPTION_FUNCS = {
    "brightness": apply_brightness,
    "shadow": apply_shadow,
    "reflection": apply_reflection,
    "blur": apply_blur,
    "noise": apply_noise,
    "occlusion": apply_occlusion,
}


def apply_corruption(image: Image.Image, corruption_type: str, level: int, image_seed: int = 0) -> Image.Image:
    if corruption_type not in CORRUPTION_FUNCS:
        raise KeyError(f"알 수 없는 corruption 타입: {corruption_type}")
    if level not in LEVELS:
        raise ValueError(f"level은 {LEVELS} 중 하나여야 합니다: {level}")
    return CORRUPTION_FUNCS[corruption_type](image, level, image_seed)
