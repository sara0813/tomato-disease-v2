"""PyTorch 데이터 로더.

학습·평가 스크립트가 공통으로 쓰는 데이터 파이프라인.
data/processed/plantvillage/{train,val,test}/<class>/*.jpg 구조를
torchvision.datasets.ImageFolder로 그대로 읽는다.

클래스 순서는 class_info.CLASS_NAMES로 고정한다 — ImageFolder는 기본적으로
폴더명을 알파벳순으로 정렬해서 라벨을 매기는데, CLASS_NAMES도 알파벳순으로
정의해 둬서 둘이 항상 일치한다(내부/외부 평가에서 라벨이 어긋나지 않도록).

주의: 이 프로젝트에서 학습 데이터 증강은 사용하지 않는다(교수님 피드백 03).
      이미지 변형은 학습이 아니라 강건성 평가(src/corruption)에만 쓴다.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from class_info import CLASS_NAMES

# ImageNet 사전학습 가중치(MobileNetV2 등 reference 모델)와 통계를 맞춰둔다.
# Tiny CNN처럼 스크래치로 학습하는 모델도 같은 정규화를 써서 파이프라인을 통일한다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_transform(img_size: tuple[int, int]) -> transforms.Compose:
    """리사이즈 + 텐서 변환 + 정규화만 수행한다 (증강 없음)."""
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_dataset(directory: Path, img_size: tuple[int, int]) -> ImageFolder:
    """ImageFolder 데이터셋을 만들고 클래스 순서가 CLASS_NAMES와 일치하는지 검증한다."""
    dataset = ImageFolder(root=str(directory), transform=make_transform(img_size))

    if dataset.classes != CLASS_NAMES:
        raise ValueError(
            f"클래스 순서 불일치: {directory} 의 폴더 목록이 CLASS_NAMES와 다릅니다.\n"
            f"  ImageFolder: {dataset.classes}\n"
            f"  CLASS_NAMES: {CLASS_NAMES}"
        )

    return dataset


def make_dataloader(
    directory: Path,
    img_size: tuple[int, int],
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = make_dataset(directory, img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def count_per_class(directory: Path) -> dict[str, int]:
    """클래스별 이미지 장수를 센다 (불균형 확인용, 텐서 변환 없이 파일만 카운트)."""
    counts: dict[str, int] = {}
    for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        from utils.io import iter_images

        counts[class_dir.name] = sum(1 for _ in iter_images(class_dir))
    return counts
