"""2단계: PlantVillage 원본 테스트셋 성능 평가 (PyTorch).

config.model_path(name)에서 state_dict를 불러와 model.eval() + torch.no_grad()로
추론한다.

산출물 → results/internal/<model>/
    metrics.json          Accuracy, Macro F1, Weighted F1
    classification_report.csv
    confusion_matrix.csv
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_NAMES  # noqa: E402
from config import INTERNAL_RESULT_DIR, TEST_DIR, TINY_MODELS, model_path  # noqa: E402
from dataset import make_dataloader  # noqa: E402
from models import build_model, input_shape_for  # noqa: E402
from utils.io import save_json  # noqa: E402


@torch.no_grad()
def predict_all(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def evaluate_model(model_name: str, batch_size: int = 32) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = model_path(model_name)
    if not weights_path.exists():
        raise FileNotFoundError(f"학습된 가중치가 없습니다: {weights_path} (먼저 train_model.py 실행)")

    img_size = input_shape_for(model_name)[1:]
    loader = make_dataloader(TEST_DIR, img_size, batch_size, shuffle=False)

    model = build_model(model_name).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    preds, labels = predict_all(model, loader, device)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    report = classification_report(
        labels, preds, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))

    out_dir = INTERNAL_RESULT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": model_name,
        "n_test": len(labels),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    save_json(metrics, out_dir / "metrics.json")

    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv", encoding="utf-8-sig")
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )

    print(
        f"[{model_name}] n_test={len(labels)}  accuracy={acc:.4f}  "
        f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}"
    )
    return metrics


def main() -> None:
    results = [evaluate_model(name) for name in TINY_MODELS]

    print("\n=== 요약 ===")
    for r in results:
        print(f"{r['model']:12s} acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}  weighted_f1={r['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
