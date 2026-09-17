"""5단계: 외부 데이터 일반화 평가 (Taiwan / Bangladesh BBox).

외부 데이터에는 PlantVillage 10개 클래스 중 일부만 존재하므로,
Accuracy 등은 실제로 존재하는(true label이 1개 이상 있는) 클래스에 한정해 계산한다.
반면 confusion matrix와 예측 분포는 10개 클래스 전체로 남겨서, 모델이 없는
클래스로 얼마나 편향되게 예측하는지(domain shift 진단)를 볼 수 있게 한다.
(1차 실험에서 다수 모델이 외부 이미지를 Late Blight로 편향 예측했다.)

산출물 → results/external/<dataset>/<model>/
    metrics.json
    classification_report.csv     (존재하는 클래스만)
    confusion_matrix.csv          (10개 클래스 전체, true는 존재 클래스만 non-zero)
    prediction_distribution.csv   (10개 클래스 전체 예측 분포)
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
from config import EXTERNAL_DIRS, EXTERNAL_RESULT_DIR, TINY_MODELS, model_path  # noqa: E402
from dataset import make_external_dataloader  # noqa: E402
from evaluate.evaluate_internal import predict_all  # noqa: E402
from models import build_model, input_shape_for  # noqa: E402
from utils.io import save_json  # noqa: E402

ALL_LABEL_IDS = list(range(len(CLASS_NAMES)))


def evaluate_on_external(model_name: str, dataset_key: str, data_dir: Path, batch_size: int = 32) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = model_path(model_name)
    if not weights_path.exists():
        raise FileNotFoundError(f"학습된 가중치가 없습니다: {weights_path}")

    img_size = input_shape_for(model_name)[1:]
    loader = make_external_dataloader(data_dir, img_size, batch_size)

    model = build_model(model_name).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    preds, labels = predict_all(model, loader, device)

    present_ids = sorted(set(labels))
    present_names = [CLASS_NAMES[i] for i in present_ids]

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, labels=present_ids, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, labels=present_ids, average="weighted", zero_division=0)

    report = classification_report(
        labels, preds, labels=present_ids, target_names=present_names, output_dict=True, zero_division=0
    )
    # 10개 클래스 전체 기준 confusion matrix -> 없는 클래스로의 오분류(편향)까지 보임
    cm = confusion_matrix(labels, preds, labels=ALL_LABEL_IDS)

    pred_counts = pd.Series(preds).value_counts().reindex(ALL_LABEL_IDS, fill_value=0)
    pred_dist = pd.DataFrame(
        {
            "class": CLASS_NAMES,
            "predicted_count": pred_counts.values,
            "predicted_pct": (pred_counts.values / len(preds) * 100).round(2),
            "present_in_ground_truth": [i in present_ids for i in ALL_LABEL_IDS],
        }
    ).sort_values("predicted_count", ascending=False)

    out_dir = EXTERNAL_RESULT_DIR / dataset_key / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": model_name,
        "dataset": dataset_key,
        "n_total": len(labels),
        "n_classes_present": len(present_ids),
        "classes_present": present_names,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    save_json(metrics, out_dir / "metrics.json")
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv", encoding="utf-8-sig")
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    pred_dist.to_csv(out_dir / "prediction_distribution.csv", index=False, encoding="utf-8-sig")

    print(
        f"[{model_name}] {dataset_key:16s} n={len(labels):5d}  "
        f"acc={acc:.4f}  macro_f1={macro_f1:.4f}  ({len(present_ids)}개 클래스 존재)"
    )
    return metrics


def main() -> None:
    all_results = []
    for dataset_key, data_dir in EXTERNAL_DIRS.items():
        for model_name in TINY_MODELS:
            all_results.append(evaluate_on_external(model_name, dataset_key, data_dir))

    print("\n=== 요약 ===")
    for r in all_results:
        print(f"{r['dataset']:16s} {r['model']:12s} acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
