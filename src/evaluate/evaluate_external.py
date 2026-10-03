"""5단계: 외부 데이터 일반화 평가 (Taiwan / Bangladesh BBox).

외부 데이터에는 PlantVillage 10개 클래스 중 일부만 존재하므로,
Accuracy 등은 실제로 존재하는(true label이 1개 이상 있는) 클래스에 한정해 계산한다.
반면 confusion matrix와 예측 분포는 10개 클래스 전체로 남겨서, 모델이 없는
클래스로 얼마나 편향되게 예측하는지(domain shift 진단)를 볼 수 있게 한다.
(1차 실험에서 다수 모델이 외부 이미지를 Late Blight로 편향 예측했다.)

산출물 → results/external/<dataset>/<model>/seed<seed>/
    metrics.json
    classification_report.csv     (존재하는 클래스만)
    confusion_matrix.csv          (10개 클래스 전체, true는 존재 클래스만 non-zero)
    prediction_distribution.csv   (10개 클래스 전체 예측 분포)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_NAMES  # noqa: E402
from config import EXTERNAL_DIRS, SEED, TINY_MODELS, external_result_dir, model_path  # noqa: E402
from dataset import INTERPOLATION_MODES, make_external_dataloader  # noqa: E402
from evaluate.evaluate_internal import predict_all  # noqa: E402
from models import MODEL_BUILDERS, TINY_MODEL_NAMES, input_shape_for  # noqa: E402
from utils.io import save_json  # noqa: E402

ALL_LABEL_IDS = list(range(len(CLASS_NAMES)))


def _predict_all_masked(model, loader, device, class_mask) -> tuple[list[int], list[int]]:
    """predict_all과 동일하지만, target에 없다고 미리 알려진 클래스의 로짓에 -inf를 더해
    argmax 후보에서 제외한다 (class-restriction 마스킹, domain_adaptation.py 참고)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images) + class_mask
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_preds, all_labels


def evaluate_model_on_external(
    model,
    model_name: str,
    run_name: str,
    dataset_key: str,
    data_dir: Path,
    img_size: tuple[int, int],
    interp: str,
    seed: int,
    device,
    batch_size: int = 32,
    class_mask=None,
) -> dict:
    """이미 준비된 model(가중치 로드/도메인 적응 등 끝난 상태)을 외부 데이터로 평가하고 저장한다.
    run_name이 결과 저장 경로(results/external/<dataset>/<run_name>/seed<seed>/)를 결정하므로,
    domain adaptation처럼 같은 base 가중치에서 파생된 여러 변형을 구분할 때 run_name에 접미사를
    붙여서 호출한다 (예: "tiny_cnn_c_res96_bicubic_adabn").

    class_mask: (num_classes,) 텐서, target에 없는 클래스 위치가 -inf면 해당 클래스로는
    예측하지 않는다 (class-restriction 마스킹 평가 전용, 기본은 마스킹 없음)."""
    loader = make_external_dataloader(data_dir, img_size, batch_size, interpolation=INTERPOLATION_MODES[interp])

    if class_mask is not None:
        preds, labels = _predict_all_masked(model, loader, device, class_mask)
    else:
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

    out_dir = external_result_dir(dataset_key, run_name, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": model_name,
        "run_name": run_name,
        "img_size": list(img_size),
        "interp": interp,
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
        f"[{run_name}] {dataset_key:16s} n={len(labels):5d}  "
        f"acc={acc:.4f}  macro_f1={macro_f1:.4f}  ({len(present_ids)}개 클래스 존재)"
    )
    return metrics


def evaluate_on_external(
    model_name: str,
    dataset_key: str,
    data_dir: Path,
    seed: int = SEED,
    batch_size: int = 32,
    img_size: tuple[int, int] | None = None,
    interp: str = "bilinear",
) -> dict:
    """img_size/interp를 주면 train_model.py --img-size/--interp로 학습한 run을 평가한다
    (해상도·보간법 비교 실험용, RQ3). source-only(가중치 그대로) 평가 전용 — domain adaptation처럼
    가중치를 먼저 손본 모델을 평가하려면 evaluate_model_on_external()을 직접 쓴다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if img_size is None:
        img_size = input_shape_for(model_name)[1:]
        run_name = model_name
    else:
        run_name = f"{model_name}_res{img_size[0]}"
    if interp != "bilinear":
        run_name = f"{run_name}_{interp}"

    weights_path = model_path(run_name, seed)
    if not weights_path.exists():
        raise FileNotFoundError(f"학습된 가중치가 없습니다: {weights_path}")

    model = MODEL_BUILDERS[model_name](input_shape=(3, *img_size)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    return evaluate_model_on_external(
        model, model_name, run_name, dataset_key, data_dir, img_size, interp, seed, device, batch_size
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="5단계: 외부 데이터 일반화 평가")
    parser.add_argument("--model", choices=sorted(TINY_MODEL_NAMES), default=None, help="지정하면 이 모델만 평가 (기본: tiny_cnn 3종 전체)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--img-size", type=int, default=None,
        help="train_model.py --img-size로 학습한 run을 평가 (해상도 비교 실험용, RQ3)",
    )
    parser.add_argument(
        "--interp", choices=list(INTERPOLATION_MODES.keys()), default="bilinear",
        help="train_model.py --interp로 학습한 run을 평가",
    )
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size) if args.img_size else None
    model_names = [args.model] if args.model else TINY_MODELS

    all_results = []
    for dataset_key, data_dir in EXTERNAL_DIRS.items():
        for model_name in model_names:
            all_results.append(
                evaluate_on_external(
                    model_name, dataset_key, data_dir, seed=args.seed, img_size=img_size, interp=args.interp
                )
            )

    print("\n=== 요약 ===")
    for r in all_results:
        print(f"{r['dataset']:16s} {r['run_name']:20s} acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
