"""SHOT이 정확히 어느 단계에서 개선되고 어느 단계에서 실패하는지 진단한다.

SHOT은 두 단계로 이뤄진다: ① pseudo-label 할당(클래스 중심점에 가장 가까운 클래스 선택)
② 그 pseudo-label + information-maximization으로 feature extractor 미세조정.
epoch마다 둘 다 다시 계산되는데(pseudo-label은 매 epoch 시작 시 현재 모델 기준으로 재할당),
기존 shot_epoch_sensitivity 분석은 "최종 정확도"만 봤지 "pseudo-label 자체가 얼마나
정확한지"는 안 봤다. 이 스크립트는 target label을 적응에는 전혀 안 쓰고 오직 진단용으로만
써서(SHOT 학습 루프 자체는 수정하지 않음), epoch마다 ① pseudo-label 정확도 ② 그걸로
학습한 뒤의 실제 모델 정확도를 같이 기록한다 — 두 수치가 같이 떨어지면 "pseudo-label
품질 저하가 원인", 모델 정확도만 떨어지면 "fine-tuning 자체가 과적합"이라고 구분할 수 있다.

산출물 → results/external/<dataset>/<base_run_name>_shot_stage_analysis/seed<seed>/stage_analysis.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adapt.domain_adaptation import (  # noqa: E402
    _entropy_of,
    _shot_pseudo_labels,
    class_mask_for,
    load_base_model,
)
from class_info import CLASS_NAMES, EXTERNAL_PRESENT_CLASSES  # noqa: E402
from config import EXTERNAL_DIRS, SEED, external_result_dir  # noqa: E402
from dataset import INTERPOLATION_MODES, make_external_dataloader  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def analyze(
    model_name: str,
    base_run_name: str,
    dataset_key: str,
    data_dir: Path,
    img_size: tuple[int, int],
    interp: str,
    seed: int,
    device,
    epochs: int = 5,
    lr: float = 1e-3,
    cls_loss_weight: float = 0.3,
    div_weight: float = 1.0,
    batch_size: int = 32,
) -> pd.DataFrame:
    set_seed(seed)
    model = load_base_model(model_name, base_run_name, seed, img_size, device)
    loader = make_external_dataloader(data_dir, img_size, batch_size, interpolation=INTERPOLATION_MODES[interp])
    class_mask = class_mask_for(dataset_key, device)
    allowed_idx = sorted({CLASS_NAMES.index(c) for c in EXTERNAL_PRESENT_CLASSES[dataset_key]})
    num_classes = len(CLASS_NAMES)

    # target label은 여기서만, 진단(분석) 목적으로만 모아둔다 — 적응 루프 어디에도 안 들어간다.
    true_labels = []
    for _, labels in loader:
        true_labels.extend(labels.tolist())
    true_labels = torch.tensor(true_labels)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.features.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.features.parameters(), lr=lr)

    def eval_preds() -> torch.Tensor:
        model.eval()
        preds = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                logits = model(images) + class_mask
                preds.extend(logits.argmax(dim=1).cpu().tolist())
        return torch.tensor(preds)

    y_true = true_labels.numpy()

    def per_class_stats(preds: torch.Tensor) -> pd.DataFrame:
        """실험 A 확장: precision만으로는 "그 클래스로 예측된 게 없어서 0%"인지
        "예측은 했는데 다 틀려서 0%"인지 구분이 안 된다 — recall과 예측 개수를 같이 본다."""
        y_pred = preds.numpy()
        prec = precision_score(y_true, y_pred, labels=allowed_idx, average=None, zero_division=0)
        rec = recall_score(y_true, y_pred, labels=allowed_idx, average=None, zero_division=0)
        pred_counts = pd.Series(y_pred).value_counts()
        true_counts = pd.Series(y_true).value_counts()
        rows = []
        for c, p, r in zip(allowed_idx, prec, rec):
            rows.append({
                "class": CLASS_NAMES[c],
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "predicted_count": int(pred_counts.get(c, 0)),
                "true_count": int(true_counts.get(c, 0)),
            })
        return pd.DataFrame(rows)

    def confusion_row(preds: torch.Tensor, true_class_name: str) -> dict:
        """true_class_name인 이미지들이 실제로 어느 클래스로 예측됐는지 분포."""
        true_idx = CLASS_NAMES.index(true_class_name)
        mask = y_true == true_idx
        if mask.sum() == 0:
            return {}
        sub_preds = preds.numpy()[mask]
        counts = pd.Series(sub_preds).value_counts()
        return {CLASS_NAMES[c]: int(counts.get(c, 0)) for c in allowed_idx if counts.get(c, 0) > 0}

    source_preds = eval_preds()
    source_acc = (source_preds == true_labels).float().mean().item()
    source_stats = per_class_stats(source_preds)
    print(f"[{dataset_key}] source softmax acc={source_acc:.4f}")
    print(source_stats.to_string(index=False))

    records = [{"epoch": 0, "stage": "source(적응 전)", "pseudo_label_acc": None, "model_acc": source_acc}]
    source_stats.insert(0, "stage", "source softmax")
    source_stats.insert(0, "epoch", 0)
    stats_tables = [source_stats]

    for epoch in range(1, epochs + 1):
        pseudo_labels = _shot_pseudo_labels(model, loader, device, allowed_idx, class_mask, num_classes)
        if epoch == 1:
            pl_stats = per_class_stats(pseudo_labels)
            print(f"[{dataset_key}] 초기 SHOT pseudo-label per-class:")
            print(pl_stats.to_string(index=False))
            pl_stats.insert(0, "stage", "초기 SHOT pseudo-label")
            pl_stats.insert(0, "epoch", 1)
            stats_tables.append(pl_stats)
            if dataset_key == "taiwan":
                print(f"[{dataset_key}] healthy 이미지가 실제로 어디로 예측됐나:")
                print("  source:", confusion_row(source_preds, "Tomato___healthy"))
                print("  초기 pseudo-label:", confusion_row(pseudo_labels, "Tomato___healthy"))
        pseudo_acc = (pseudo_labels == true_labels).float().mean().item()

        model.train()
        idx = 0
        for images, _ in loader:
            images = images.to(device)
            bsz = images.size(0)
            batch_pseudo = pseudo_labels[idx: idx + bsz].to(device)
            idx += bsz

            optimizer.zero_grad()
            feats = model.head.pool(model.features(images)).flatten(1)
            logits = model.head.fc(feats) + class_mask
            probs = logits.softmax(dim=1)
            im_loss = _entropy_of(probs).mean() - div_weight * _entropy_of(probs.mean(dim=0))
            cls_loss = F.cross_entropy(logits, batch_pseudo)
            loss = im_loss + cls_loss_weight * cls_loss
            loss.backward()
            optimizer.step()

        model_preds = eval_preds()
        model_acc = (model_preds == true_labels).float().mean().item()
        records.append({
            "epoch": epoch, "stage": "pseudo-label 할당 직후",
            "pseudo_label_acc": pseudo_acc, "model_acc": model_acc,
        })
        print(f"[{dataset_key}] epoch {epoch}: pseudo_label_acc={pseudo_acc:.4f}  model_acc_after={model_acc:.4f}")

    df = pd.DataFrame(records)
    out_dir = external_result_dir(dataset_key, f"{base_run_name}_shot_stage_analysis", seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(stats_tables, ignore_index=True).to_csv(
        out_dir / "initial_per_class_precision.csv", index=False, encoding="utf-8-sig"
    )
    df.to_csv(out_dir / "stage_analysis.csv", index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="SHOT 단계별(pseudo-label vs fine-tuning) 진단")
    parser.add_argument("--model", default="tiny_cnn_c")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--interp", choices=list(INTERPOLATION_MODES.keys()), default="bicubic")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataset", choices=list(EXTERNAL_DIRS.keys()) + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cls-loss-weight", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (args.img_size, args.img_size)
    base_run_name = f"{args.model}_res{args.img_size}" if args.interp == "bilinear" else f"{args.model}_res{args.img_size}_{args.interp}"

    datasets = list(EXTERNAL_DIRS.items()) if args.dataset == "all" else [(args.dataset, EXTERNAL_DIRS[args.dataset])]
    for dataset_key, data_dir in datasets:
        analyze(
            args.model, base_run_name, dataset_key, data_dir, img_size, args.interp, args.seed, device,
            epochs=args.epochs, lr=args.lr, cls_loss_weight=args.cls_loss_weight,
        )


if __name__ == "__main__":
    main()
