"""실험 B — SHOT Oracle Pseudo-label 상한 실험 (진단 전용, 정상 SFDA 결과 아님).

SHOT의 pseudo-label을 실제 target 정답으로 바꿔서 fine-tuning하면 성능이 얼마나
나오는지 측정한다 — pseudo-label 품질이 SHOT 성능의 진짜 병목인지 확인하는 용도다.

중요: 이 실험은 target label을 학습(fine-tuning)에 직접 사용한다. source-free·
label-free 조건을 전제로 하는 메인 파이프라인(AdaBN/TENT/SHOT/EATA)과는 성격이
다른 "상한선 진단"이며, 정상적인 비지도 SFDA 결과로 보고하면 안 된다.

데이터 유출을 막기 위해 각 외부 데이터셋을 클래스 비율 유지(stratified) 70/30으로
나눠, 70%는 "실제 정답을 pseudo-label처럼 써서" fine-tuning하고 30%는 학습에 전혀
쓰지 않은 채 최종 평가에만 쓴다.

산출물 → results/external/<dataset>/<base_run_name>_shot_oracle_upperbound/seed<seed>/oracle_result.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adapt.domain_adaptation import _entropy_of, class_mask_for, load_base_model  # noqa: E402
from class_info import CLASS_NAMES, EXTERNAL_PRESENT_CLASSES  # noqa: E402
from config import EXTERNAL_DIRS, SEED, external_result_dir  # noqa: E402
from dataset import INTERPOLATION_MODES, ExternalImageDataset  # noqa: E402
from utils.io import save_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def run_oracle(
    model_name: str,
    base_run_name: str,
    dataset_key: str,
    data_dir: Path,
    img_size: tuple[int, int],
    interp: str,
    seed: int,
    device,
    train_frac: float = 0.7,
    epochs: int = 2,
    lr: float = 1e-3,
    cls_loss_weight: float = 0.3,
    div_weight: float = 1.0,
    batch_size: int = 32,
) -> dict:
    set_seed(seed)
    model = load_base_model(model_name, base_run_name, seed, img_size, device)
    class_mask = class_mask_for(dataset_key, device)
    num_classes = len(CLASS_NAMES)

    full_dataset = ExternalImageDataset(data_dir, img_size, INTERPOLATION_MODES[interp])
    labels = [label for _, label in full_dataset.samples]
    indices = list(range(len(full_dataset)))
    train_idx, eval_idx = train_test_split(
        indices, train_size=train_frac, random_state=seed, stratify=labels,
    )

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(Subset(full_dataset, eval_idx), batch_size=batch_size, shuffle=False)

    def eval_accuracy() -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, batch_labels in eval_loader:
                images, batch_labels = images.to(device), batch_labels.to(device)
                logits = model(images) + class_mask
                correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                total += images.size(0)
        return correct / total

    source_acc_on_eval_split = eval_accuracy()  # 비교 기준: 적응 전, 같은 eval split 기준 source 정확도

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.features.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.features.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for images, true_labels in train_loader:
            images, true_labels = images.to(device), true_labels.to(device)
            optimizer.zero_grad()
            feats = model.head.pool(model.features(images)).flatten(1)
            logits = model.head.fc(feats) + class_mask
            probs = logits.softmax(dim=1)
            im_loss = _entropy_of(probs).mean() - div_weight * _entropy_of(probs.mean(dim=0))
            cls_loss = F.cross_entropy(logits, true_labels)  # pseudo-label 대신 실제 정답(oracle)
            loss = im_loss + cls_loss_weight * cls_loss
            loss.backward()
            optimizer.step()

    oracle_acc_on_eval_split = eval_accuracy()

    result = {
        "dataset": dataset_key,
        "n_train": len(train_idx),
        "n_eval": len(eval_idx),
        "source_acc_on_eval_split": source_acc_on_eval_split,
        "oracle_shot_acc_on_eval_split": oracle_acc_on_eval_split,
        "note": "target label을 fine-tuning에 직접 사용한 진단 전용 상한 실험 — 정상 SFDA 결과 아님",
    }
    print(f"[{dataset_key}] source(eval split)={source_acc_on_eval_split:.4f}  "
          f"oracle-SHOT(eval split)={oracle_acc_on_eval_split:.4f}  (n_train={len(train_idx)}, n_eval={len(eval_idx)})")

    out_dir = external_result_dir(dataset_key, f"{base_run_name}_shot_oracle_upperbound", seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, out_dir / "oracle_result.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SHOT Oracle pseudo-label 상한 실험 (진단 전용)")
    parser.add_argument("--model", default="tiny_cnn_c")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--interp", choices=list(INTERPOLATION_MODES.keys()), default="bicubic")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataset", choices=list(EXTERNAL_DIRS.keys()) + ["all"], default="all")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (args.img_size, args.img_size)
    base_run_name = f"{args.model}_res{args.img_size}" if args.interp == "bilinear" else f"{args.model}_res{args.img_size}_{args.interp}"

    datasets = list(EXTERNAL_DIRS.items()) if args.dataset == "all" else [(args.dataset, EXTERNAL_DIRS[args.dataset])]
    for dataset_key, data_dir in datasets:
        run_oracle(
            args.model, base_run_name, dataset_key, data_dir, img_size, args.interp, args.seed, device,
            train_frac=args.train_frac, epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
