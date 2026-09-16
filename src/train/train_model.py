"""학습 스크립트 (모델 이름을 인자로 받는 단일 진입점, PyTorch).

사용 예:
    python src/train/train_model.py --model tiny_cnn_a
    python src/train/train_model.py --model tiny_cnn_b --epochs 20 --patience 4

V1처럼 모델마다 train_*.py를 따로 두지 않고 한 파일로 통일한다.
models.build_model(name)으로 구조만 가져온 뒤, 여기서 optimizer/loss/loop을
모든 모델에 동일하게 적용해 학습 설정 일관성을 유지한다.

  loss       nn.CrossEntropyLoss
  optimizer  Adam
  조기 종료   validation loss가 config.EARLY_STOPPING_PATIENCE 연속 개선 안 되면 중단,
              가장 좋았던 시점의 가중치를 최종본으로 저장 (상한선 config.EPOCHS)
  device     CPU 기준 (torch.cuda.is_available()이면 자동으로 사용)

체크포인트: val_loss가 갱신될 때마다 즉시 디스크에 저장한다(메모리에만 들고
있다가 끝에 한 번에 저장하지 않음). 학습 도중 세션이 끊겨도 그 시점까지의
best 가중치와 epoch별 로그가 남는다 (실제로 한 번 겪은 문제라 이렇게 바꿈).

V2에서는 config.TINY_MODELS(tiny_cnn_a/b/c) 3개만 학습 대상이다.
나머지(baseline_cnn 등)는 1차 실험 결과를 인용하고 재학습하지 않는다
(config.CITED_REFERENCE_MODELS 참고). 다만 이 스크립트 자체는 등록된 모델이면
어떤 이름이든 받아서 학습할 수 있게 범용으로 만든다.

학습 곡선과 학습 시간은 results/internal/<model>/training_log.{json,csv} 에 저장하고,
가장 좋은 val loss 시점의 가중치는 config.model_path(model_name) (.pt, state_dict)에 저장한다.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch import nn

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    INTERNAL_RESULT_DIR,
    SEED,
    TRAIN_DIR,
    VAL_DIR,
    model_path,
)
from dataset import make_dataloader  # noqa: E402
from models import MODEL_BUILDERS, build_model, input_shape_for  # noqa: E402
from utils.io import save_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def run_epoch(
    model, loader, loss_fn, optimizer, device, train: bool, log_prefix: str = "", log_every: int = 20
) -> tuple[float, float]:
    """한 epoch을 돌고 (평균 loss, accuracy)를 반환한다. train=False면 평가만 한다.

    log_every 배치마다 지금까지의 누적 loss/acc를 한 줄씩 출력한다 (실시간으로
    tail 가능하도록 — epoch 끝날 때 한 줄만 찍으면 8~12분 동안 아무것도 안 보여서).
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_batches = len(loader)
    t0 = time.time()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_count += batch_size

            if log_prefix and (batch_idx % log_every == 0 or batch_idx == n_batches):
                elapsed = time.time() - t0
                print(
                    f"{log_prefix} batch {batch_idx:4d}/{n_batches}  "
                    f"loss={total_loss/total_count:.4f} acc={total_correct/total_count:.4f}  "
                    f"({elapsed:.0f}s)"
                )

    return total_loss / total_count, total_correct / total_count


def train_model(
    model_name: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = 1e-3,
    patience: int = EARLY_STOPPING_PATIENCE,
    seed: int = SEED,
) -> dict:
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_name}] device={device}")

    img_size = input_shape_for(model_name)[1:]  # (C,H,W) -> (H,W)
    train_loader = make_dataloader(TRAIN_DIR, img_size, batch_size, shuffle=True)
    val_loader = make_dataloader(VAL_DIR, img_size, batch_size, shuffle=False)
    print(f"[{model_name}] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  img_size={img_size}")

    model = build_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    save_path = model_path(model_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = INTERNAL_RESULT_DIR / model_name
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_acc = None
    best_epoch = -1
    patience_counter = 0
    history = []

    run_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, optimizer, device, train=True,
            log_prefix=f"[{model_name}] epoch {epoch:2d}/{epochs} [train]",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, loss_fn, optimizer, device, train=False,
            log_prefix=f"[{model_name}] epoch {epoch:2d}/{epochs} [val]",
        )

        epoch_time = time.time() - epoch_start
        print(
            f"[{model_name}] epoch {epoch:2d}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({epoch_time:.1f}s)"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
            }
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # 개선될 때마다 즉시 저장 (끝까지 기다리지 않음)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        # 매 epoch마다 지금까지의 로그를 저장해서, 중간에 끊겨도 진행분이 남게 한다.
        partial_summary = {
            "model": model_name,
            "status": "running",
            "epochs_ran": len(history),
            "epochs_limit": epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "elapsed_sec": time.time() - run_start,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "seed": seed,
            "history": history,
        }
        save_json(partial_summary, log_dir / "training_log.json")
        with open(log_dir / "training_log.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        if not improved and patience_counter >= patience:
            print(f"[{model_name}] early stopping (patience={patience}, best_epoch={best_epoch})")
            break

    total_time = time.time() - run_start

    print(f"[{model_name}] best 가중치는 이미 저장돼 있음 (best_epoch={best_epoch}, val_loss={best_val_loss:.4f}): {save_path}")

    summary = {
        "model": model_name,
        "status": "completed",
        "epochs_ran": len(history),
        "epochs_limit": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "total_train_time_sec": total_time,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "seed": seed,
        "history": history,
    }
    save_json(summary, log_dir / "training_log.json")

    print(f"[{model_name}] 학습 로그 저장: {log_dir}")
    print(f"[{model_name}] 총 학습 시간: {total_time/60:.1f}분")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 학습 (PyTorch)")
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
