"""4단계: 강건성 평가 (PyTorch).

data/corrupted/<type>/level<n>/ 전체에 대해 모델별 정확도를 측정하고,
원본 테스트셋(results/internal/<model>/seed<seed>/metrics.json) 대비 성능 저하율을 계산한다.

    저하율(%) = (원본 Accuracy - 변형 Accuracy) / 원본 Accuracy * 100

산출물 → results/corruption/<model>/seed<seed>/corruption_metrics.csv
         (컬럼: model, corruption_type, level, n, accuracy, macro_f1, drop_rate_pct)
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    CORRUPTED_DIR,
    SEED,
    TINY_MODELS,
    corruption_result_dir,
    internal_result_dir,
    model_path,
)
from corruption.transforms import CORRUPTION_TYPES, LEVELS  # noqa: E402
from dataset import make_dataloader  # noqa: E402
from evaluate.evaluate_internal import predict_all  # noqa: E402
from models import build_model, input_shape_for  # noqa: E402
from utils.io import load_json  # noqa: E402


def evaluate_condition(model, loader, device) -> tuple[float, float, int]:
    preds, labels = predict_all(model, loader, device)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, macro_f1, len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="4단계: 강건성 평가")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    for model_name in TINY_MODELS:
        weights_path = model_path(model_name, seed)
        if not weights_path.exists():
            print(f"[스킵] {model_name}: 학습된 가중치 없음 ({weights_path})")
            continue

        baseline = load_json(internal_result_dir(model_name, seed) / "metrics.json")
        baseline_acc = baseline["accuracy"]

        img_size = input_shape_for(model_name)[1:]
        model = build_model(model_name).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        rows = []
        t0 = time.time()

        for corruption_type in CORRUPTION_TYPES:
            for level in LEVELS:
                cond_dir = CORRUPTED_DIR / corruption_type / f"level{level}"
                loader = make_dataloader(cond_dir, img_size, batch_size, shuffle=False)

                acc, macro_f1, n = evaluate_condition(model, loader, device)
                drop_rate = (baseline_acc - acc) / baseline_acc * 100

                rows.append(
                    {
                        "model": model_name,
                        "corruption_type": corruption_type,
                        "level": level,
                        "n": n,
                        "accuracy": acc,
                        "macro_f1": macro_f1,
                        "baseline_accuracy": baseline_acc,
                        "drop_rate_pct": drop_rate,
                    }
                )
                print(
                    f"[{model_name}] {corruption_type:10s} level{level}  "
                    f"acc={acc:.4f}  drop={drop_rate:+.2f}%  ({time.time()-t0:.0f}s)"
                )

        out_dir = corruption_result_dir(model_name, seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "corruption_metrics.csv", index=False, encoding="utf-8-sig")

        mean_drop = df["drop_rate_pct"].mean()
        print(f"[{model_name}] 평균 저하율: {mean_drop:.2f}%  (총 {time.time()-t0:.0f}s)")
        print(f"[{model_name}] 저장: {out_dir / 'corruption_metrics.csv'}\n")


if __name__ == "__main__":
    main()
