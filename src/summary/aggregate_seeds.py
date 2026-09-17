"""반복실험(멀티 seed) 결과 집계.

config.SEEDS에 등록된 seed 중 실제로 학습·평가(2단계, results/internal/<model>/seed<seed>/
metrics.json)가 끝난 것만 모아서 모델별 평균 Accuracy · 표준편차 · 평균 Macro F1 ·
최고/최저 성능을 계산한다. 아직 일부 seed만 완료됐어도(현재는 seed42뿐) 동작하며,
seed가 1개뿐이면 표준편차는 계산하지 않고(None) 넘어간다.

학습이 끝날 때마다 다시 실행하면 되는 집계 전용 스크립트다 — 학습 자체는 하지 않는다.

산출물 → results/summary/
    seed_aggregate.csv
    seed_aggregate.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import SEEDS, SUMMARY_RESULT_DIR, TINY_MODELS, internal_result_dir  # noqa: E402
from utils.io import load_json  # noqa: E402


def collect_seed_metrics(model_name: str) -> list[dict]:
    """완료된 seed들의 2단계(내부 성능) metrics.json을 모은다."""
    rows = []
    for seed in SEEDS:
        path = internal_result_dir(model_name, seed) / "metrics.json"
        if not path.exists():
            continue
        m = load_json(path)
        rows.append(
            {
                "seed": seed,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
            }
        )
    return rows


def aggregate_model(model_name: str) -> dict | None:
    rows = collect_seed_metrics(model_name)
    if not rows:
        return None

    accs = [r["accuracy"] for r in rows]
    f1s = [r["macro_f1"] for r in rows]
    best = max(rows, key=lambda r: r["accuracy"])
    worst = min(rows, key=lambda r: r["accuracy"])

    return {
        "model": model_name,
        "n_seeds": len(rows),
        "seeds_used": ",".join(str(r["seed"]) for r in rows),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else None,
        "best_seed": best["seed"],
        "best_accuracy": best["accuracy"],
        "worst_seed": worst["seed"],
        "worst_accuracy": worst["accuracy"],
    }


def main() -> None:
    results = []
    for model_name in TINY_MODELS:
        agg = aggregate_model(model_name)
        if agg is None:
            print(f"[스킵] {model_name}: 완료된 seed 결과가 없음 (train_model.py 먼저 실행)")
            continue
        results.append(agg)

        std_str = f"{agg['std_accuracy']:.4f}" if agg["std_accuracy"] is not None else "N/A (seed 1개뿐)"
        print(
            f"{model_name:12s} n_seeds={agg['n_seeds']} (seed {agg['seeds_used']})  "
            f"mean_acc={agg['mean_accuracy']:.4f}  std_acc={std_str}  "
            f"mean_macro_f1={agg['mean_macro_f1']:.4f}  "
            f"best={agg['best_accuracy']:.4f}(seed{agg['best_seed']})  "
            f"worst={agg['worst_accuracy']:.4f}(seed{agg['worst_seed']})"
        )

    if not results:
        print("완료된 seed 결과가 하나도 없어서 집계할 것이 없음.")
        return

    df = pd.DataFrame(results)

    SUMMARY_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_RESULT_DIR / "seed_aggregate.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_path = SUMMARY_RESULT_DIR / "seed_aggregate.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 반복실험(멀티 seed) 집계\n\n")
        f.write(
            f"config.SEEDS = {SEEDS} 중 완료된 seed만 집계한 결과다. "
            "seed가 1개뿐인 모델은 표준편차를 계산하지 않는다(N/A).\n\n"
        )
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(f"\n저장: {csv_path}")
    print(f"저장: {md_path}")


if __name__ == "__main__":
    main()
