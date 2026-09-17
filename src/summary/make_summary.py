"""6단계 준비: 종합 비교표 (최종 모델 선정은 하지 않음).

내부 성능 · 효율성 · corruption 저하율 · 외부 일반화, V2에서 직접 측정한 3개
Tiny CNN과 V1에서 인용한 5개 모델(results/summary/v1_cited_reference.json)을
한 표로 합친다. V1은 효율성/corruption을 측정하지 않았으므로 해당 칸은 비워둔다
(억지로 채우지 않음).

이 스크립트는 표만 만든다 — "어떤 모델이 최종이다"라는 결론/추천은 내리지 않는다.
그 판단은 별도로 사람과 상의해서 정한다.

산출물 → results/summary/
    model_comparison.csv
    model_comparison.md
"""

import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    EFFICIENCY_RESULT_DIR,
    SUMMARY_RESULT_DIR,
    TINY_MODELS,
    corruption_result_dir,
    external_result_dir,
    internal_result_dir,
)
from utils.io import load_json  # noqa: E402

COLUMNS = [
    "model", "group",
    "params", "flops_mflops", "file_size_mb", "cpu_inference_ms",
    "internal_accuracy", "internal_macro_f1", "internal_weighted_f1",
    "corruption_mean_drop_pct",
    "taiwan_accuracy", "bangladesh_accuracy", "plantdoc_accuracy",
]


def build_v2_rows() -> list[dict]:
    eff = pd.read_csv(EFFICIENCY_RESULT_DIR / "efficiency.csv").set_index("model")
    rows = []

    for m in TINY_MODELS:
        internal = load_json(internal_result_dir(m) / "metrics.json")
        taiwan = load_json(external_result_dir("taiwan", m) / "metrics.json")
        bangladesh = load_json(external_result_dir("bangladesh_bbox", m) / "metrics.json")
        plantdoc = load_json(external_result_dir("plantdoc", m) / "metrics.json")
        corruption = pd.read_csv(corruption_result_dir(m) / "corruption_metrics.csv")

        rows.append(
            {
                "model": m,
                "group": "V2 (직접 학습·측정)",
                "params": int(eff.loc[m, "params"]),
                "flops_mflops": eff.loc[m, "flops_mflops"],
                "file_size_mb": eff.loc[m, "file_size_mb"],
                "cpu_inference_ms": eff.loc[m, "cpu_inference_ms"],
                "internal_accuracy": internal["accuracy"],
                "internal_macro_f1": internal["macro_f1"],
                "internal_weighted_f1": internal["weighted_f1"],
                "corruption_mean_drop_pct": corruption["drop_rate_pct"].mean(),
                "taiwan_accuracy": taiwan["accuracy"],
                "bangladesh_accuracy": bangladesh["accuracy"],
                "plantdoc_accuracy": plantdoc["accuracy"],
            }
        )
    return rows


def build_v1_rows() -> list[dict]:
    v1 = load_json(SUMMARY_RESULT_DIR / "v1_cited_reference.json")
    internal = v1["internal"]
    taiwan = v1["external_taiwan"]
    bangladesh = v1["external_bangladesh_bbox"]

    rows = []
    for model_name, metrics in internal.items():
        rows.append(
            {
                "model": model_name,
                "group": "V1 (1차 실험 인용, 재학습 안 함)",
                "params": None,
                "flops_mflops": None,
                "file_size_mb": None,
                "cpu_inference_ms": None,
                "internal_accuracy": metrics["accuracy"],
                "internal_macro_f1": metrics["macro_f1"],
                "internal_weighted_f1": metrics["weighted_f1"],
                "corruption_mean_drop_pct": None,
                "taiwan_accuracy": taiwan.get(model_name),
                "bangladesh_accuracy": bangladesh.get(model_name),
                "plantdoc_accuracy": None,  # V1은 PlantDoc으로 평가한 적 없음
            }
        )
    return rows


def main() -> None:
    rows = build_v2_rows() + build_v1_rows()
    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("internal_accuracy", ascending=False)

    SUMMARY_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_RESULT_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 사람이 읽기 좋은 % 표기 버전으로 md 저장
    display_df = df.copy()
    for col in ["internal_accuracy", "internal_macro_f1", "internal_weighted_f1",
                "corruption_mean_drop_pct", "taiwan_accuracy", "bangladesh_accuracy", "plantdoc_accuracy"]:
        display_df[col] = display_df[col].map(lambda v: f"{v*100:.2f}%" if pd.notna(v) and col != "corruption_mean_drop_pct" else (f"{v:.2f}%" if pd.notna(v) else "—"))
    for col in ["params", "flops_mflops", "file_size_mb", "cpu_inference_ms"]:
        display_df[col] = display_df[col].map(lambda v: "—" if pd.isna(v) else v)

    md_path = SUMMARY_RESULT_DIR / "model_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 종합 비교표\n\n")
        f.write(
            "이 표는 비교 자료일 뿐, 최종 모델을 결정하지 않는다. "
            "V1 모델은 효율성/corruption을 측정하지 않아 해당 칸이 비어 있다(`—`).\n\n"
        )
        f.write(display_df.to_markdown(index=False))
        f.write("\n")

    print(df.to_string(index=False))
    print(f"\n저장: {csv_path}")
    print(f"저장: {md_path}")


if __name__ == "__main__":
    main()
