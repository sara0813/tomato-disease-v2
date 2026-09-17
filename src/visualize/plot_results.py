"""그래프 생성 → results/figures/

    accuracy_comparison.png     V2 Tiny CNN 3종 + V1 인용 모델의 내부 정확도
    efficiency_tradeoff.png     파라미터 수(FLOPs) vs 정확도 산점도 (V2 3종만 — 실측치)
    corruption_drop.png         조건·강도별 성능 저하율 (V2 3종)
    external_generalization.png 내부 vs 외부(Taiwan/Bangladesh) 정확도 격차 (V2 3종)
    confusion_matrix_<model>.png

V1 인용 모델(results/summary/v1_cited_reference.json)은 정확도만 있고
효율성/corruption 수치가 없다 — 그 항목 그래프에는 아예 넣지 않는다
(측정 안 한 걸 억지로 채우지 않음).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_INFO, CLASS_NAMES  # noqa: E402
from config import (  # noqa: E402
    CORRUPTION_RESULT_DIR,
    EFFICIENCY_RESULT_DIR,
    EXTERNAL_RESULT_DIR,
    FIGURE_DIR,
    INTERNAL_RESULT_DIR,
    SUMMARY_RESULT_DIR,
    TINY_MODELS,
)
from utils.io import load_json  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

COLOR_V2 = "#2a78d6"    # V2에서 실제로 학습·측정한 모델
COLOR_V1 = "#898781"    # V1에서 인용한 참고 수치
MODEL_COLORS = {"tiny_cnn_a": "#86b6ef", "tiny_cnn_b": "#2a78d6", "tiny_cnn_c": "#eb6834"}
KO_NAMES = {c: CLASS_INFO[c]["name_ko"] for c in CLASS_NAMES}


def plot_accuracy_comparison():
    v2_acc = {m: load_json(INTERNAL_RESULT_DIR / m / "metrics.json")["accuracy"] for m in TINY_MODELS}
    v1 = load_json(SUMMARY_RESULT_DIR / "v1_cited_reference.json")["internal"]

    rows = [(m, acc, "V2 (직접 학습)") for m, acc in v2_acc.items()]
    rows += [(m, d["accuracy"], "V1 (인용)") for m, d in v1.items()]
    df = pd.DataFrame(rows, columns=["model", "accuracy", "source"]).sort_values("accuracy")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [COLOR_V2 if s == "V2 (직접 학습)" else COLOR_V1 for s in df["source"]]
    ax.barh(df["model"], df["accuracy"] * 100, color=colors)
    for y, v in enumerate(df["accuracy"] * 100):
        ax.text(v + 0.5, y, f"{v:.2f}%", va="center", fontsize=9)

    ax.set_xlabel("내부 테스트 Accuracy (%)")
    ax.set_xlim(0, 105)
    ax.set_title("내부 성능 비교: V2 Tiny CNN vs V1 인용 모델")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_V2), plt.Rectangle((0, 0), 1, 1, color=COLOR_V1)]
    ax.legend(handles, ["V2 (직접 학습·측정)", "V1 (1차 실험 인용, 재학습 안 함)"], loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)


def plot_efficiency_tradeoff():
    eff = pd.read_csv(EFFICIENCY_RESULT_DIR / "efficiency.csv")
    acc = {m: load_json(INTERNAL_RESULT_DIR / m / "metrics.json")["accuracy"] for m in TINY_MODELS}
    eff["accuracy"] = eff["model"].map(acc) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, x_col, x_label in [
        (axes[0], "params", "파라미터 수"),
        (axes[1], "flops_mflops", "FLOPs (M)"),
    ]:
        for _, row in eff.iterrows():
            ax.scatter(row[x_col], row["accuracy"], s=220, color=MODEL_COLORS[row["model"]], zorder=3)
            ax.annotate(
                row["model"], (row[x_col], row["accuracy"]),
                textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("내부 테스트 Accuracy (%)")
        ax.grid(True, alpha=0.3)

    axes[0].set_title("파라미터 수 vs 정확도")
    axes[1].set_title("FLOPs vs 정확도")
    fig.suptitle("효율성-정확도 트레이드오프 (V2 Tiny CNN 3종)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURE_DIR / "efficiency_tradeoff.png", dpi=150)
    plt.close(fig)


def plot_corruption_drop():
    dfs = [pd.read_csv(CORRUPTION_RESULT_DIR / m / "corruption_metrics.csv") for m in TINY_MODELS]
    df = pd.concat(dfs, ignore_index=True)

    conditions = df.groupby("corruption_type")["drop_rate_pct"].mean().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(conditions))
    width = 0.25

    for i, m in enumerate(TINY_MODELS):
        sub = df[df["model"] == m].groupby("corruption_type")["drop_rate_pct"].mean().reindex(conditions)
        ax.bar(x + (i - 1) * width, sub.values, width, label=m, color=MODEL_COLORS[m])

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("평균 성능 저하율 (%)")
    ax.set_title("Corruption 조건별 평균 성능 저하율 (3단계 강도 평균)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "corruption_drop.png", dpi=150)
    plt.close(fig)

    # 조건 x 강도 상세 (3단계 강도별로 펼친 버전)
    fig, axes = plt.subplots(1, len(TINY_MODELS), figsize=(16, 5), sharey=True)
    for ax, m in zip(axes, TINY_MODELS):
        sub = df[df["model"] == m]
        pivot = sub.pivot_table(index="corruption_type", columns="level", values="drop_rate_pct")
        pivot = pivot.reindex(conditions)
        pivot.plot(kind="bar", ax=ax, color=["#cde2fb", "#5598e7", "#184f95"], legend=(m == TINY_MODELS[-1]))
        ax.set_title(m)
        ax.set_xlabel("")
        ax.set_ylabel("저하율 (%)" if m == TINY_MODELS[0] else "")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("조건 x 강도별 성능 저하율", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGURE_DIR / "corruption_drop_by_level.png", dpi=150)
    plt.close(fig)


def plot_external_generalization():
    rows = []
    for m in TINY_MODELS:
        internal = load_json(INTERNAL_RESULT_DIR / m / "metrics.json")["accuracy"]
        taiwan = load_json(EXTERNAL_RESULT_DIR / "taiwan" / m / "metrics.json")["accuracy"]
        bangladesh = load_json(EXTERNAL_RESULT_DIR / "bangladesh_bbox" / m / "metrics.json")["accuracy"]
        plantdoc = load_json(EXTERNAL_RESULT_DIR / "plantdoc" / m / "metrics.json")["accuracy"]
        rows.append({
            "model": m, "내부 (PlantVillage)": internal, "Taiwan": taiwan,
            "Bangladesh": bangladesh, "PlantDoc": plantdoc,
        })

    df = pd.DataFrame(rows).set_index("model") * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(df))
    width = 0.19
    cols = df.columns.tolist()
    colors = ["#2a78d6", "#eb6834", "#e34948", "#1baf7a"]

    offset0 = -(len(cols) - 1) / 2
    for i, col in enumerate(cols):
        bars = ax.bar(x + (offset0 + i) * width, df[col], width, label=col, color=colors[i])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("내부 vs 외부 데이터 일반화 격차")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "external_generalization.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices():
    for m in TINY_MODELS:
        cm = pd.read_csv(INTERNAL_RESULT_DIR / m / "confusion_matrix.csv", index_col=0)
        cm_norm = cm.div(cm.sum(axis=1), axis=0)  # 행(실제 클래스) 기준 정규화

        labels = [KO_NAMES[c] for c in cm.index]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_norm.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("예측")
        ax.set_ylabel("실제")
        ax.set_title(f"Confusion Matrix — {m}")

        for i in range(len(labels)):
            for j in range(len(labels)):
                v = cm_norm.values[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                             color="white" if v > 0.5 else "black", fontsize=7)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"confusion_matrix_{m}.png", dpi=150)
        plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_accuracy_comparison()
    plot_efficiency_tradeoff()
    plot_corruption_drop()
    plot_external_generalization()
    plot_confusion_matrices()
    print(f"저장 완료: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
