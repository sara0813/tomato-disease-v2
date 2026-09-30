"""그래프 생성 → results/figures/

    accuracy_comparison.png     V2 Tiny CNN 3종 + V1 인용 모델의 내부 정확도
    accuracy_comparison_tiny_cnn.png  V2 tiny_cnn_a/b/c 3종만 비교
    efficiency_tradeoff.png     파라미터 수(FLOPs) vs 정확도 산점도 (V2 3종만 — 실측치)
    efficiency_flops.png        모델별 FLOPs 막대그래프 (V2 3종 — 실측치)
    corruption_drop.png         조건·강도별 성능 저하율 (V2 3종)
    external_generalization.png 내부 vs 외부(Taiwan/Bangladesh) 정확도 격차 (V2 3종)
    confusion_matrix_<model>.png
    resolution_ablation_tiny_cnn_c.png  해상도(64/96/128/224)별 정확도·학습시간 비교 (RQ3)
    seed_stability.png           A/B/C seed(42/123/2026) 반복 결과 평균±표준편차
    interpolation_comparison_res<N>.png  tiny_cnn_c 해상도 N에서 보간법별 정확도
    resolution_interp_matrix_tiny_cnn_c.png  해상도 x 보간법 전체 조합 정확도 히트맵
    training_time_matrix_tiny_cnn_c.png  해상도 x 보간법 전체 조합 학습 시간 히트맵
    external_interp_matrix_tiny_cnn_c.png  해상도 x 보간법 전체 조합 외부 평균 정확도 히트맵

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
    EFFICIENCY_RESULT_DIR,
    FIGURE_DIR,
    SEEDS,
    SUMMARY_RESULT_DIR,
    TINY_MODELS,
    corruption_result_dir,
    external_result_dir,
    internal_result_dir,
)
from utils.io import load_json  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# PPT에 바로 넣을 수 있게 기본 폰트를 크게 잡는다.
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 22,
})

COLOR_V2 = "#2a78d6"    # V2에서 실제로 학습·측정한 모델
COLOR_V1 = "#898781"    # V1에서 인용한 참고 수치
MODEL_COLORS = {"tiny_cnn_a": "#86b6ef", "tiny_cnn_b": "#2a78d6", "tiny_cnn_c": "#eb6834"}
KO_NAMES = {c: CLASS_INFO[c]["name_ko"] for c in CLASS_NAMES}


def plot_accuracy_comparison():
    v2_acc = {m: load_json(internal_result_dir(m) / "metrics.json")["accuracy"] for m in TINY_MODELS}
    v1 = load_json(SUMMARY_RESULT_DIR / "v1_cited_reference.json")["internal"]

    rows = [(m, acc, "V2 (직접 학습)") for m, acc in v2_acc.items()]
    rows += [(m, d["accuracy"], "V1 (인용)") for m, d in v1.items()]
    df = pd.DataFrame(rows, columns=["model", "accuracy", "source"]).sort_values("accuracy")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = [COLOR_V2 if s == "V2 (직접 학습)" else COLOR_V1 for s in df["source"]]
    ax.barh(df["model"], df["accuracy"] * 100, color=colors)
    for y, v in enumerate(df["accuracy"] * 100):
        ax.text(v + 0.5, y, f"{v:.2f}%", va="center", fontsize=14, fontweight="bold")

    ax.set_xlabel("내부 테스트 Accuracy (%)")
    ax.set_xlim(0, 105)
    ax.set_title("내부 성능 비교: V2 Tiny CNN vs V1 인용 모델")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_V2), plt.Rectangle((0, 0), 1, 1, color=COLOR_V1)]
    ax.legend(handles, ["V2 (직접 학습·측정)", "V1 (1차 실험 인용, 재학습 안 함)"], loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)


def plot_accuracy_comparison_tiny_only():
    """V1 인용 모델 없이 V2 tiny_cnn_a/b/c 3종의 내부 정확도만 비교."""
    acc = {m: load_json(internal_result_dir(m) / "metrics.json")["accuracy"] * 100 for m in TINY_MODELS}
    order = sorted(acc, key=acc.get)  # 오름차순 (barh는 아래부터 쌓임)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(order, [acc[m] for m in order], color=[MODEL_COLORS[m] for m in order])
    for bar, m in zip(bars, order):
        v = acc[m]
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va="center", fontsize=16, fontweight="bold")

    ax.set_xlabel("내부 테스트 Accuracy (%)", fontsize=18, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_title("tiny_cnn A/B/C 정확도 비교")
    ax.tick_params(axis="both", labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "accuracy_comparison_tiny_cnn.png", dpi=150)
    plt.close(fig)


def plot_seed_stability():
    """A/B/C를 seed(42/123/2026) 3개로 반복 학습한 내부 정확도의 평균±표준편차.
    아직 안 돌린 seed가 있으면 있는 것만으로 평균을 낸다."""
    rows = []
    for m in TINY_MODELS:
        accs = []
        for seed in SEEDS:
            p = internal_result_dir(m, seed) / "metrics.json"
            if p.exists():
                accs.append(load_json(p)["accuracy"] * 100)
        if accs:
            rows.append({"model": m, "mean": np.mean(accs), "std": np.std(accs), "n": len(accs)})

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("mean")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(
        df["model"], df["mean"], xerr=df["std"], color=[MODEL_COLORS[m] for m in df["model"]],
        capsize=8, error_kw={"linewidth": 2.5, "ecolor": "black"},
    )
    for bar, mean, std, n in zip(bars, df["mean"], df["std"], df["n"]):
        label = f"{mean:.2f}% (n={n})" if n < 2 else f"{mean:.2f}% ± {std:.2f}"
        ax.text(mean + std + 0.5, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=14, fontweight="bold")

    ax.set_xlabel("내부 테스트 Accuracy (%)", fontsize=18, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_title("Seed 재현성 (평균 ± 표준편차)")
    ax.tick_params(axis="both", labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "seed_stability.png", dpi=150)
    plt.close(fig)


INTERP_METHODS = ("bilinear", "bicubic", "lanczos", "area")


def _tiny_cnn_c_interp_run_name(resolution: int, interp: str) -> str:
    # bilinear 기본 실행만 128px에서 --img-size 없이 돌렸다(run_name="tiny_cnn_c").
    # bicubic/lanczos/area는 128px에서도 항상 --img-size를 명시해서 돌렸으므로
    # run_name이 "tiny_cnn_c_res128_<interp>"로 남는다 — 128px라고 특별 취급하면 안 됨.
    if interp == "bilinear":
        return _tiny_cnn_c_run_name(resolution)
    return f"tiny_cnn_c_res{resolution}_{interp}"


def plot_interpolation_comparison(resolution: int):
    """tiny_cnn_c를 고정 해상도에서 리사이즈 보간법(bilinear/bicubic/lanczos/area)별로
    학습한 내부 정확도 비교. 아직 안 돌린 보간법이 있으면 있는 것만으로 그린다."""
    rows = []
    for interp in INTERP_METHODS:
        run_name = _tiny_cnn_c_interp_run_name(resolution, interp)
        p = internal_result_dir(run_name) / "metrics.json"
        if p.exists():
            rows.append({"interp": interp, "accuracy": load_json(p)["accuracy"] * 100})

    if len(rows) < 2:
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["interp"], df["accuracy"], color=MODEL_COLORS["tiny_cnn_c"])
    for bar, v in zip(bars, df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.2f}%", ha="center", fontsize=15, fontweight="bold")

    ax.set_xlabel("리사이즈 보간법", fontsize=18, fontweight="bold")
    ax.set_ylabel("내부 테스트 Accuracy (%)", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(df["accuracy"]) * 1.15)
    ax.set_title(f"tiny_cnn_c {resolution}px 보간법 비교")
    ax.tick_params(axis="both", labelsize=15)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"interpolation_comparison_res{resolution}.png", dpi=150)
    plt.close(fig)


def plot_resolution_interp_matrix(resolutions=(64, 96, 128, 224)):
    """tiny_cnn_c의 해상도 x 보간법 조합별 내부 테스트 정확도를 한 장의 표(히트맵)로 정리.
    아직 안 돌린 조합은 빈 칸으로 남긴다."""
    data = pd.DataFrame(index=INTERP_METHODS, columns=resolutions, dtype=float)
    for res in resolutions:
        for interp in INTERP_METHODS:
            run_name = _tiny_cnn_c_interp_run_name(res, interp)
            p = internal_result_dir(run_name) / "metrics.json"
            if p.exists():
                data.loc[interp, res] = load_json(p)["accuracy"] * 100

    if data.isna().all().all():
        return

    fig, ax = plt.subplots(figsize=(2.6 * len(resolutions) + 2, 2.2 * len(INTERP_METHODS) + 1.5))
    valid = data.values[~pd.isna(data.values)]
    vmin, vmax = (valid.min() - 1, valid.max() + 1) if len(valid) else (0, 100)
    im = ax.imshow(data.values, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(resolutions)))
    ax.set_xticklabels([f"{r}px" for r in resolutions])
    ax.set_yticks(range(len(INTERP_METHODS)))
    ax.set_yticklabels(INTERP_METHODS)
    ax.set_xlabel("입력 해상도", fontsize=18, fontweight="bold")
    ax.set_ylabel("리사이즈 보간법", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for i in range(len(INTERP_METHODS)):
        for j in range(len(resolutions)):
            v = data.values[i, j]
            text = f"{v:.2f}%" if not pd.isna(v) else "—"
            color = "white" if (not pd.isna(v) and v > (vmin + vmax) / 2) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=15, fontweight="bold", color=color)

    ax.set_title("tiny_cnn_c 해상도 x 보간법 내부 정확도")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "resolution_interp_matrix_tiny_cnn_c.png", dpi=150)
    plt.close(fig)


def plot_training_time_matrix(resolutions=(64, 96, 128, 224)):
    """tiny_cnn_c의 해상도 x 보간법 조합별 총 학습 시간(분)을 한 장의 표(히트맵)로 정리.
    경량화가 목표라 정확도뿐 아니라 학습 비용도 같이 봐야 해서 만든 그래프.
    아직 안 돌린 조합은 빈 칸으로 남긴다."""
    data = pd.DataFrame(index=INTERP_METHODS, columns=resolutions, dtype=float)
    for res in resolutions:
        for interp in INTERP_METHODS:
            run_name = _tiny_cnn_c_interp_run_name(res, interp)
            p = internal_result_dir(run_name) / "training_log.json"
            if p.exists():
                data.loc[interp, res] = load_json(p)["total_train_time_sec"] / 60

    if data.isna().all().all():
        return

    fig, ax = plt.subplots(figsize=(2.6 * len(resolutions) + 2, 2.2 * len(INTERP_METHODS) + 1.5))
    valid = data.values[~pd.isna(data.values)]
    vmin, vmax = (valid.min(), valid.max()) if len(valid) else (0, 1)
    im = ax.imshow(data.values, cmap="Oranges", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(resolutions)))
    ax.set_xticklabels([f"{r}px" for r in resolutions])
    ax.set_yticks(range(len(INTERP_METHODS)))
    ax.set_yticklabels(INTERP_METHODS)
    ax.set_xlabel("입력 해상도", fontsize=18, fontweight="bold")
    ax.set_ylabel("리사이즈 보간법", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for i in range(len(INTERP_METHODS)):
        for j in range(len(resolutions)):
            v = data.values[i, j]
            text = f"{v:.0f}분" if not pd.isna(v) else "—"
            color = "white" if (not pd.isna(v) and v > (vmin + vmax) / 2) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=15, fontweight="bold", color=color)

    ax.set_title("tiny_cnn_c 해상도 x 보간법 총 학습 시간")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "training_time_matrix_tiny_cnn_c.png", dpi=150)
    plt.close(fig)


EXTERNAL_DATASETS = ("taiwan", "bangladesh_bbox", "plantdoc")


def plot_external_interp_matrix(resolutions=(64, 96, 128, 224)):
    """tiny_cnn_c의 해상도 x 보간법 조합별 외부(Taiwan/Bangladesh/PlantDoc) 평균 정확도를
    한 장의 표(히트맵)로 정리. 아직 안 돌린 조합은 빈 칸으로 남긴다."""
    data = pd.DataFrame(index=INTERP_METHODS, columns=resolutions, dtype=float)
    for res in resolutions:
        for interp in INTERP_METHODS:
            run_name = _tiny_cnn_c_interp_run_name(res, interp)
            accs = []
            for ds in EXTERNAL_DATASETS:
                p = external_result_dir(ds, run_name) / "metrics.json"
                if p.exists():
                    accs.append(load_json(p)["accuracy"] * 100)
            if len(accs) == len(EXTERNAL_DATASETS):
                data.loc[interp, res] = sum(accs) / len(accs)

    if data.isna().all().all():
        return

    fig, ax = plt.subplots(figsize=(2.6 * len(resolutions) + 2, 2.2 * len(INTERP_METHODS) + 1.5))
    valid = data.values[~pd.isna(data.values)]
    vmin, vmax = (valid.min() - 1, valid.max() + 1) if len(valid) else (0, 100)
    im = ax.imshow(data.values, cmap="Greens", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(resolutions)))
    ax.set_xticklabels([f"{r}px" for r in resolutions])
    ax.set_yticks(range(len(INTERP_METHODS)))
    ax.set_yticklabels(INTERP_METHODS)
    ax.set_xlabel("입력 해상도", fontsize=18, fontweight="bold")
    ax.set_ylabel("리사이즈 보간법", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for i in range(len(INTERP_METHODS)):
        for j in range(len(resolutions)):
            v = data.values[i, j]
            text = f"{v:.2f}%" if not pd.isna(v) else "—"
            color = "white" if (not pd.isna(v) and v > (vmin + vmax) / 2) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=15, fontweight="bold", color=color)

    ax.set_title("tiny_cnn_c 해상도 x 보간법 외부(Taiwan/Bangladesh/PlantDoc) 평균 정확도")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "external_interp_matrix_tiny_cnn_c.png", dpi=150)
    plt.close(fig)


def plot_efficiency_tradeoff():
    eff = pd.read_csv(EFFICIENCY_RESULT_DIR / "efficiency.csv")
    acc = {m: load_json(internal_result_dir(m) / "metrics.json")["accuracy"] for m in TINY_MODELS}
    eff["accuracy"] = eff["model"].map(acc) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, x_col, x_label in [
        (axes[0], "params", "파라미터 수"),
        (axes[1], "flops_mflops", "FLOPs (M)"),
    ]:
        for _, row in eff.iterrows():
            ax.scatter(row[x_col], row["accuracy"], s=300, color=MODEL_COLORS[row["model"]], zorder=3)
            ax.annotate(
                row["model"], (row[x_col], row["accuracy"]),
                textcoords="offset points", xytext=(0, 14), ha="center", fontsize=14, fontweight="bold",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("내부 테스트 Accuracy (%)")
        ax.grid(True, alpha=0.3)

    axes[0].set_title("파라미터 수 vs 정확도")
    axes[1].set_title("FLOPs vs 정확도")
    fig.suptitle("효율성-정확도 트레이드오프", fontsize=24, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGURE_DIR / "efficiency_tradeoff.png", dpi=150)
    plt.close(fig)


def plot_efficiency_flops():
    """모델별 FLOPs를 가로 막대로 비교 (efficiency.csv 실측치, RQ3 — Depthwise Separable 효과)."""
    eff = pd.read_csv(EFFICIENCY_RESULT_DIR / "efficiency.csv").set_index("model")
    order = eff["flops_mflops"].sort_values().index.tolist()  # 적은 순 (barh는 아래부터 쌓이므로 뒤집어서 위쪽에 최소값)
    order = order[::-1]
    values = eff.loc[order, "flops_mflops"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(order, values, color=[MODEL_COLORS[m] for m in order])
    for bar, v in zip(bars, values):
        ax.text(v + values.max() * 0.015, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center", fontsize=16, fontweight="bold")

    ax.set_xlabel("FLOPs (M)", fontsize=18, fontweight="bold")
    ax.set_xlim(0, values.max() * 1.15)
    ax.set_title("모델별 FLOPs 비교")
    ax.tick_params(axis="both", labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "efficiency_flops.png", dpi=150)
    plt.close(fig)


def plot_corruption_drop():
    dfs = [pd.read_csv(corruption_result_dir(m) / "corruption_metrics.csv") for m in TINY_MODELS]
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
    ax.set_title("Corruption 조건별 평균 성능 저하율")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "corruption_drop.png", dpi=150)
    plt.close(fig)

    # 조건 x 강도 상세 (3단계 강도별로 펼친 버전)
    fig, axes = plt.subplots(1, len(TINY_MODELS), figsize=(18, 6), sharey=True)
    for ax, m in zip(axes, TINY_MODELS):
        sub = df[df["model"] == m]
        pivot = sub.pivot_table(index="corruption_type", columns="level", values="drop_rate_pct")
        pivot = pivot.reindex(conditions)
        pivot.plot(kind="bar", ax=ax, color=["#cde2fb", "#5598e7", "#184f95"], legend=(m == TINY_MODELS[-1]))
        ax.set_title(m)
        ax.set_xlabel("")
        ax.set_ylabel("저하율 (%)" if m == TINY_MODELS[0] else "")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("조건 x 강도별 성능 저하율", fontsize=24, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIGURE_DIR / "corruption_drop_by_level.png", dpi=150)
    plt.close(fig)


def plot_external_generalization():
    rows = []
    for m in TINY_MODELS:
        internal = load_json(internal_result_dir(m) / "metrics.json")["accuracy"]
        taiwan = load_json(external_result_dir("taiwan", m) / "metrics.json")["accuracy"]
        bangladesh = load_json(external_result_dir("bangladesh_bbox", m) / "metrics.json")["accuracy"]
        plantdoc = load_json(external_result_dir("plantdoc", m) / "metrics.json")["accuracy"]
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
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("내부 vs 외부 데이터 일반화 격차")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "external_generalization.png", dpi=150)
    plt.close(fig)


DA_METHODS = ("source", "adabn", "tent")
DA_METHOD_LABELS = {"source": "Source-only", "adabn": "AdaBN", "tent": "TENT"}
DA_METHOD_SUFFIX = {"source": "", "adabn": "_adabn", "tent": "_tent"}


def plot_domain_adaptation_comparison(base_run_name="tiny_cnn_c_res96_bicubic", datasets=EXTERNAL_DATASETS):
    """최종 모델(base_run_name)에 대해 source-only vs AdaBN vs TENT 외부 정확도를 데이터셋별로 비교.
    아직 안 돌린 조합은 빈 칸으로 남긴다."""
    rows = []
    for ds in datasets:
        for method in DA_METHODS:
            run_name = f"{base_run_name}{DA_METHOD_SUFFIX[method]}"
            p = external_result_dir(ds, run_name) / "metrics.json"
            if p.exists():
                rows.append({"dataset": ds, "method": method, "accuracy": load_json(p)["accuracy"] * 100})

    if not rows:
        return

    df = pd.DataFrame(rows).pivot(index="dataset", columns="method", values="accuracy").reindex(columns=DA_METHODS)
    df = df.reindex(datasets)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(df))
    width = 0.25
    colors = {"source": "#898781", "adabn": "#2a78d6", "tent": "#eb6834"}

    offset0 = -(len(DA_METHODS) - 1) / 2
    for i, method in enumerate(DA_METHODS):
        vals = df[method]
        bars = ax.bar(x + (offset0 + i) * width, vals, width, label=DA_METHOD_LABELS[method], color=colors[method])
        for b, v in zip(bars, vals):
            if not pd.isna(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("외부 Accuracy (%)", fontsize=16, fontweight="bold")
    ax.set_title("Domain Adaptation 방법별 외부 정확도 비교")
    ax.legend()
    ax.tick_params(axis="both", labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "domain_adaptation_comparison.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices():
    for m in TINY_MODELS:
        cm = pd.read_csv(internal_result_dir(m) / "confusion_matrix.csv", index_col=0)
        cm_norm = cm.div(cm.sum(axis=1), axis=0)  # 행(실제 클래스) 기준 정규화

        labels = [KO_NAMES[c] for c in cm.index]

        fig, ax = plt.subplots(figsize=(10.5, 9.5))
        im = ax.imshow(cm_norm.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)
        ax.set_xlabel("예측")
        ax.set_ylabel("실제")
        ax.set_title(f"Confusion Matrix — {m}")

        for i in range(len(labels)):
            for j in range(len(labels)):
                v = cm_norm.values[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                             color="white" if v > 0.5 else "black", fontsize=11)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"confusion_matrix_{m}.png", dpi=150)
        plt.close(fig)


def _tiny_cnn_c_run_name(resolution: int) -> str:
    return "tiny_cnn_c" if resolution == 128 else f"tiny_cnn_c_res{resolution}"


def plot_resolution_ablation(resolutions=(64, 96, 128, 224)):
    """tiny_cnn_c를 해상도별로 학습한 결과 비교 (RQ3: 해상도 축소 효과).

    train_model.py --img-size로 만든 run(run_name=f"tiny_cnn_c_res{H}")과
    기본 128px(run_name="tiny_cnn_c") 결과를 모아서 비교한다. 아직 안 돌린
    해상도가 있으면 있는 것만으로 그린다.
    """
    rows = []
    for res in resolutions:
        run_name = _tiny_cnn_c_run_name(res)
        metrics_path = internal_result_dir(run_name) / "metrics.json"
        log_path = internal_result_dir(run_name) / "training_log.json"
        if not metrics_path.exists() or not log_path.exists():
            continue
        metrics = load_json(metrics_path)
        log = load_json(log_path)
        rows.append({
            "resolution": res,
            "accuracy": metrics["accuracy"] * 100,
            "macro_f1": metrics["macro_f1"] * 100,
            "train_time_min": log["total_train_time_sec"] / 60,
        })

    if len(rows) < 2:
        return  # 비교할 데이터가 2개 미만이면 그리지 않음

    df = pd.DataFrame(rows).sort_values("resolution")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    axes[0].plot(df["resolution"], df["accuracy"], marker="o", markersize=10, linewidth=3, color=MODEL_COLORS["tiny_cnn_c"], label="Accuracy")
    axes[0].plot(df["resolution"], df["macro_f1"], marker="s", markersize=10, linewidth=3, color="#898781", label="Macro F1")
    for _, r in df.iterrows():
        axes[0].annotate(
            f"{r['accuracy']:.2f}%", (r["resolution"], r["accuracy"]),
            textcoords="offset points", xytext=(0, 12), ha="center", fontsize=13, fontweight="bold",
        )
    axes[0].set_xlabel("입력 해상도 (px)")
    axes[0].set_ylabel("내부 테스트 성능 (%)")
    axes[0].set_xticks(df["resolution"])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("해상도별 정확도")
    ymin, ymax = axes[0].get_ylim()  # 정확도 라벨이 제목이랑 안 겹치게 위쪽 여유 확보
    axes[0].set_ylim(ymin, ymax + (ymax - ymin) * 0.15)

    axes[1].bar(df["resolution"].astype(str), df["train_time_min"], color=MODEL_COLORS["tiny_cnn_c"])
    for i, v in enumerate(df["train_time_min"]):
        axes[1].text(i, v + 1, f"{v:.0f}분", ha="center", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("입력 해상도 (px)")
    axes[1].set_ylabel("총 학습 시간 (분)")
    axes[1].set_title("해상도별 학습 시간")

    fig.suptitle("tiny_cnn_c 해상도 비교", fontsize=24, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGURE_DIR / "resolution_ablation_tiny_cnn_c.png", dpi=150)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_accuracy_comparison()
    plot_accuracy_comparison_tiny_only()
    plot_efficiency_tradeoff()
    plot_efficiency_flops()
    plot_corruption_drop()
    plot_external_generalization()
    plot_confusion_matrices()
    plot_resolution_ablation()
    plot_seed_stability()
    for res in (64, 96, 128, 224):
        plot_interpolation_comparison(res)
    plot_resolution_interp_matrix()
    plot_training_time_matrix()
    plot_external_interp_matrix()
    plot_domain_adaptation_comparison()
    print(f"저장 완료: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
