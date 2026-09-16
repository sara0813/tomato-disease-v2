"""results/_common/dataset_stats.csv를 읽어 클래스 분포 막대그래프를 PNG로 저장한다.

산출물 → results/figures/class_distribution.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_INFO  # noqa: E402
from config import COMMON_RESULT_DIR, FIGURE_DIR  # noqa: E402

# Windows 한글 폰트
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

COLOR_PV = "#2a78d6"
COLOR_TAIWAN = "#2a78d6"
COLOR_BD = "#eb6834"


def main() -> None:
    df = pd.read_csv(COMMON_RESULT_DIR / "dataset_stats.csv")
    wide = df.pivot(index="class", columns="dataset", values="count").fillna(0)
    wide["ko"] = [CLASS_INFO[c]["name_ko"] for c in wide.index]
    wide = wide.sort_values("raw_plantvillage", ascending=True)  # barh는 아래->위로 그려지므로 오름차순

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={"width_ratios": [1.3, 1, 1]})

    # 1) PlantVillage
    ax = axes[0]
    ax.barh(wide["ko"], wide["raw_plantvillage"], color=COLOR_PV)
    ax.set_title("PlantVillage (원본, 18,160장)")
    ax.set_xlabel("이미지 수")
    for y, v in enumerate(wide["raw_plantvillage"]):
        ax.text(v + 50, y, f"{int(v):,}", va="center", fontsize=8)

    # 2) Taiwan
    ax = axes[1]
    ax.barh(wide["ko"], wide["taiwan_external"], color=COLOR_TAIWAN)
    ax.set_title("Taiwan 외부셋 (314장)")
    ax.set_xlabel("이미지 수")
    ax.set_yticklabels([])
    for y, v in enumerate(wide["taiwan_external"]):
        ax.text(v + 2, y, f"{int(v)}", va="center", fontsize=8)

    # 3) Bangladesh
    ax = axes[2]
    ax.barh(wide["ko"], wide["bangladesh_bbox_external"], color=COLOR_BD)
    ax.set_title("Bangladesh 외부셋 (1,101장)")
    ax.set_xlabel("이미지 수")
    ax.set_yticklabels([])
    for y, v in enumerate(wide["bangladesh_bbox_external"]):
        ax.text(v + 5, y, f"{int(v)}", va="center", fontsize=8)

    fig.suptitle("토마토 병해 데이터셋 클래스 분포", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "class_distribution.png"
    fig.savefig(out_path, dpi=150)
    print(f"저장: {out_path.resolve()}")


if __name__ == "__main__":
    main()
