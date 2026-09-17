"""7단계: 토마토 잎 병해 분류 웹 프로토타입 (PyTorch).

잎 이미지를 업로드하면 정상/비정상을 색으로 먼저 크게 보여주고, 병해 클래스와
한국어 설명(class_info.CLASS_INFO)을 이어서 보여준다.

아직 최종 모델을 선정하지 않았으므로(6단계 보류), 학습이 끝난 3개 Tiny CNN
(tiny_cnn_a/b/c) 중 하나를 직접 골라서 써볼 수 있게 만들었다. 나중에 최종
모델이 정해지면 DEFAULT_MODEL만 바꾸면 된다.

디자인은 docs/의 프로젝트 보고서(PDF)와 같은 톤(남색 헤더, eyebrow 캡션,
컬러 콜아웃 박스)을 따라서 같은 프로젝트 문서군처럼 보이게 했다.

실행: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_INFO, CLASS_NAMES  # noqa: E402
from config import model_path  # noqa: E402
from dataset import make_transform  # noqa: E402
from models import build_model, input_shape_for  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 팔레트 (docs/ 보고서와 동일 톤)
# ============================================================
NAVY = "#1c2d4a"
BLUE = "#2a5db0"
GOOD_GREEN = "#0ca30c"
GOOD_GREEN_BG = "#eaf7ea"
CRITICAL_RED = "#c0392b"
CRITICAL_RED_BG = "#fbeceb"
MUTED = "#6b7280"
SURFACE = "#fafaf9"

DEFAULT_MODEL = "tiny_cnn_b"  # 6단계에서 최종 모델이 정해지면 여기만 바꾸면 됨

MODEL_INFO = {
    "tiny_cnn_a": {"label": "Tiny CNN-A · 최소형", "params": "94,762", "acc": "94.99%"},
    "tiny_cnn_b": {"label": "Tiny CNN-B · 중간형", "params": "242,474", "acc": "97.66%"},
    "tiny_cnn_c": {"label": "Tiny CNN-C · Depthwise", "params": "31,498", "acc": "97.29%"},
}

st.set_page_config(page_title="토마토 잎 병해 분류", page_icon="🍅", layout="centered")


# ============================================================
# 전역 CSS — docs/ 보고서와 같은 톤 (남색 헤더바, eyebrow, 콜아웃 박스)
# ============================================================
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: "Segoe UI", "Malgun Gothic", sans-serif;
    }}
    .block-container {{
        padding-top: 1.2rem;
        max-width: 760px;
    }}
    #MainMenu, footer {{ visibility: hidden; }}

    .topbar {{
        height: 6px;
        background: {NAVY};
        border-radius: 3px;
        margin-bottom: 1.4rem;
    }}
    .eyebrow {{
        color: {MUTED};
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }}
    .hero-title {{
        color: {NAVY};
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        line-height: 1.25;
    }}
    .hero-sub {{
        color: {MUTED};
        font-size: 0.88rem;
        margin-bottom: 1.6rem;
    }}
    .section-label {{
        color: {BLUE};
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-bottom: 2px solid {BLUE};
        display: inline-block;
        padding-bottom: 3px;
        margin: 1.4rem 0 0.9rem 0;
    }}

    .status-card {{
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        border-left: 6px solid;
    }}
    .status-card.good {{
        background: {GOOD_GREEN_BG};
        border-left-color: {GOOD_GREEN};
    }}
    .status-card.bad {{
        background: {CRITICAL_RED_BG};
        border-left-color: {CRITICAL_RED};
    }}
    .status-label {{
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}
    .status-label.good {{ color: {GOOD_GREEN}; }}
    .status-label.bad {{ color: {CRITICAL_RED}; }}
    .status-title {{
        font-size: 1.5rem;
        font-weight: 800;
        color: #1a1a1a;
        margin-bottom: 0.15rem;
    }}
    .status-conf {{
        font-size: 0.85rem;
        color: {MUTED};
    }}

    .detail-card {{
        background: {SURFACE};
        border: 1px solid #e5e4e0;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        line-height: 1.6;
    }}
    .detail-card b {{ color: {NAVY}; }}

    .model-badge {{
        display: inline-block;
        background: {NAVY};
        color: white;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 0.8rem;
    }}

    .disclaimer {{
        background: #fff8e6;
        border-left: 4px solid #c98500;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.78rem;
        color: #6b5100;
        margin-top: 1.2rem;
    }}
    </style>
    <div class="topbar"></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    weights_path = model_path(model_name)
    if not weights_path.exists():
        return None
    model = build_model(model_name)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def predict(model, image, img_size):
    transform = make_transform(img_size)
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    return probs


def render_prob_chart(probs, top_class):
    items = sorted(
        ((CLASS_INFO[c]["name_ko"], float(probs[i]), c == top_class) for i, c in enumerate(CLASS_NAMES)),
        key=lambda t: t[1],
    )
    names = [t[0] for t in items]
    values = [t[1] * 100 for t in items]
    colors = [("#eb6834" if t[2] else "#9ec5f4") for t in items]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(names, values, color=colors, height=0.65)
    for b, v in zip(bars, values):
        if v > 1:
            ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8, color="#333")
    ax.set_xlim(0, 108)
    ax.set_xlabel("확률 (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def main():
    st.markdown('<div class="eyebrow">CAPSTONE DESIGN · WEB PROTOTYPE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">🍅 토마토 잎 병해 분류</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">경량 CNN 기반 토마토 잎 병해 분류 및 강건성 평가 — 7단계 웹 프로토타입</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("**모델 선택**")
        st.caption("6단계(최종 모델 선정)가 아직 진행 전이라, 학습된 3개 모델을 직접 비교해볼 수 있습니다.")
        model_name = st.radio(
            "사용할 모델",
            options=list(MODEL_INFO.keys()),
            index=list(MODEL_INFO.keys()).index(DEFAULT_MODEL),
            format_func=lambda k: MODEL_INFO[k]["label"],
            label_visibility="collapsed",
        )
        info = MODEL_INFO[model_name]
        st.markdown(
            f"""
            <div style="background:{SURFACE};border:1px solid #e5e4e0;border-radius:8px;
                        padding:0.8rem 1rem;font-size:0.8rem;line-height:1.7;">
                파라미터 <b>{info['params']}</b><br>
                내부 test acc <b>{info['acc']}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption(
            "⚠️ 외부 데이터(Taiwan/Bangladesh)에서는 세 모델 다 20% 안팎으로 정확도가 급락합니다. "
            "domain shift 문제가 아직 해결되지 않았습니다."
        )

    model = load_model(model_name)
    if model is None:
        st.error(f"학습된 가중치가 없습니다: {model_path(model_name)}\n먼저 train_model.py를 실행하세요.")
        return

    st.markdown('<div class="section-label">1. 이미지 업로드</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "토마토 잎 이미지를 업로드하세요", type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("이미지를 업로드하면 분류 결과가 여기에 표시됩니다.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="업로드한 이미지", width=280)

    img_size = input_shape_for(model_name)[1:]
    probs = predict(model, image, img_size)

    top_idx = int(torch.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_prob = float(probs[top_idx])
    info = CLASS_INFO[top_class]
    is_good = info["status"] == "정상"

    st.markdown('<div class="section-label">2. 진단 결과</div>', unsafe_allow_html=True)

    status_class = "good" if is_good else "bad"
    status_icon = "✅" if is_good else "⚠️"
    status_word = "정상" if is_good else "병해 의심"
    st.markdown(
        f"""
        <div class="status-card {status_class}">
            <div class="status-label {status_class}">{status_icon} {status_word}</div>
            <div class="status-title">{info['name_ko']}</div>
            <div class="status-conf">확신도 {top_prob*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="detail-card">
            <b>증상</b><br>{info['description']}<br><br>
            <b>권장 조치</b><br>{info['recommendation']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">3. 클래스별 확률</div>', unsafe_allow_html=True)
    fig = render_prob_chart(probs, top_class)
    st.pyplot(fig, use_container_width=True)

    st.markdown(
        """
        <div class="disclaimer">
            이 모델은 PlantVillage(실험실 환경) 데이터로만 학습했습니다.
            실제 야외 촬영 이미지에서는 정확도가 크게 떨어질 수 있습니다 (5단계 외부 평가 참고).
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
