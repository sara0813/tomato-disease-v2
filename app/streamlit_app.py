"""7단계: 토마토 잎 병해 분류 웹 프로토타입 (PyTorch).

잎 이미지를 업로드하면 병해 클래스와 한국어 설명(class_info.CLASS_INFO)을 보여준다.

아직 최종 모델을 선정하지 않았으므로(6단계 보류), 학습이 끝난 3개 Tiny CNN
(tiny_cnn_a/b/c) 중 하나를 직접 골라서 써볼 수 있게 만들었다. 나중에 최종
모델이 정해지면 DEFAULT_MODEL만 바꾸면 된다.

실행: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st
import torch

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_INFO, CLASS_NAMES  # noqa: E402
from config import model_path  # noqa: E402
from dataset import make_transform  # noqa: E402
from models import build_model, input_shape_for  # noqa: E402

st.set_page_config(page_title="토마토 잎 병해 분류", page_icon="🍅", layout="centered")

DEFAULT_MODEL = "tiny_cnn_b"  # 6단계에서 최종 모델이 정해지면 여기만 바꾸면 됨

MODEL_INFO = {
    "tiny_cnn_a": {"label": "Tiny CNN-A (최소형, 9.5만 파라미터)"},
    "tiny_cnn_b": {"label": "Tiny CNN-B (중간형, 24만 파라미터)"},
    "tiny_cnn_c": {"label": "Tiny CNN-C (Depthwise, 3.1만 파라미터)"},
}


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


def main():
    st.title("🍅 토마토 잎 병해 분류")
    st.caption("경량 CNN 기반 토마토 잎 병해 분류 · 강건성 평가 프로젝트 — 웹 프로토타입")

    with st.sidebar:
        st.header("모델 선택")
        st.caption("6단계(최종 모델 선정)가 아직 진행 전이라, 학습된 3개 모델을 직접 비교해볼 수 있습니다.")
        model_name = st.radio(
            "사용할 모델",
            options=list(MODEL_INFO.keys()),
            index=list(MODEL_INFO.keys()).index(DEFAULT_MODEL),
            format_func=lambda k: MODEL_INFO[k]["label"],
        )
        st.divider()
        st.caption("참고: 내부 test 정확도")
        st.markdown(
            "- tiny_cnn_a: 94.99%\n"
            "- tiny_cnn_b: 97.66%\n"
            "- tiny_cnn_c: 97.29%\n\n"
            "(외부 데이터에서는 세 모델 다 20% 안팎으로 급락합니다 — "
            "domain shift 문제가 아직 안 풀렸다는 뜻입니다.)"
        )

    model = load_model(model_name)
    if model is None:
        st.error(f"학습된 가중치가 없습니다: {model_path(model_name)}\n먼저 train_model.py를 실행하세요.")
        return

    uploaded = st.file_uploader("토마토 잎 이미지를 업로드하세요", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded is None:
        st.info("이미지를 업로드하면 분류 결과가 여기에 표시됩니다.")
        return

    from PIL import Image

    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.image(image, caption="업로드한 이미지", use_container_width=True)

    img_size = input_shape_for(model_name)[1:]
    probs = predict(model, image, img_size)

    top_idx = int(torch.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_prob = float(probs[top_idx])
    info = CLASS_INFO[top_class]

    with col2:
        status_emoji = "✅" if info["status"] == "정상" else "⚠️"
        st.subheader(f"{status_emoji} {info['name_ko']}")
        st.metric("확신도", f"{top_prob*100:.1f}%")
        st.write(info["description"])
        st.markdown(f"**권장 조치**: {info['recommendation']}")

    st.divider()
    st.subheader("클래스별 확률")
    prob_data = {
        CLASS_INFO[CLASS_NAMES[i]]["name_ko"]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }
    sorted_items = sorted(prob_data.items(), key=lambda kv: kv[1], reverse=True)
    for name, p in sorted_items:
        st.progress(p, text=f"{name}: {p*100:.1f}%")

    st.caption(
        "⚠️ 이 모델은 PlantVillage(실험실 환경) 데이터로만 학습했습니다. "
        "실제 야외 촬영 이미지에서는 정확도가 크게 떨어질 수 있습니다 (5단계 외부 평가 참고)."
    )


if __name__ == "__main__":
    main()
