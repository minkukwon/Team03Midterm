"""AI 안경 탐지 시스템 GUI (Streamlit)
실행:  py -3.12 -m streamlit run app.py
"""
import streamlit as st
from PIL import Image

import config as C
from pipeline import AIGlassesSystem

st.set_page_config(page_title="AI 안경 탐지 시스템", page_icon="🕶️", layout="wide")

VERDICT_STYLE = {
    C.LABEL_AI: ("#e63946", "카메라 장착 AI 안경으로 판단됩니다. 감독관이 직접 확인하세요."),
    C.LABEL_SUSPECT: ("#f4a261", "AI 안경으로 의심됩니다. 감독관 확인 또는 재촬영이 필요합니다."),
    C.LABEL_NORMAL: ("#2a9d8f", "AI 안경이 아닌 것으로 추정됩니다."),
    C.LABEL_RETAKE: ("#6c757d", "안경을 탐지하지 못했습니다. 정면·측면 사진을 다시 촬영/입력하세요."),
}


@st.cache_resource(show_spinner="모델 불러오는 중...")
def load_system():
    return AIGlassesSystem()


def fmt(v):
    return "-" if v is None else f"{v:.2f}"


# ── 사이드바 ─────────────────────────────────────────
with st.sidebar:
    st.header("판정 기준")
    st.markdown(
        f"- 점수 **≥ {C.AI_THRESHOLD}** → 🔴 {C.LABEL_AI}\n"
        f"- **{C.NORMAL_THRESHOLD} < 점수 < {C.AI_THRESHOLD}** → 🟠 {C.LABEL_SUSPECT}\n"
        f"- 점수 **≤ {C.NORMAL_THRESHOLD}** → 🟢 {C.LABEL_NORMAL}"
    )
    if not C.DET_WEIGHTS.exists():
        st.error(f"탐지 모델이 없습니다: {C.DET_WEIGHTS}\n\ntrain.py 를 먼저 실행하세요.")
        st.stop()
    system = load_system()
    with st.expander("통합 판단 파라미터", expanded=False):
        st.caption("optimize_decision.py 로 최적화된 값" if C.DECISION_PARAMS.exists() else "기본값 (최적화 전)")
        st.json(system.params)
    st.caption("※ 본 시스템은 보조 도구이며 부정행위를 자동으로 확정하지 않습니다.")

st.title("🕶️ AI 안경 탐지 시스템")
st.caption("응시자의 정면·측면 사진으로 카메라 장착 AI 안경 착용 여부를 판단합니다. (YOLO11m 탐지 + YOLO11m-cls 분류)")

mode = st.radio("입력 방식", ["이미지 업로드", "카메라 촬영"], horizontal=True)

cols = st.columns(2)
inputs = {}
for col, key, label in ((cols[0], "front", "정면 사진"), (cols[1], "side", "측면 사진")):
    with col:
        st.subheader(label)
        if mode == "이미지 업로드":
            f = st.file_uploader(label, type=["jpg", "jpeg", "png", "bmp", "webp"], key=f"up_{key}",
                                 label_visibility="collapsed")
        else:
            f = st.camera_input(label, key=f"cam_{key}", label_visibility="collapsed")
        if f is not None:
            inputs[key] = Image.open(f).convert("RGB")
            if mode == "이미지 업로드":
                st.image(inputs[key], width="stretch")

ready = len(inputs) == 2
if not ready:
    st.info("정면 사진과 측면 사진을 모두 입력하세요.")

if st.button("🔍 분석 실행", type="primary", disabled=not ready, width="stretch"):
    with st.spinner("분석 중..."):
        st.session_state["result"] = system.analyze(inputs["front"], inputs["side"])

res = st.session_state.get("result")
if res and ready:
    color, msg = VERDICT_STYLE[res["verdict"]]
    score_txt = "-" if res["score"] is None else f"{res['score']:.3f}"
    st.markdown(
        f"""<div style="background:{color};color:white;padding:22px;border-radius:12px;text-align:center;margin:12px 0">
        <div style="font-size:15px;opacity:.9">최종 판정</div>
        <div style="font-size:40px;font-weight:700">{res['verdict']}</div>
        <div style="font-size:18px">통합 점수 {score_txt} · {res['fusion']}</div>
        <div style="font-size:15px;margin-top:6px">{msg}</div></div>""",
        unsafe_allow_html=True,
    )
    if res["score"] is not None:
        st.progress(min(max(res["score"], 0.0), 1.0))

    rc = st.columns(2)
    for col, key, label in ((rc[0], "front", "정면"), (rc[1], "side", "측면")):
        v = res[key]
        with col:
            st.subheader(f"{label} 분석 결과")
            st.image(v["image"], width="stretch")
            m = st.columns(4)
            m[0].metric("이미지 점수", fmt(v["score"]))
            m[1].metric("탐지(AI)", fmt(v["p_det"]))
            m[2].metric("분류(AI)", fmt(v["p_cls"]))
            m[3].metric("렌즈", fmt(v["lens_conf"]) if v["lens_found"] else "미탐지")
            if v["score"] is None:
                st.warning(f"{label}: {v['status']}")
            else:
                st.caption(f"상태: {v['status']}")
            if v["boxes"]:
                st.dataframe(
                    [{"클래스": C.DET_NAMES[b["cls"]], "신뢰도": round(b["conf"], 3),
                      "분류 P(AI)": None if b.get("p_cls") is None else round(b["p_cls"], 3)} for b in v["boxes"]],
                    width="stretch", hide_index=True)
    if res["front"]["score"] is not None and res["side"]["score"] is not None \
            and abs(res["front"]["score"] - res["side"]["score"]) >= 0.4:
        st.warning("정면·측면 결과가 크게 다릅니다. 통합 판단 기준을 적용했으며, 감독관 확인을 권장합니다.")
