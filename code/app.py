"""
AI 안경 탐지 시스템 - 웹 GUI (Streamlit)

실행
  streamlit run app.py
  → 브라우저에서 http://localhost:8501

모델 파일 위치 (기본: 이 파일 옆의 models 폴더, 환경변수 GLASSES_MODEL_DIR 로 변경 가능)
  models/best.pt            YOLO11m 가중치
  models/rf_glasses.joblib  Random Forest 분류기
  models/best_config.json   rf_fusion.py 가 고른 최적 통합 방식과 임계값
"""
import json
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from ultralytics import YOLO

from cv_pipeline import meta_vector, raw_scores
from rf_fusion import AI, LENS, NORMAL, decide3, fuse


def analyze(model, rf, name, det_conf, img):
    """YOLO 탐지 → 점수 (cv_pipeline 과 같은 특징 사용) → RF AI 확률"""
    r = raw_scores(model, name, det_conf=det_conf, img=img)
    r["image"] = name
    if r["detected"]:
        r["rf_p"] = float(rf.predict_proba(meta_vector(r)[None])[0, 1])
    return r

MODEL_DIR = Path(os.environ.get("GLASSES_MODEL_DIR", Path(__file__).with_name("models")))
CLASS_NAME = {AI: "AIGlasses", LENS: "AIGlasses_Lens", NORMAL: "NormalGlasses"}
CLASS_COLOR = {AI: (60, 60, 230), LENS: (0, 200, 255), NORMAL: (60, 180, 60)}  # BGR
METHOD_KO = {"mean": "평균", "max": "최댓값", "weighted": "가중평균", "condition": "조건 기반(렌즈 탐지 반영)",
             "condition3": "조건 기반 3단계(렌즈 → AI 의심 / AI형 모양 → 확인 필요 / 그 외 → 일반)"}
SCEN_KO = {"A": "YOLO11m 단독", "D": "YOLO11m + Random Forest"}

st.set_page_config(page_title="AI 안경 탐지", page_icon="👓", layout="wide")


@st.cache_resource
def load_models():
    cfg = json.loads((MODEL_DIR / "best_config.json").read_text(encoding="utf-8"))
    return YOLO(str(MODEL_DIR / "best.pt")), joblib.load(MODEL_DIR / "rf_glasses.joblib"), cfg


def decode(file):
    if file is None:
        return None
    data = np.frombuffer(file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def draw(img, res):
    out = img.copy()
    t = max(2, min(out.shape[:2]) // 250)
    for c, conf, (x1, y1, x2, y2) in res.get("dets", []):
        col = CLASS_COLOR.get(c, (200, 200, 200))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, t)
        label = f"{CLASS_NAME.get(c, c)} {conf:.2f}"
        fs = max(0.5, min(out.shape[:2]) / 900)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        cv2.rectangle(out, (int(x1), int(y1) - th - 8), (int(x1) + tw + 6, int(y1)), col, -1)
        cv2.putText(out, label, (int(x1) + 3, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def image_input(label, mode, key):
    if mode == "이미지 업로드":
        return st.file_uploader(label, type=["png", "jpg", "jpeg", "webp", "bmp"], key=key)
    return st.camera_input(label, key=key)


# ---------------------------------------------------------------- 화면
st.title("👓 AI 안경 탐지 시스템")
st.caption("응시자의 정면·측면 사진으로 카메라가 달린 AI 안경 착용 여부를 판단하는 보조 도구입니다. "
           "부정행위를 자동으로 확정하지 않으며, 의심·불확실한 결과는 감독관이 직접 확인해야 합니다.")

try:
    model, rf, cfg = load_models()
except FileNotFoundError as e:
    st.error(f"모델 파일을 찾을 수 없습니다: {e.filename}\n\n`{MODEL_DIR}` 폴더에 best.pt, rf_glasses.joblib, "
             "best_config.json 을 넣거나 환경변수 GLASSES_MODEL_DIR 로 위치를 지정하세요.")
    st.stop()

with st.sidebar:
    st.header("설정")
    mode = st.radio("입력 방식", ["이미지 업로드", "카메라 촬영"])
    det_conf = st.slider("YOLO 탐지 신뢰도 기준", 0.05, 0.9, float(cfg.get("yolo_det_conf", 0.25)), 0.05)
    three_way = cfg["method"] == "condition3"
    if three_way:
        lens_t = st.slider("렌즈 판정 기준 (이상이면 AI 의심)", 0.1, 0.95, float(cfg["params"]["lens_t"]), 0.05)
        ai_t = st.slider("AI형 모양 기준 (렌즈 없이 이상이면 확인 필요)", 0.1, 0.95, float(cfg["params"]["ai_t"]), 0.05)
    else:
        margin = st.slider("'확인 필요' 구간 폭 (임계값 ±)", 0.0, 0.3, 0.10, 0.01,
                           help="통합 점수가 임계값 근처이면 판정을 미루고 '확인 필요'로 표시합니다.")
    st.divider()
    st.markdown(f"**판정 방식**  \n{SCEN_KO.get(cfg['scenario'], cfg['scenario'])}  \n"
                f"통합: {METHOD_KO.get(cfg['method'], cfg['method'])}")

c1, c2 = st.columns(2)
with c1:
    st.subheader("정면 사진")
    front_file = image_input("정면 사진을 올리거나 촬영하세요", mode, "front")
with c2:
    st.subheader("측면 사진")
    side_file = image_input("측면 사진을 올리거나 촬영하세요", mode, "side")

front, side = decode(front_file), decode(side_file)
run = st.button("분석하기", type="primary", disabled=front is None or side is None, use_container_width=True)
if front is None or side is None:
    st.info("정면 사진과 측면 사진을 모두 넣으면 분석할 수 있습니다.")

if run:
    with st.spinner("분석 중..."):
        rf_res = analyze(model, rf, "front", det_conf, front)
        rs_res = analyze(model, rf, "side", det_conf, side)
        key = "yolo_p" if cfg["scenario"] == "A" else "rf_p"
        if three_way:
            verdict, score = decide3(rf_res, rs_res, key, lens_t, ai_t)
        else:
            score = fuse(cfg["method"], rf_res, rs_res, cfg["scenario"], cfg["params"])
            t = cfg["threshold"]
            verdict = None if score is None else ("AI" if score >= t + margin else "Normal" if score < t - margin else "check")

    for col, title, img, res in ((c1, "정면", front, rf_res), (c2, "측면", side, rs_res)):
        with col:
            st.image(draw(img, res), caption=f"{title} 탐지 결과", use_container_width=True)
            if res["detected"]:
                m1, m2, m3 = st.columns(3)
                m1.metric("AI 안경 점수", f"{res[key]:.2f}")
                m2.metric("렌즈 탐지", f"{res['lens']:.2f}" if res["lens"] > 0 else "없음")
                m3.metric("안경 탐지 신뢰도", f"{res['glasses_conf']:.2f}")
            else:
                st.warning(f"{title} 사진에서 안경을 찾지 못했습니다. 안경이 잘 보이게 다시 촬영해 주세요.")

    st.divider()
    if verdict is None:
        st.error("### 판정 불가 — 재촬영 필요\n두 사진 모두에서 안경을 탐지하지 못했습니다. 밝은 곳에서 안경이 잘 보이게 다시 촬영하세요. "
                 "재촬영 후에도 판정되지 않으면 관리자가 직접 확인하세요.")
    else:
        lens_max = max([r["lens"] for r in (rf_res, rs_res) if r["detected"]], default=0.0)
        st.progress(min(max(score, 0.0), 1.0), text=f"AI 안경 점수(평균) {score:.2f} · 렌즈 최대 신뢰도 {lens_max:.2f}")
        if verdict == "AI":
            st.error("### 🚨 AI 안경 의심\n안경 테에서 카메라 렌즈가 탐지되었습니다. 감독관이 직접 확인하세요.")
        elif verdict == "Normal":
            st.success("### ✅ 일반 안경 추정\n카메라 렌즈나 AI 안경의 특징이 충분히 발견되지 않았습니다.")
        else:
            st.warning("### ⚠️ 확인 필요\n안경 모양은 AI 안경과 비슷하지만 카메라 렌즈가 확인되지 않았습니다. "
                       "두꺼운 뿔테 일반 안경일 수도 있으니 감독관이 직접 확인하세요.")
        if not (rf_res["detected"] and rs_res["detected"]):
            st.caption("한쪽 사진에서만 안경이 탐지되어, 탐지된 사진만으로 판정했습니다.")
        st.caption("※ 카메라 렌즈가 보이지 않는다는 이유만으로 일반 안경이라고 확정하지 않습니다.")
