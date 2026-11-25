"""탐지(YOLO11m) → 분류(YOLO11m-cls) → 정면·측면 통합 판단 파이프라인

한 장(view)의 점수 산출
  1. YOLO11m 탐지 (낮은 conf로 후보를 모두 받은 뒤 det_conf로 필터)
  2. 겹치는 AIGlasses / NormalGlasses 박스를 하나의 '안경 영역'으로 묶음
       p_det = 0.5 + 0.5 * (AI 신뢰도 - Normal 신뢰도)            ∈ [0, 1]
  3. 안경 영역을 잘라 YOLO11m-cls 로 분류 → p_cls = P(AIGlasses)
  4. p = w_cls * p_cls + (1 - w_cls) * p_det     (가장 확실하게 탐지된 안경 영역 사용)
  5. 카메라 렌즈(AIGlasses_Lens) 신뢰도 L >= lens_conf 이면
       p = p + (1 - p) * lens_weight * L          (렌즈 증거로 점수 상향)
  - 안경 없음 + NoGlasses 탐지 → 0.0 (안경 미착용)
  - 아무것도 탐지 안 됨 → None (재촬영 요청)

정면·측면 통합 (fusion)
  mean     : 평균
  max      : 최댓값
  weighted : w_front * 정면 + (1 - w_front) * 측면
  cond     : 어느 한쪽이라도 렌즈가 확인되면 max, 아니면 weighted   (조건 기반)
  한쪽이 None이면 다른 쪽 점수만 사용, 둘 다 None이면 재촬영.

최종 판정: >= 0.8 AI 안경 / <= 0.5 AI 안경 아님 / 그 사이 AI 안경 의심
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config as C

RAW_CONF = 0.05      # 후보 수집용 최소 conf (det_conf 최적화를 위해 낮게)
MAX_CROPS = 5
PAD = 0.10


def to_pil(img):
    if isinstance(img, (str, Path)):
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")   # RGB ndarray 가정
    return img.convert("RGB")


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_params():
    p = dict(C.DEFAULT_PARAMS)
    if C.DECISION_PARAMS.exists():
        p.update(json.loads(C.DECISION_PARAMS.read_text(encoding="utf-8")).get("params", {}))
    return p


# ── 순수 함수: raw 탐지 결과 + 파라미터 → 점수 (최적화에서 재사용) ──────────
def score_view(raw, params):
    dc = params["det_conf"]
    boxes = [b for b in raw["boxes"] if b["conf"] >= dc]
    glasses = sorted([b for b in boxes if b["cls"] in C.GLASSES_CLASSES], key=lambda b: -b["conf"])
    lens = [b for b in raw["boxes"] if b["cls"] == C.CLS_LENS and b["conf"] >= params["lens_conf"]]
    L = max((b["conf"] for b in lens), default=0.0)

    # 겹치는 안경 박스 묶기
    clusters = []
    for g in glasses:
        for cl in clusters:
            if iou(cl["lead"]["xyxy"], g["xyxy"]) > 0.5:
                cl["members"].append(g)
                break
        else:
            clusters.append({"lead": g, "members": [g]})

    out = {"lens_conf": L, "lens_found": L > 0, "p_det": None, "p_cls": None,
           "glasses_box": None, "status": "", "score": None}
    if clusters:
        cl = clusters[0]   # 가장 신뢰도 높은 안경 영역
        s_ai = max((m["conf"] for m in cl["members"] if m["cls"] == C.CLS_AI), default=0.0)
        s_nm = max((m["conf"] for m in cl["members"] if m["cls"] == C.CLS_NORMAL), default=0.0)
        p_det = 0.5 + 0.5 * (s_ai - s_nm)
        p_cls = cl["lead"].get("p_cls")
        w = params["w_cls"] if p_cls is not None else 0.0
        p = w * (p_cls if p_cls is not None else 0.0) + (1 - w) * p_det
        out.update(p_det=p_det, p_cls=p_cls, glasses_box=cl["lead"]["xyxy"], status="안경 탐지")
    elif L > 0:
        p = 0.5   # 안경 테두리는 못 찾았지만 렌즈가 보임 → 렌즈 증거로만 판단
        out["status"] = "렌즈만 탐지"
    elif any(b["cls"] == C.CLS_NONE for b in boxes):
        out.update(score=0.0, status="안경 미착용")
        return out
    else:
        out["status"] = "안경 탐지 실패 - 재촬영 필요"
        return out

    if L > 0:
        p = p + (1 - p) * params["lens_weight"] * L
    out["score"] = float(np.clip(p, 0.0, 1.0))
    return out


def fuse(front, side, params):
    """front/side: score_view 결과. 반환: (통합 점수 or None, 설명)"""
    f, s = front["score"], side["score"]
    if f is None and s is None:
        return None, "정면·측면 모두 판정 불가"
    if f is None or s is None:
        v = s if f is None else f
        return v, f"{'측면' if f is None else '정면'} 사진만으로 판정 (다른 쪽 재촬영 권장)"
    m, wf = params["fusion"], params["w_front"]
    if m == "mean":
        return (f + s) / 2, "평균"
    if m == "max":
        return max(f, s), "최댓값"
    if m == "weighted":
        return wf * f + (1 - wf) * s, f"가중평균 (정면 {wf:.2f})"
    # cond
    if front["lens_found"] or side["lens_found"]:
        return max(f, s), "조건 기반: 렌즈 확인 → 최댓값"
    return wf * f + (1 - wf) * s, f"조건 기반: 렌즈 미확인 → 가중평균 (정면 {wf:.2f})"


class AIGlassesSystem:
    def __init__(self, det_weights=C.DET_WEIGHTS, cls_weights=C.CLS_WEIGHTS, params=None):
        from ultralytics import YOLO
        self.det = YOLO(str(det_weights))
        self.cls = YOLO(str(cls_weights)) if Path(cls_weights).exists() else None
        self.det_imgsz, self.cls_imgsz = 640, 224
        for task in ("det", "cls"):
            hp = C.MODELS_DIR / f"{task}_best_hparams.json"
            if hp.exists():
                sz = json.loads(hp.read_text(encoding="utf-8"))["hparams"].get("imgsz")
                if sz:
                    setattr(self, f"{task}_imgsz", sz)
        if self.cls is not None:
            names = {v: k for k, v in self.cls.names.items()}
            self.cls_ai_idx = names["AIGlasses"]
        self.params = params or load_params()

    # 모델 추론 (파라미터와 무관한 raw 결과)
    def raw_view(self, img):
        im = to_pil(img)
        r = self.det.predict(im, conf=RAW_CONF, imgsz=self.det_imgsz, verbose=False)[0]
        boxes = [{"cls": int(c), "conf": float(p), "xyxy": [float(v) for v in xy]}
                 for c, p, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        glasses = sorted([b for b in boxes if b["cls"] in C.GLASSES_CLASSES], key=lambda b: -b["conf"])[:MAX_CROPS]
        if self.cls is not None and glasses:
            W, H = im.size
            crops = []
            for b in glasses:
                x1, y1, x2, y2 = b["xyxy"]
                px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
                crops.append(im.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))))
            for b, cr in zip(glasses, self.cls.predict(crops, imgsz=self.cls_imgsz, verbose=False)):
                b["p_cls"] = float(cr.probs.data[self.cls_ai_idx])
        return {"boxes": boxes, "size": im.size}

    def analyze_view(self, img):
        im = to_pil(img)
        raw = self.raw_view(im)
        res = score_view(raw, self.params)
        res["image"] = draw(im, raw, self.params)
        res["boxes"] = [b for b in raw["boxes"] if b["conf"] >= self.params["det_conf"]]
        return res

    def analyze(self, front_img, side_img):
        front, side = self.analyze_view(front_img), self.analyze_view(side_img)
        score, how = fuse(front, side, self.params)
        return {"front": front, "side": side, "score": score, "fusion": how, "verdict": C.verdict(score)}


COLORS = {C.CLS_AI: (230, 57, 70), C.CLS_LENS: (255, 165, 0), C.CLS_NORMAL: (46, 160, 67), C.CLS_NONE: (120, 120, 120)}


def draw(im, raw, params):
    im = im.copy()
    d = ImageDraw.Draw(im)
    lw = max(2, im.size[0] // 300)
    try:
        font = ImageFont.load_default(size=max(14, im.size[0] // 45))
    except TypeError:
        font = ImageFont.load_default()
    for b in raw["boxes"]:
        thr = params["lens_conf"] if b["cls"] == C.CLS_LENS else params["det_conf"]
        if b["conf"] < thr:
            continue
        col = COLORS[b["cls"]]
        x1, y1, x2, y2 = b["xyxy"]
        d.rectangle((x1, y1, x2, y2), outline=col, width=lw)
        txt = f"{C.DET_NAMES[b['cls']]} {b['conf']:.2f}"
        if b.get("p_cls") is not None:
            txt += f" | cls AI {b['p_cls']:.2f}"
        tb = d.textbbox((x1, y1), txt, font=font)
        ty = y1 - (tb[3] - tb[1]) - 4 if y1 > (tb[3] - tb[1]) + 4 else y2 + 2
        d.rectangle((x1, ty, x1 + tb[2] - tb[0] + 6, ty + tb[3] - tb[1] + 4), fill=col)
        d.text((x1 + 3, ty), txt, fill=(255, 255, 255), font=font)
    return im
