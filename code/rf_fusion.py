"""
시나리오 D (YOLO11m 탐지 + Random Forest 분류) 와 시나리오 A (YOLO 단독) 를 만들고,
정면·측면 통합 방식 4가지(평균 / 최댓값 / 가중평균 / 조건 기반)를 비교해 최적 조합을 고른다.

흐름
 1) Random Forest 학습: train 의 정답 박스(AIGlasses=1, NormalGlasses=0)를 잘라 HOG + 색상 + 형태 특징 추출
 2) 이미지 1장 점수: YOLO 로 안경·렌즈 탐지 → 가장 확실한 안경 박스를 잘라
      - A: YOLO 의 AI/일반 신뢰도 비율
      - D: Random Forest 의 AI 확률
    렌즈 점수 = 그 안경 근처에서 탐지된 AIGlasses_Lens 의 최대 신뢰도
 3) 정면·측면 통합 후보와 임계값을 val 의 '유사 쌍'(같은 클래스의 정면형·측면형 사진 조합)으로 조정
 4) 고른 설정을 test 의 실제 정면·측면 쌍(test_pairs.csv)으로 최종 확인

사용법
  python rf_fusion.py --data <정리된 데이터셋 폴더> --weights <best.pt> --out <결과 폴더>
"""
import argparse
import csv
import itertools
import json
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog as sk_hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from ultralytics import YOLO

AI, LENS, NORMAL = 0, 1, 2
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------- 공통
def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)  # 한글 경로 대응
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_gt(label_path, W, H):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, x, y, w, h = int(p[0]), *map(float, p[1:])
            boxes.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
    return boxes


def crop(img, xyxy, pad=0.08):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    x2, y2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return img[y1:y2, x1:x2]


def features(c):
    """안경 crop → HOG(형태) + HSV 히스토그램(색) + 통계/형태 특징"""
    h, w = c.shape[:2]
    g = cv2.cvtColor(cv2.resize(c, (128, 64)), cv2.COLOR_BGR2GRAY)
    hog = sk_hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    hsv = cv2.cvtColor(cv2.resize(c, (128, 64)), cv2.COLOR_BGR2HSV)
    hist = np.concatenate([
        cv2.calcHist([hsv], [0], None, [16], [0, 180]).ravel(),
        cv2.calcHist([hsv], [1], None, [8], [0, 256]).ravel(),
        cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel(),
    ])
    hist = hist / (hist.sum() + 1e-6)
    edges = cv2.Canny(g, 60, 150)
    stats = np.array([
        w / max(h, 1),                     # 가로/세로 비 (정면·측면 차이)
        edges.mean() / 255.0,              # 엣지 밀도
        g.mean() / 255.0, g.std() / 255.0,
        (g < 50).mean(),                   # 어두운 픽셀 비율 (두꺼운 검은 테)
    ])
    return np.concatenate([hog, hist, stats]).astype(np.float32)


def jitter(img, xyxy, rng):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    s = rng.uniform(0.9, 1.15)
    cx, cy = (x1 + x2) / 2 + rng.uniform(-0.05, 0.05) * w, (y1 + y2) / 2 + rng.uniform(-0.05, 0.05) * h
    c = crop(img, (cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2), pad=0.0)
    if c is not None and rng.random() < 0.5:
        c = cv2.flip(c, 1)
    return c


def list_images(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


# ---------------------------------------------------------------- 1) Random Forest
def build_rf_set(data, split, augment, seed=0):
    rng = random.Random(seed)
    X, y = [], []
    for ip in list_images(data / "images" / split):
        img = imread(ip)
        H, W = img.shape[:2]
        for c, *xyxy in read_gt(data / "labels" / split / (ip.stem + ".txt"), W, H):
            if c not in (AI, NORMAL):
                continue
            crops = [crop(img, xyxy)] + [jitter(img, xyxy, rng) for _ in range(augment)]
            for cr in crops:
                if cr is not None:
                    X.append(features(cr))
                    y.append(1 if c == AI else 0)
    return np.array(X), np.array(y)


def train_rf(data, out):
    Xtr, ytr = build_rf_set(data, "train", augment=5)
    Xva, yva = build_rf_set(data, "val", augment=0)
    best, best_score = None, -1
    log = []
    for depth, feat, leaf in itertools.product([None, 20], ["sqrt", 0.05], [1, 2]):
        rf = RandomForestClassifier(n_estimators=500, max_depth=depth, max_features=feat, min_samples_leaf=leaf,
                                    class_weight="balanced", n_jobs=-1, random_state=42)
        rf.fit(Xtr, ytr)
        p = rf.predict(Xva)
        score = balanced_accuracy_score(yva, p)
        log.append({"max_depth": depth, "max_features": feat, "min_samples_leaf": leaf, "val_bal_acc": round(score, 4)})
        if score > best_score:
            best, best_score = rf, score
    joblib.dump(best, out / "rf_glasses.joblib")
    p = best.predict(Xva)
    crop_metrics = {
        "train_crops": int(len(ytr)), "val_crops": int(len(yva)),
        "val_accuracy": round(accuracy_score(yva, p), 4), "val_balanced_accuracy": round(best_score, 4),
        "val_f1_AI": round(f1_score(yva, p), 4),
        "val_confusion[[N->N,N->AI],[AI->N,AI->AI]]": confusion_matrix(yva, p).tolist(),
        "grid": log,
    }
    return best, crop_metrics


# ---------------------------------------------------------------- 2) 이미지 1장 분석
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


def analyze(model, rf, ip, det_conf=0.25, img=None):
    """ip: 이미지 경로 (img 를 직접 주면 경로는 이름 표시에만 사용)"""
    if img is None:
        img = imread(ip)
    name = Path(str(ip)).stem
    r = model.predict(img, conf=0.05, iou=0.5, verbose=False)[0]
    dets = [(int(c), float(s), b.tolist()) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    glasses = [d for d in dets if d[0] in (AI, NORMAL)]
    if not glasses or max(d[1] for d in glasses) < det_conf:
        return {"image": name, "detected": False, "dets": [d for d in dets if d[1] >= det_conf]}
    main = max(glasses, key=lambda d: d[1])
    mb = main[2]
    near = [d for d in glasses if iou(d[2], mb) > 0.5]
    ai_c = max([d[1] for d in near if d[0] == AI], default=0.0)
    n_c = max([d[1] for d in near if d[0] == NORMAL], default=0.0)
    yolo_p = ai_c / (ai_c + n_c + 1e-6)
    w, h = mb[2] - mb[0], mb[3] - mb[1]
    ex = (mb[0] - 0.15 * w, mb[1] - 0.3 * h, mb[2] + 0.15 * w, mb[3] + 0.3 * h)
    lens = max([d[1] for d in dets if d[0] == LENS and ex[0] <= (d[2][0] + d[2][2]) / 2 <= ex[2]
                and ex[1] <= (d[2][1] + d[2][3]) / 2 <= ex[3]], default=0.0)
    rf_p = float(rf.predict_proba(features(crop(img, mb))[None])[0, 1])
    return {"image": name, "detected": True, "glasses_conf": main[1], "yolo_p": yolo_p, "rf_p": rf_p, "lens": lens,
            "main_box": mb, "dets": [d for d in dets if d[1] >= det_conf]}


# ---------------------------------------------------------------- 3) 정면·측면 통합
def fuse(method, f, s, scen, p):
    key = "yolo_p" if scen == "A" else "rf_p"
    a = f[key] if f["detected"] else None
    b = s[key] if s["detected"] else None
    if a is None and b is None:
        return None
    if a is None or b is None:  # 한쪽만 탐지되면 탐지된 쪽 점수 사용
        one = a if a is not None else b
        lens = (f.get("lens", 0) if f["detected"] else 0, s.get("lens", 0) if s["detected"] else 0)
        if method == "condition" and max(lens) >= p["lens_t"]:
            return 1.0
        return one
    if method == "mean":
        return (a + b) / 2
    if method == "max":
        return max(a, b)
    if method == "weighted":
        return p["w"] * a + (1 - p["w"]) * b
    if method == "condition":  # 렌즈가 확실히 보이면 AI 의심, 아니면 평균
        if max(f["lens"], s["lens"]) >= p["lens_t"]:
            return 1.0
        return (a + b) / 2
    raise ValueError(method)


def decide3(f, s, key="yolo_p", lens_t=0.55, ai_t=0.7):
    """3단계 조건 기반 최종 판정 (교차검증으로 선택된 방식)
    - 정면/측면 중 하나라도 카메라 렌즈 신뢰도 >= lens_t  → 'AI'     (AI 안경 의심)
    - 렌즈는 없지만 두 사진 평균 AI 점수 >= ai_t          → 'check'  (확인 필요: 모양은 AI 안경과 비슷)
    - 그 외                                               → 'Normal' (일반 안경 추정)
    - 두 사진 모두 안경 탐지 실패                          → None     (재촬영 필요)
    """
    dets = [x for x in (f, s) if x["detected"]]
    if not dets:
        return None, None
    lens = max(x["lens"] for x in dets)
    sc = sum(x[key] for x in dets) / len(dets)
    if lens >= lens_t:
        return "AI", sc
    if sc >= ai_t:
        return "check", sc
    return "Normal", sc


PARAM_GRID = {
    "mean": [{}],
    "max": [{}],
    "weighted": [{"w": w} for w in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)],
    "condition": [{"lens_t": t} for t in (0.25, 0.35, 0.45, 0.55, 0.65)],
}
THRESHOLDS = [round(t, 2) for t in np.arange(0.2, 0.81, 0.05)]


def evaluate(pairs, res, method, scen, p, t):
    y, yhat, abstain = [], [], 0
    for fr, sd, label in pairs:
        sc = fuse(method, res[fr], res[sd], scen, p)
        if sc is None:
            abstain += 1
            sc = -1  # 판정 불가 → AI 를 놓친 것으로 계산 (보수적)
        y.append(1 if label == "AI" else 0)
        yhat.append(1 if sc >= t else 0)
    return {
        "n": len(y), "abstain": abstain,
        "bal_acc": balanced_accuracy_score(y, yhat), "acc": accuracy_score(y, yhat),
        "f1_AI": f1_score(y, yhat, zero_division=0), "recall_AI": recall_score(y, yhat, zero_division=0),
        "precision_AI": precision_score(y, yhat, zero_division=0),
        "cm": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def val_pseudo_pairs(data, res, seed=0, cap=400):
    """val 에서 같은 클래스끼리 '정면형'(가로로 넓은 박스)과 '측면형'(좁은 박스)을 짝지어 유사 쌍 생성"""
    info = {"AI": [], "Normal": []}
    for ip in list_images(data / "images" / "val"):
        img = imread(ip)
        H, W = img.shape[:2]
        gt = [g for g in read_gt(data / "labels" / "val" / (ip.stem + ".txt"), W, H) if g[0] in (AI, NORMAL)]
        if not gt:
            continue
        c, x1, y1, x2, y2 = max(gt, key=lambda g: (g[3] - g[1]) * (g[4] - g[2]))
        info["AI" if c == AI else "Normal"].append((ip.stem, (x2 - x1) / max(y2 - y1, 1)))
    rng = random.Random(seed)
    pairs = []
    for label, lst in info.items():
        lst.sort(key=lambda t: -t[1])
        half = len(lst) // 2
        fronts, sides = lst[:half] or lst, lst[half:] or lst
        combos = [(f[0], s[0], label) for f in fronts for s in sides if f[0] != s[0]]
        rng.shuffle(combos)
        pairs += combos[:cap]
    return pairs


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pairs", default=str(Path(__file__).with_name("test_pairs.csv")))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1] Random Forest 학습")
    rf, crop_metrics = train_rf(data, out)
    print("    crop 단위 val:", {k: v for k, v in crop_metrics.items() if k != "grid"})

    print("[2] YOLO 탐지 + 이미지별 점수")
    model = YOLO(args.weights)
    res = {}
    for split in ("val", "test"):
        for ip in list_images(data / "images" / split):
            res[ip.stem if split == "test" else ip.stem] = analyze(model, rf, ip)
    with open(out / "per_image_scores.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    vpairs = val_pseudo_pairs(data, res)
    tpairs = [(r["front"], r["side"], r["label"]) for r in csv.DictReader(open(args.pairs, encoding="utf-8"))]
    print(f"    val 유사 쌍 {len(vpairs)}개, test 실제 쌍 {len(tpairs)}개")

    print("[3] 통합 방식 비교 (val 에서 파라미터·임계값 조정 → test 로 확인)")
    rows = []
    for scen in ("A", "D"):
        for method, grid in PARAM_GRID.items():
            best = None
            for p in grid:
                for t in THRESHOLDS:
                    m = evaluate(vpairs, res, method, scen, p, t)
                    key = (m["bal_acc"], m["recall_AI"], -abs(t - 0.5))
                    if best is None or key > best[0]:
                        best = (key, p, t, m)
            _, p, t, vm = best
            tm = evaluate(tpairs, res, method, scen, p, t)
            rows.append({"scenario": scen, "method": method, "params": p, "threshold": t, "val": vm, "test": tm})
            print(f"    {scen}-{method:9s} {str(p):16s} t={t:.2f} | val balAcc={vm['bal_acc']:.3f} "
                  f"recallAI={vm['recall_AI']:.3f} | test acc={tm['acc']:.3f} cm={tm['cm']}")

    best = max(rows, key=lambda r: (r["val"]["bal_acc"], r["val"]["recall_AI"], r["test"]["acc"]))
    summary = {"rf_crop_metrics": crop_metrics, "comparison": rows, "best": best}
    with open(out / "fusion_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    with open(out / "best_config.json", "w", encoding="utf-8") as f:
        json.dump({"scenario": best["scenario"], "method": best["method"], "params": best["params"],
                   "threshold": best["threshold"], "yolo_det_conf": 0.25}, f, ensure_ascii=False, indent=1)
    print("[최적]", best["scenario"], best["method"], best["params"], "t=", best["threshold"])


if __name__ == "__main__":
    main()
