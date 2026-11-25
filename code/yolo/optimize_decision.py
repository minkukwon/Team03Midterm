"""통합 판단 파라미터 최적화 (5-fold out-of-fold 예측 기준)

train+val 풀의 모든 이미지를, 그 이미지를 학습하지 않은 fold 모델(models/folds/*_fold{k}.pt)로
예측해 캐시한다(out-of-fold). 이 예측으로 아래 파라미터를 격자 탐색한다.
  det_conf, w_cls, lens_weight, lens_conf, fusion
동일 안경의 정면·측면 쌍이 없으므로, 같은 정답 클래스의 이미지 2장을 무작위로 묶은
'모의 쌍'으로 통합 판단을 평가한다.

목적함수 (최종 판정 임계값 0.8 / 0.5 를 그대로 반영)
  objective = 0.5 * F1(AI, 점수>0.5 를 양성으로: AI 안경 + 의심)
            + 0.5 * 확정 정확도(AI 쌍은 >=0.8, 일반 쌍은 <=0.5 로 확정 판정된 비율)
"""
import itertools
import json
import random

import numpy as np

import config as C
from pipeline import AIGlassesSystem, fuse, score_view

GRID = {
    "det_conf": [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    "w_cls": [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    "lens_weight": [0.0, 0.25, 0.5, 0.75, 1.0],
    "lens_conf": [0.2, 0.3, 0.4, 0.5],
    "fusion": ["mean", "max", "cond"],
}
N_PAIRS = 300  # 클래스별 모의 쌍 수


def image_label(img_path):
    """1 = AI 안경, 0 = AI 안경 아님(일반 안경 / 안경 미착용), None = 라벨 없음"""
    lp = img_path.parent.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")
    if not lp.exists():
        return None
    cls = {int(l.split()[0]) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()}
    if cls & {C.CLS_AI, C.CLS_LENS}:
        return 1
    if cls:
        return 0
    return None


def collect(system, split):
    from prepare_data import images
    items = []
    for p in images(split):
        y = image_label(p)
        if y is not None:
            items.append({"path": p, "y": y, "raw": system.raw_view(p)})
    return items


def collect_oof():
    """fold k 모델로 fold k 이미지를 예측 (학습에 쓰이지 않은 데이터만 평가)."""
    from pathlib import Path
    folds = json.loads((C.FOLDS_DIR / "folds.json").read_text(encoding="utf-8"))
    items = []
    for k in range(C.N_FOLDS):
        system = AIGlassesSystem(C.FOLD_MODELS_DIR / f"det_fold{k}.pt", C.FOLD_MODELS_DIR / f"cls_fold{k}.pt",
                                 params=dict(C.DEFAULT_PARAMS))
        for r in folds:
            if r["fold"] != k:
                continue
            p = Path(r["path"])
            y = image_label(p)
            if y is not None:
                items.append({"path": p, "y": y, "fold": k, "raw": system.raw_view(p)})
        print(f"   fold{k} out-of-fold 예측 완료")
    return items


def make_pairs(items, n, seed=0):
    rng = random.Random(seed)
    pairs = []
    for y in (0, 1):
        idx = [i for i, it in enumerate(items) if it["y"] == y]
        if len(idx) < 2:
            continue
        for _ in range(n):
            a, b = rng.sample(idx, 2)
            pairs.append((a, b, y))
    return pairs


def metrics_from_scores(scores, ys):
    s = np.array([-1.0 if v is None else v for v in scores])
    y = np.array(ys)
    pred = s > C.NORMAL_THRESHOLD
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    decisive = float(np.mean(np.where(y == 1, s >= C.AI_THRESHOLD, (s >= 0) & (s <= C.NORMAL_THRESHOLD))))
    verdicts = [C.verdict(v) for v in scores]
    ternary = {}
    for name, cls in (("AI", 1), ("Normal", 0)):
        vs = [v for v, t in zip(verdicts, ys) if t == cls]
        ternary[name] = {k: vs.count(k) for k in (C.LABEL_AI, C.LABEL_SUSPECT, C.LABEL_NORMAL, C.LABEL_RETAKE)}
    return {
        "objective": 0.5 * f1 + 0.5 * decisive,
        "f1": f1, "precision": prec, "recall": rec, "decisive_acc": decisive,
        "ai_miss_rate": fn / (tp + fn) if tp + fn else 0.0,        # AI 안경 미탐률
        "normal_false_alarm": fp / (fp + tn) if fp + tn else 0.0,  # 일반 안경 오탐률
        "confusion(>0.5)": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "ternary": ternary,
    }


def evaluate_params(items, pairs, params):
    views = [score_view(it["raw"], params) for it in items]
    single = metrics_from_scores([v["score"] for v in views], [it["y"] for it in items])
    fused = [fuse(views[a], views[b], params)[0] for a, b, _ in pairs]
    paired = metrics_from_scores(fused, [y for *_, y in pairs])
    return single, paired


def main():
    items = collect_oof()
    pairs = make_pairs(items, N_PAIRS)
    print(f"OOF 이미지 {len(items)}장 (AI {sum(i['y'] for i in items)}), 모의 쌍 {len(pairs)}개")

    _, base = evaluate_params(items, pairs, C.DEFAULT_PARAMS)
    best = (base["objective"], dict(C.DEFAULT_PARAMS))
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        params = {**C.DEFAULT_PARAMS, **dict(zip(keys, combo))}
        _, paired = evaluate_params(items, pairs, params)
        if paired["objective"] > best[0] + 1e-9:
            best = (paired["objective"], params)

    single, paired = evaluate_params(items, pairs, best[1])
    print(f"기본 파라미터 objective={base['objective']:.4f}")
    print(f"최적 파라미터 objective={best[0]:.4f}: {best[1]}")
    print("단일 이미지:", {k: round(v, 4) for k, v in single.items() if isinstance(v, float)})
    print("모의 쌍   :", {k: round(v, 4) for k, v in paired.items() if isinstance(v, float)})
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.DECISION_PARAMS.write_text(json.dumps({
        "params": best[1], "objective": "0.5*F1(>0.5) + 0.5*decisive_acc on 5-fold out-of-fold simulated pairs",
        "baseline_oof_pair": base, "oof_single": single, "oof_pair": paired,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {C.DECISION_PARAMS}")


if __name__ == "__main__":
    main()
