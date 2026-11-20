"""
교차검증으로 '처음 보는 사진'에서의 점수를 만들어 RF·정면측면 통합 방식을 공정하게 고르는 파이프라인

1) train+val 을 비슷한 사진 그룹 단위로 5-fold 분할 (같은 제품 사진이 학습/평가에 동시에 들어가지 않게)
2) fold 마다 YOLO11m 학습 → 빠진 fold 이미지에서 탐지 점수(YOLO AI 확률, 렌즈 점수)와 안경 crop 특징 추출
3) Random Forest(2단계 분류기): [crop 외형 특징 + YOLO 점수 + 렌즈 점수 + 탐지 신뢰도] → AI 확률
   (RF 도 그룹 단위 교차검증으로 out-of-fold 확률 계산)
4) 정면형·측면형 유사 쌍으로 통합 방식(평균/최댓값/가중평균/조건 기반) × 시나리오(A: YOLO 단독, D: YOLO+RF) 비교
5) 전체 데이터로 최종 YOLO·RF 학습 → test 의 실제 정면·측면 쌍으로 확인, models 폴더에 저장

사용법
  python cv_pipeline.py --data <정리된 데이터셋> --work <작업 폴더> --models <모델 저장 폴더>
"""
import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from ultralytics import YOLO

from rf_fusion import (AI, LENS, NORMAL, PARAM_GRID, THRESHOLDS, crop, evaluate, features, imread, iou,
                       jitter, list_images, read_gt)

TRAIN_ARGS = dict(imgsz=640, batch=16, device=0, workers=4, patience=25, cos_lr=True, close_mosaic=10,
                  cache="ram", amp=True, seed=42, deterministic=True, plots=False, verbose=False)


# ---------------------------------------------------------------- 데이터 묶기
def dhash(path):
    im = Image.open(path)
    out = []
    for x in (im, ImageOps.mirror(im)):
        a = np.asarray(x.convert("L").resize((17, 16), Image.LANCZOS), dtype=np.int16)
        out.append((a[:, 1:] > a[:, :-1]).flatten())
    return out


def collect(data):
    items = []
    for split in ("train", "val"):
        for ip in list_images(data / "images" / split):
            lp = data / "labels" / split / (ip.stem + ".txt")
            W, H = Image.open(ip).size
            gt = read_gt(lp, W, H)
            g = [b for b in gt if b[0] in (AI, NORMAL)]
            kind = "AI" if any(b[0] in (AI, LENS) for b in gt) else ("Normal" if g else "None")
            main = max(g, key=lambda b: (b[3] - b[1]) * (b[4] - b[2])) if g else None
            aspect = (main[3] - main[1]) / max(main[4] - main[2], 1) if main else None
            items.append({"id": f"{split}/{ip.stem}", "img": ip, "lab": lp, "kind": kind, "aspect": aspect})
    # 비슷한 사진 그룹
    hs = [dhash(i["img"]) for i in items]
    H0 = np.array([h[0] for h in hs]); H1 = np.array([h[1] for h in hs])
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(len(items)):
        d = np.minimum((H0[i] != H0).sum(1), (H0[i] != H1).sum(1))
        for j in np.where(d <= 30)[0]:
            if j > i:
                parent[find(j)] = find(i)
    for i, it in enumerate(items):
        it["group"] = find(i)
    return items


def make_fold_dataset(root, train_items, val_items, names):
    if root.exists():
        shutil.rmtree(root)
    for split, its in (("train", train_items), ("val", val_items)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for it in its:
            stem = it["id"].replace("/", "__")
            shutil.copy2(it["img"], root / "images" / split / (stem + it["img"].suffix))
            shutil.copy2(it["lab"], root / "labels" / split / (stem + ".txt"))
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")
    return root / "data.yaml"


# ---------------------------------------------------------------- 탐지 → 원시 점수
def raw_scores(model, ip, det_conf=0.25, img=None):
    """이미지 1장 → 탐지 결과와 점수. img 를 주면 경로 대신 그 배열을 사용 (웹 GUI 용)"""
    if img is None:
        img = imread(ip)
    r = model.predict(img, conf=0.05, iou=0.5, verbose=False)[0]
    dets = [(int(c), float(s), b.tolist()) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    glasses = [d for d in dets if d[0] in (AI, NORMAL)]
    if not glasses or max(d[1] for d in glasses) < det_conf:
        return {"detected": False, "dets": [d for d in dets if d[1] >= det_conf]}
    main = max(glasses, key=lambda d: d[1])
    mb = main[2]
    near = [d for d in glasses if iou(d[2], mb) > 0.5]
    ai_c = max([d[1] for d in near if d[0] == AI], default=0.0)
    n_c = max([d[1] for d in near if d[0] == NORMAL], default=0.0)
    w, h = mb[2] - mb[0], mb[3] - mb[1]
    ex = (mb[0] - 0.15 * w, mb[1] - 0.3 * h, mb[2] + 0.15 * w, mb[3] + 0.3 * h)
    lens = max([d[1] for d in dets if d[0] == LENS and ex[0] <= (d[2][0] + d[2][2]) / 2 <= ex[2]
                and ex[1] <= (d[2][1] + d[2][3]) / 2 <= ex[3]], default=0.0)
    return {"detected": True, "yolo_p": ai_c / (ai_c + n_c + 1e-6), "ai_c": ai_c, "n_c": n_c, "lens": lens,
            "glasses_conf": main[1], "box_aspect": w / max(h, 1),
            "app": features(crop(img, mb)).tolist(),
            "main_box": mb, "dets": [d for d in dets if d[1] >= det_conf]}


def meta_vector(r):
    """RF 입력: crop 외형 특징 + YOLO 점수들"""
    return np.concatenate([np.array(r["app"], dtype=np.float32),
                           np.array([r["yolo_p"], r["ai_c"], r["n_c"], r["lens"], r["glasses_conf"], r["box_aspect"]],
                                    dtype=np.float32)])


def new_rf():
    return RandomForestClassifier(n_estimators=600, max_features="sqrt", min_samples_leaf=2,
                                  class_weight="balanced", n_jobs=-1, random_state=42)


# ---------------------------------------------------------------- 쌍 만들기
def pseudo_pairs(records, seed=0, cap=600):
    rng = random.Random(seed)
    pairs = []
    for label in ("AI", "Normal"):
        lst = sorted([(k, r["gt_aspect"]) for k, r in records.items() if r["kind"] == label], key=lambda t: -t[1])
        half = len(lst) // 2
        fronts, sides = lst[:half], lst[half:]
        combos = [(f[0], s[0], label) for f in fronts for s in sides]
        rng.shuffle(combos)
        pairs += combos[:cap]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--pairs", default=str(Path(__file__).with_name("test_pairs.csv")))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    data, work, models = Path(args.data), Path(args.work), Path(args.models)
    work.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    names = ["AIGlasses", "AIGlasses_Lens", "NormalGlasses"]

    items = collect(data)
    items = [i for i in items if i["kind"] != "None"] + [i for i in items if i["kind"] == "None"]
    y = [i["kind"] for i in items]
    groups = [i["group"] for i in items]
    print(f"[데이터] {len(items)}장, 그룹 {len(set(groups))}개, AI {y.count('AI')} / Normal {y.count('Normal')} / 배경 {y.count('None')}")

    # 1~2) fold 별 YOLO 학습 + out-of-fold 점수
    oof_path = work / "oof_records.json"
    records = json.loads(oof_path.read_text(encoding="utf-8")) if oof_path.exists() else {}
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
    for k, (tr, va) in enumerate(sgkf.split(items, y, groups)):
        va_ids = [items[i]["id"] for i in va]
        if all(v in records for v in va_ids):
            print(f"[fold {k}] 이미 계산됨, 건너뜀")
            continue
        yaml = make_fold_dataset(work / f"fold{k}" / "data", [items[i] for i in tr], [items[i] for i in va], names)
        print(f"[fold {k}] YOLO 학습: train {len(tr)} / held-out {len(va)}")
        model = YOLO("yolo11m.pt")
        model.train(data=str(yaml), epochs=args.epochs, project=str(work / f"fold{k}"), name="yolo", exist_ok=True,
                    **TRAIN_ARGS)
        best = YOLO(str(work / f"fold{k}" / "yolo" / "weights" / "best.pt"))
        for i in va:
            it = items[i]
            r = raw_scores(best, it["img"])
            r.pop("dets", None); r.pop("main_box", None)
            r.update({"kind": it["kind"], "group": int(it["group"]), "gt_aspect": it["aspect"], "fold": k})
            records[it["id"]] = r
        oof_path.write_text(json.dumps(records), encoding="utf-8")
        shutil.rmtree(work / f"fold{k}" / "data", ignore_errors=True)

    # 3) RF (2단계) out-of-fold 확률
    keys = [k for k, r in records.items() if r["kind"] in ("AI", "Normal") and r["detected"]]
    X = np.array([meta_vector(records[k]) for k in keys])
    yy = np.array([1 if records[k]["kind"] == "AI" else 0 for k in keys])
    gg = np.array([records[k]["group"] for k in keys])
    folds = np.array([records[k]["fold"] for k in keys])
    oof_rf = np.zeros(len(keys))
    for f in sorted(set(folds)):
        m = new_rf().fit(X[folds != f], yy[folds != f])
        oof_rf[folds == f] = m.predict_proba(X[folds == f])[:, 1]
    for k, p in zip(keys, oof_rf):
        records[k]["rf_p"] = float(p)
    n_det = sum(1 for r in records.values() if r["kind"] in ("AI", "Normal") and r["detected"])
    n_all = sum(1 for r in records.values() if r["kind"] in ("AI", "Normal"))
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    img_metrics = {
        "detected": f"{n_det}/{n_all}",
        "A_yolo_auc": round(roc_auc_score(yy, [records[k]["yolo_p"] for k in keys]), 4),
        "D_rf_auc": round(roc_auc_score(yy, oof_rf), 4),
        "A_yolo_balacc@0.5": round(balanced_accuracy_score(yy, [records[k]["yolo_p"] >= 0.5 for k in keys]), 4),
        "D_rf_balacc@0.5": round(balanced_accuracy_score(yy, oof_rf >= 0.5), 4),
    }
    print("[이미지 1장 단위 (교차검증)]", img_metrics)

    # 4) 정면·측면 통합 비교
    res = {k: dict(r, detected=r["detected"]) for k, r in records.items()}
    for r in res.values():
        if r["detected"] and "rf_p" not in r:
            r["rf_p"] = r["yolo_p"]
    pairs = pseudo_pairs({k: r for k, r in records.items() if r["kind"] in ("AI", "Normal") and r["gt_aspect"]})
    rows = []
    for scen in ("A", "D"):
        for method, grid in PARAM_GRID.items():
            best = None
            for p in grid:
                for t in THRESHOLDS:
                    m = evaluate(pairs, res, method, scen, p, t)
                    key = (round(m["bal_acc"], 4), m["recall_AI"], -abs(t - 0.5))
                    if best is None or key > best[0]:
                        best = (key, p, t, m)
            _, p, t, m = best
            rows.append({"scenario": scen, "method": method, "params": p, "threshold": t, "cv": m})
            print(f"  {scen}-{method:9s} {str(p):16s} t={t:.2f} | CV 쌍 balAcc={m['bal_acc']:.3f} "
                  f"recallAI={m['recall_AI']:.3f} precAI={m['precision_AI']:.3f} cm={m['cm']}")
    chosen = max(rows, key=lambda r: (round(r["cv"]["bal_acc"], 4), r["cv"]["recall_AI"]))
    print("[선택]", chosen["scenario"], chosen["method"], chosen["params"], "t=", chosen["threshold"])

    # 5) 최종 모델: 전체 데이터로 YOLO 학습, RF 는 out-of-fold 기록 전체로 학습
    final_yaml = make_fold_dataset(work / "final" / "data", items, [i for i in items if i["id"].startswith("val/")], names)
    final_w = work / "final" / "yolo" / "weights" / "best.pt"
    if not final_w.exists():
        print("[최종] 전체 데이터로 YOLO 학습")
        YOLO("yolo11m.pt").train(data=str(final_yaml), epochs=args.epochs, project=str(work / "final"), name="yolo",
                                 exist_ok=True, **dict(TRAIN_ARGS, patience=args.epochs))
    final_rf = new_rf().fit(X, yy)
    shutil.copy2(final_w, models / "best.pt")
    joblib.dump(final_rf, models / "rf_glasses.joblib")
    cfg = {"scenario": chosen["scenario"], "method": chosen["method"], "params": chosen["params"],
           "threshold": chosen["threshold"], "yolo_det_conf": 0.25}
    (models / "best_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    # test 실제 쌍 확인 (모든 방식)
    fmodel = YOLO(str(final_w))
    tres = {}
    for ip in list_images(data / "images" / "test"):
        r = raw_scores(fmodel, ip)
        if r["detected"]:
            r["rf_p"] = float(final_rf.predict_proba(meta_vector(r)[None])[0, 1])
        r.pop("app", None)
        tres[ip.stem] = r
    tpairs = [(r["front"], r["side"], r["label"]) for r in csv.DictReader(open(args.pairs, encoding="utf-8"))]
    for row in rows:
        row["test"] = evaluate(tpairs, tres, row["method"], row["scenario"], row["params"], row["threshold"])
    print("[test 실제 쌍]")
    for row in rows:
        mark = " ← 선택" if row is chosen else ""
        print(f"  {row['scenario']}-{row['method']:9s} acc={row['test']['acc']:.3f} cm={row['test']['cm']}{mark}")
    for k, r in tres.items():
        if r["detected"]:
            print(f"   {k}: yolo_p={r['yolo_p']:.2f} rf_p={r['rf_p']:.2f} lens={r['lens']:.2f}")
        else:
            print(f"   {k}: 탐지 실패")
    (work / "cv_results.json").write_text(json.dumps({"image_level": img_metrics, "comparison": rows,
                                                       "chosen": cfg, "test_scores": tres}, ensure_ascii=False,
                                                      indent=1, default=float), encoding="utf-8")
    print("저장:", models)


if __name__ == "__main__":
    main()
