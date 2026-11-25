"""YOLO11m 탐지 + YOLO11m-cls 분류 학습 — 5-fold 교차검증 기반 하이퍼파라미터 탐색

절차 (탐지/분류 공통)
  1) 하이퍼파라미터 후보마다 5개 fold 모두 학습 → fold별 val 지표의 평균±표준편차로 비교
       - 탐지 fitness = 0.1*mAP50 + 0.9*mAP50-95 (Ultralytics 기본)
       - 분류 fitness = top-1 정확도
     · 첫 fold 점수가 현재 최고 평균의 PRUNE_RATIO 미만이면 나머지 fold 생략 (시간 절약)
  2) 최고 후보의 fold별 best.pt → models/folds/ (optimize_decision.py 의 out-of-fold 예측용)
  3) 최고 후보로 train+val 전체를 학습. 에폭 = fold별 최적 에폭의 중앙값 × 1.15
     → models/det_best.pt, models/cls_best.pt

사용 예
  py -3.12 train.py                                  # 탐지 → 분류 전체
  py -3.12 train.py --stage det --trials 4 --search-epochs 30
  py -3.12 train.py --stage cls --skip-search        # 저장된 최적값으로 최종 학습만
"""
import argparse
import csv
import json
import math
import random
import shutil
import statistics

from ultralytics import YOLO

import config as C

PRUNE_RATIO = 0.85

# ── 탐색 공간 (옵티마이저별 lr 범위 분리) ─────────────
LR_RANGE = {"SGD": (0.003, 0.02), "AdamW": (1e-4, 1e-3)}
DET_SPACE = {
    "optimizer": ["SGD", "AdamW"],
    "imgsz": [640, 960],               # 작은 렌즈 탐지를 위해 고해상도 후보 포함
    "mosaic": ("uni", 0.7, 1.0),
    "scale": ("uni", 0.3, 0.6),
    "degrees": ("uni", 0.0, 8.0),
    "mixup": ("uni", 0.0, 0.15),
    "hsv_s": ("uni", 0.5, 0.8),
}
DET_DEFAULT = {"optimizer": "SGD", "lr0": 0.01, "imgsz": 640, "mosaic": 1.0, "scale": 0.5,
               "degrees": 0.0, "mixup": 0.0, "hsv_s": 0.7}
DET_SEEDS = [DET_DEFAULT, {**DET_DEFAULT, "imgsz": 960}]   # 항상 평가하는 기준 후보

CLS_SPACE = {
    "optimizer": ["SGD", "AdamW"],
    "imgsz": [224, 320],
    "dropout": ("uni", 0.0, 0.3),
    "degrees": ("uni", 0.0, 15.0),
    "hsv_s": ("uni", 0.4, 0.8),
}
CLS_DEFAULT = {"optimizer": "AdamW", "lr0": 0.001, "imgsz": 224, "dropout": 0.0, "degrees": 0.0, "hsv_s": 0.7}
CLS_SEEDS = [CLS_DEFAULT, {**CLS_DEFAULT, "imgsz": 320}]


def sample(space, rng):
    hp = {}
    for k, v in space.items():
        if isinstance(v, list):
            hp[k] = rng.choice(v)
        else:
            hp[k] = round(rng.uniform(v[1], v[2]), 4)
    lo, hi = LR_RANGE[hp["optimizer"]]
    hp["lr0"] = round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 6)
    return hp


def run(task, hp, data, epochs, project, name, patience):
    hp = dict(hp)
    imgsz = hp.pop("imgsz")
    if task == "det":
        model = YOLO("yolo11m.pt")
        batch = 16 if imgsz <= 640 else 8
    else:
        model = YOLO("yolo11m-cls.pt")
        batch = 32
    model.train(data=str(data), imgsz=imgsz, batch=batch, **hp, epochs=epochs, patience=patience,
                project=str(project), name=name, exist_ok=True, workers=4, seed=0, deterministic=False,
                plots=False, verbose=False, amp=True)
    return model.trainer.save_dir


def best_from_results(save_dir, task):
    """results.csv 에서 최고 fitness 와 그 에폭."""
    with open(save_dir / "results.csv", encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    best = (-1.0, 0)
    for r in rows:
        if task == "det":
            fit = 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])
        else:
            fit = float(r["metrics/accuracy_top1"])
        if fit > best[0]:
            best = (fit, int(float(r["epoch"])))
    detail = {}
    if task == "det":
        r = next(x for x in rows if int(float(x["epoch"])) == best[1])
        detail = {"mAP50": float(r["metrics/mAP50(B)"]), "mAP50-95": float(r["metrics/mAP50-95(B)"]),
                  "precision": float(r["metrics/precision(B)"]), "recall": float(r["metrics/recall(B)"])}
    return best[0], best[1], detail


def fold_data(task, k):
    d = C.FOLDS_DIR / f"fold{k}"
    return d / "data.yaml" if task == "det" else d / "cls"


def cross_validate(task, hp, epochs, trial_dir, best_mean):
    fits, epochs_best, details = [], [], []
    for k in range(C.N_FOLDS):
        save_dir = run(task, hp, fold_data(task, k), epochs, trial_dir, f"fold{k}", patience=max(10, epochs // 3))
        fit, ep, det = best_from_results(save_dir, task)
        fits.append(fit), epochs_best.append(ep), details.append(det)
        (save_dir / "weights" / "last.pt").unlink(missing_ok=True)
        print(f"   [{task}] fold{k}: fitness={fit:.4f} (best epoch {ep}) {det}")
        if k == 0 and best_mean > 0 and fit < PRUNE_RATIO * best_mean:
            print(f"   [{task}] 첫 fold가 최고 평균({best_mean:.4f})의 {PRUNE_RATIO:.0%} 미만 → 조기 중단")
            return {"mean": fit, "std": 0.0, "folds": fits, "best_epochs": epochs_best, "pruned": True}
    mean_detail = {m: statistics.mean(d[m] for d in details) for m in details[0]} if details[0] else {}
    return {"mean": statistics.mean(fits), "std": statistics.pstdev(fits), "folds": fits,
            "best_epochs": epochs_best, "pruned": False, "detail": mean_detail}


def search(task, trials, epochs, seed):
    space, seeds = (DET_SPACE, DET_SEEDS) if task == "det" else (CLS_SPACE, CLS_SEEDS)
    rng = random.Random(seed)
    candidates = seeds[:trials] + [sample(space, rng) for _ in range(max(0, trials - len(seeds)))]
    root = C.RUNS_DIR / task
    root.mkdir(parents=True, exist_ok=True)
    rows, best = [], None
    for i, hp in enumerate(candidates):
        print(f"\n===== [{task}] 후보 {i + 1}/{len(candidates)}: {hp}")
        trial_dir = root / f"trial_{i:02d}"
        res = cross_validate(task, hp, epochs, trial_dir, best["res"]["mean"] if best else 0.0)
        print(f"===== [{task}] 후보 {i}: CV fitness = {res['mean']:.4f} ± {res['std']:.4f}"
              f"{' (중단)' if res['pruned'] else ''}")
        rows.append({"trial": i, "cv_mean": round(res["mean"], 5), "cv_std": round(res["std"], 5),
                     "pruned": res["pruned"], "folds": " ".join(f"{f:.4f}" for f in res["folds"]),
                     **{k: round(v, 4) for k, v in res.get("detail", {}).items()}, **hp})
        if not res["pruned"] and (best is None or res["mean"] > best["res"]["mean"]):
            if best:
                shutil.rmtree(best["dir"], ignore_errors=True)   # 이전 최고 후보 가중치 삭제
            best = {"hp": hp, "res": res, "dir": trial_dir, "trial": i}
        else:
            shutil.rmtree(trial_dir, ignore_errors=True)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(root / "cv_search_results.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    C.FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for k in range(C.N_FOLDS):
        shutil.copy2(best["dir"] / f"fold{k}" / "weights" / "best.pt", C.FOLD_MODELS_DIR / f"{task}_fold{k}.pt")
    print(f"\n[{task}] 최적 후보 {best['trial']}: CV {best['res']['mean']:.4f} ± {best['res']['std']:.4f}  {best['hp']}")
    return best["hp"], best["res"]


def train_stage(task, args):
    hp_file = C.MODELS_DIR / f"{task}_best_hparams.json"
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.skip_search:
        saved = json.loads(hp_file.read_text(encoding="utf-8"))
        hp, cv = saved["hparams"], saved["cv"]
    else:
        hp, cv = search(task, args.trials, args.search_epochs, args.seed)

    final_epochs = args.final_epochs or max(15, round(statistics.median(cv["best_epochs"]) * 1.15) + 1)
    print(f"\n===== [{task}] 최종 학습 (train+val 전체) {final_epochs} epochs: {hp}")
    data = C.DET_YAML if task == "det" else C.CLS_DATA_DIR
    save_dir = run(task, hp, data, final_epochs, C.RUNS_DIR / task, "final", patience=final_epochs)
    # 최종 모델은 val이 학습 데이터와 같으므로 best가 아닌 마지막 에폭 가중치를 사용
    shutil.copy2(save_dir / "weights" / "last.pt", C.DET_WEIGHTS if task == "det" else C.CLS_WEIGHTS)
    hp_file.write_text(json.dumps({"hparams": hp, "cv": cv, "final_epochs": final_epochs},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{task}] 저장 완료. CV fitness {cv['mean']:.4f} ± {cv['std']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["det", "cls", "all"], default="all")
    ap.add_argument("--trials", type=int, default=0, help="후보 수 (기본: 탐지 4, 분류 5)")
    ap.add_argument("--search-epochs", type=int, default=30)
    ap.add_argument("--final-epochs", type=int, default=0, help="0이면 CV 최적 에폭으로 자동 결정")
    ap.add_argument("--skip-search", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (C.FOLDS_DIR / "folds.json").exists():
        raise SystemExit("먼저 prepare_data.py 를 실행하세요.")
    for task in (["det", "cls"] if args.stage == "all" else [args.stage]):
        t = argparse.Namespace(**vars(args))
        t.trials = args.trials or (4 if task == "det" else 5)
        train_stage(task, t)


if __name__ == "__main__":
    main()
