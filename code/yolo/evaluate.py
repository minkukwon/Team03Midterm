"""성능 평가 (학습에 쓰지 않은 test 데이터)
  1) 탐지 성능: YOLO11m  Precision / Recall / mAP50 / mAP50-95 (클래스별 포함)
  2) 분류 성능: YOLO11m-cls  크롭 이미지 정확도
  3) 최종 시스템: 최적화된 판단 파라미터로 단일 이미지 + 모의 정면·측면 쌍 평가
     (F1, AI 안경 미탐률, 일반 안경 오탐률, Confusion Matrix, 3단계 판정 분포)
결과는 work/eval_report.json 에 저장.
"""
import json

from ultralytics import YOLO

import config as C
from optimize_decision import collect, evaluate_params, make_pairs
from pipeline import AIGlassesSystem, score_view


def main(split="test"):
    report = {}
    for task in ("det", "cls"):   # 5-fold 교차검증 결과 (train+val)
        hp = C.MODELS_DIR / f"{task}_best_hparams.json"
        if hp.exists():
            cv = json.loads(hp.read_text(encoding="utf-8"))["cv"]
            report[f"{task}_5fold_cv"] = {"fitness_mean": cv["mean"], "fitness_std": cv["std"],
                                          "folds": cv["folds"], **cv.get("detail", {})}

    det = YOLO(str(C.DET_WEIGHTS))
    m = det.val(data=str(C.DET_YAML), split=split, imgsz=640, plots=True,
                project=str(C.RUNS_DIR / "eval"), name=f"det_{split}", exist_ok=True, verbose=False)
    report["detection"] = {
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
        "per_class_mAP50": {m.names[c]: float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50)},
    }

    cls = YOLO(str(C.CLS_WEIGHTS))
    mc = cls.val(data=str(C.CLS_DATA_DIR), split=split, plots=True,
                 project=str(C.RUNS_DIR / "eval"), name=f"cls_{split}", exist_ok=True, verbose=False)
    report["classification"] = {"top1": float(mc.top1)}

    system = AIGlassesSystem()
    items = collect(system, split)
    pairs = make_pairs(items, 100)
    single, paired = evaluate_params(items, pairs, system.params)
    report["system_params"] = system.params
    report["system_single_image"] = single
    report["system_simulated_pairs"] = paired
    report["per_image"] = []
    for it in items:
        s = score_view(it["raw"], system.params)["score"]
        report["per_image"].append({"image": it["path"].name, "label": "AI" if it["y"] else "Normal",
                                    "score": s, "verdict": C.verdict(s)})

    out = C.WORK_DIR / f"eval_report_{split}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, ensure_ascii=False, indent=2))
    print(f"저장: {out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
