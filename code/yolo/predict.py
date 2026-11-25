"""정면·측면 사진 2장으로 최종 판정 (CLI)
  py -3.12 predict.py front.jpg side.jpg [--save out_dir]
"""
import argparse
from pathlib import Path

from pipeline import AIGlassesSystem


def fmt(v):
    return "-" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("front")
    ap.add_argument("side")
    ap.add_argument("--save", default=None, help="박스가 그려진 이미지를 저장할 폴더")
    a = ap.parse_args()

    r = AIGlassesSystem().analyze(a.front, a.side)
    for name in ("front", "side"):
        v = r[name]
        print(f"[{name:5s}] 점수={fmt(v['score'])}  상태={v['status']}  "
              f"탐지={fmt(v['p_det'])}  분류={fmt(v['p_cls'])}  렌즈={fmt(v['lens_conf'])}")
    print(f"통합 점수 = {fmt(r['score'])}  ({r['fusion']})")
    print(f"최종 판정 = {r['verdict']}")
    if a.save:
        out = Path(a.save)
        out.mkdir(parents=True, exist_ok=True)
        r["front"]["image"].save(out / "front_result.jpg")
        r["side"]["image"].save(out / "side_result.jpg")


if __name__ == "__main__":
    main()
