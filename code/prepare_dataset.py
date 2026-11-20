"""
labeldFiled 원본을 건드리지 않고, 학습용으로 정리된 사본 데이터셋을 만든다.

정리 내용
1) train + val 을 합친 뒤 완전히 같은 이미지(jpg/png 중복, train·val 양쪽에 들어간 사진)는 하나만 남긴다.
2) 거의 같은 이미지(같은 제품의 비슷한 컷)는 같은 그룹으로 묶어 train/val 에 나뉘지 않게 한다 (데이터 누수 방지).
3) 그룹 단위로 클래스 비율을 맞춰 train/val 을 다시 나눈다 (기본 80:20, seed 고정).
4) README 기준 3개 클래스(AIGlasses, AIGlasses_Lens, NormalGlasses)만 사용 → 클래스 3(NoGlasses) 줄은 제거.
5) test 는 그대로 복사 (최종 평가 전용).

사용법
  python prepare_dataset.py --src <labeldFiled 경로> --dst <출력 경로>
"""
import argparse
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

NAMES = ["AIGlasses", "AIGlasses_Lens", "NormalGlasses"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def dhash(im):
    a = np.asarray(im.convert("L").resize((17, 16), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def read_label(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) == 5 and int(p[0]) in (0, 1, 2):
            out.append(line.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near-dup", type=int, default=30, help="dHash 거리 이하를 같은 그룹으로 묶음 (256비트 중)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"출력 폴더가 비어 있지 않습니다: {dst}")

    # 1) train + val 수집
    items = []
    for split in ("train", "val"):
        for f in sorted((src / "images" / split).iterdir()):
            if f.suffix.lower() not in IMG_EXT:
                continue
            im = Image.open(f)
            items.append({
                "split": split, "img": f,
                "lab": src / "labels" / split / (f.stem + ".txt"),
                "h": dhash(im), "hm": dhash(ImageOps.mirror(im)),
                "png": f.suffix.lower() == ".png", "size": im.size[0] * im.size[1],
            })
    n = len(items)
    H = np.array([i["h"] for i in items]); M = np.array([i["hm"] for i in items])

    # 2) union-find 로 그룹 만들기
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    exact = list(range(n))  # 완전 중복(거리 0) 대표
    for i in range(n):
        d = np.minimum((H[i] != H).sum(1), (H[i] != M).sum(1))
        for j in np.where(d <= args.near_dup)[0]:
            if j <= i:
                continue
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a
            if d[j] == 0 and exact[j] == j:
                exact[j] = exact[i] if exact[i] != i else i

    # 완전 중복은 하나만 남김: png, train 쪽, 큰 해상도, 라벨 박스 많은 것 우선
    dup_groups = defaultdict(list)
    for i in range(n):
        r = i
        while exact[r] != r:
            r = exact[r]
        dup_groups[r].append(i)
    keep = []
    for idxs in dup_groups.values():
        best = max(idxs, key=lambda k: (items[k]["png"], items[k]["split"] == "train",
                                        len(read_label(items[k]["lab"])), items[k]["size"]))
        keep.append(best)

    groups = defaultdict(list)
    for k in keep:
        groups[find(k)].append(k)

    # 3) 그룹 단위 분할 (그룹의 대표 성격: AI / Normal / 기타)
    def kind(g):
        cls = Counter(int(l.split()[0]) for k in g for l in read_label(items[k]["lab"]))
        if cls[0] or cls[1]:
            return "AI"
        if cls[2]:
            return "Normal"
        return "None"

    rng = random.Random(args.seed)
    by_kind = defaultdict(list)
    for g in groups.values():
        by_kind[kind(g)].append(g)
    split_of = {}
    for k, gs in by_kind.items():
        rng.shuffle(gs)
        total = sum(len(g) for g in gs)
        target, acc = total * args.val_ratio, 0
        for g in gs:
            s = "val" if acc < target else "train"
            if s == "val":
                acc += len(g)
            for idx in g:
                split_of[idx] = s

    # 4) 복사
    for s in ("train", "val", "test"):
        (dst / "images" / s).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / s).mkdir(parents=True, exist_ok=True)
    stats = {"train": Counter(), "val": Counter(), "test": Counter()}
    counts = Counter()
    used_names = set()
    for idx, s in split_of.items():
        it = items[idx]
        name = it["img"].stem
        if name in used_names:  # 서로 다른 사진인데 이름만 같은 경우
            name = f"{name}_{it['split']}"
        used_names.add(name)
        shutil.copy2(it["img"], dst / "images" / s / (name + it["img"].suffix.lower()))
        lines = read_label(it["lab"])
        (dst / "labels" / s / (name + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[s] += 1
        for l in lines:
            stats[s][NAMES[int(l.split()[0])]] += 1
    for f in sorted((src / "images" / "test").iterdir()):
        if f.suffix.lower() not in IMG_EXT:
            continue
        shutil.copy2(f, dst / "images" / "test" / f.name)
        lines = read_label(src / "labels" / "test" / (f.stem + ".txt"))
        (dst / "labels" / "test" / (f.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts["test"] += 1
        for l in lines:
            stats["test"][NAMES[int(l.split()[0])]] += 1

    yaml = (f"path: {dst.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
            "names:\n" + "".join(f"  {i}: {nm}\n" for i, nm in enumerate(NAMES)))
    (dst / "data.yaml").write_text(yaml, encoding="utf-8")

    print(f"원본 train+val 이미지 {n}장 → 완전 중복 제거 후 {len(keep)}장, 그룹 {len(groups)}개")
    for s in ("train", "val", "test"):
        print(f"  {s}: {counts[s]}장, 박스 {dict(stats[s])}")
    print("data.yaml:", dst / "data.yaml")


if __name__ == "__main__":
    main()
