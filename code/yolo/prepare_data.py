"""데이터 준비 (5-fold 교차검증)
1) train/val/test 간 중복 이미지(MD5) 제거본 생성 → work/det_data   (원본 labeldFiled는 수정하지 않음)
2) train+val 을 하나의 풀(pool)로 합쳐 5-fold 분할
   - 그룹: 동일 파일(MD5) + 거의 같은 사진(dHash 해밍거리 <= NEAR_DUP)은 같은 fold에 묶음 → 누수 방지
   - 층화: 이미지 단위 클래스(AI / Normal / NoGlasses / 배경) 비율을 fold마다 비슷하게
   - test 12장은 fold에 넣지 않고 최종 평가용으로 보존
3) fold별 탐지 yaml + 분류(yolo11m-cls)용 크롭 폴더 생성
4) 전체 풀로 최종 학습할 yaml / 분류 폴더 생성
"""
import hashlib
import json
import shutil
from collections import Counter

import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

import config as C

SPLITS = ("train", "val", "test")
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CLS_FOLDERS = {C.CLS_AI: "AIGlasses", C.CLS_NORMAL: "NormalGlasses"}
PAD = 0.10       # 크롭 여백 비율
NEAR_DUP = 4     # dHash(64bit) 해밍거리 이하이면 같은 사진으로 간주
SEED = 0


# ── 공통 유틸 ─────────────────────────────────────────
def images(split, root=None):
    d = (root or C.CLEAN_DATA_DIR) / "images" / split
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def label_path(img):
    return img.parent.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")


def read_labels(img):
    lp = label_path(img)
    if not lp.exists():
        return []
    out = []
    for line in lp.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append((int(parts[0]), *map(float, parts[1:5])))
    return out


def strat_label(img):
    cls = {l[0] for l in read_labels(img)}
    if cls & {C.CLS_AI, C.CLS_LENS}:
        return "AI"
    if C.CLS_NORMAL in cls:
        return "Normal"
    if C.CLS_NONE in cls:
        return "NoGlasses"
    return "background"


def dhash(img, size=8):
    g = np.asarray(Image.open(img).convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def key_of(img):
    return f"{img.parent.name}/{img.stem}"


# ── 1) 중복 제거본 ────────────────────────────────────
def build_clean_dataset():
    held_out = set()
    for s in ("val", "test"):
        for p in images(s, C.DATASET_DIR):
            held_out.add(hashlib.md5(p.read_bytes()).hexdigest())
    if C.CLEAN_DATA_DIR.exists():
        shutil.rmtree(C.CLEAN_DATA_DIR)
    removed = 0
    for s in SPLITS:
        (C.CLEAN_DATA_DIR / "images" / s).mkdir(parents=True)
        (C.CLEAN_DATA_DIR / "labels" / s).mkdir(parents=True)
        for p in images(s, C.DATASET_DIR):
            if s == "train" and hashlib.md5(p.read_bytes()).hexdigest() in held_out:
                removed += 1
                continue
            shutil.copy2(p, C.CLEAN_DATA_DIR / "images" / s / p.name)
            lp = C.DATASET_DIR / "labels" / s / (p.stem + ".txt")
            if lp.exists():
                shutil.copy2(lp, C.CLEAN_DATA_DIR / "labels" / s / lp.name)
    print(f"[중복제거] val/test와 동일한 train 이미지 {removed}장 제외 → {C.CLEAN_DATA_DIR}")


# ── 2) 그룹 + 층화 5-fold ─────────────────────────────
def group_ids(pool):
    """MD5 동일 또는 dHash 근접 이미지를 union-find로 묶음."""
    parent = list(range(len(pool)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    md5 = [hashlib.md5(p.read_bytes()).hexdigest() for p in pool]
    hashes = np.stack([dhash(p) for p in pool])
    merged = 0
    for i in range(len(pool)):
        dist = (hashes[i + 1:] != hashes[i]).sum(1)
        for j in np.nonzero(dist <= NEAR_DUP)[0] + i + 1:
            if find(i) != find(j):
                parent[find(j)] = find(i)
                merged += 1
        for j in range(i + 1, len(pool)):
            if md5[i] == md5[j] and find(i) != find(j):
                parent[find(j)] = find(i)
                merged += 1
    print(f"[그룹] 동일/유사 사진 {merged}쌍을 같은 fold로 묶음")
    return [find(i) for i in range(len(pool))]


def make_folds(pool):
    y = [strat_label(p) for p in pool]
    groups = group_ids(pool)
    sgkf = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=SEED)
    fold_of = np.zeros(len(pool), dtype=int)
    for k, (_, val_idx) in enumerate(sgkf.split(pool, y, groups)):
        fold_of[val_idx] = k
    for k in range(C.N_FOLDS):
        cnt = Counter(y[i] for i in range(len(pool)) if fold_of[i] == k)
        print(f"   fold{k}: {int((fold_of == k).sum()):3d}장  {dict(cnt)}")
    return fold_of, y


# ── 3) 크롭 (분류용) ──────────────────────────────────
def crop_all(pool_and_test):
    """이미지별 정답 박스 크롭을 work/crops/<split>/<class>/ 에 저장. 반환: key → [(class, crop_path)]"""
    crop_root = C.WORK_DIR / "crops"
    if crop_root.exists():
        shutil.rmtree(crop_root)
    out = {}
    for img_p in pool_and_test:
        labels = [l for l in read_labels(img_p) if l[0] in CLS_FOLDERS]
        if not labels:
            continue
        im = Image.open(img_p).convert("RGB")
        W, H = im.size
        for i, (c, xc, yc, w, h) in enumerate(labels):
            bw, bh = w * W * (1 + 2 * PAD), h * H * (1 + 2 * PAD)
            x1, y1 = max(0, xc * W - bw / 2), max(0, yc * H - bh / 2)
            x2, y2 = min(W, xc * W + bw / 2), min(H, yc * H + bh / 2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            dst = crop_root / img_p.parent.name / CLS_FOLDERS[c] / f"{img_p.stem}_{i}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.crop((x1, y1, x2, y2)).save(dst, quality=95)
            out.setdefault(key_of(img_p), []).append((CLS_FOLDERS[c], dst))
    return out


def write_cls_dir(root, split_to_keys, crops):
    if root.exists():
        shutil.rmtree(root)
    for split, keys in split_to_keys.items():
        for name in CLS_FOLDERS.values():
            (root / split / name).mkdir(parents=True, exist_ok=True)
        for k in keys:
            for cname, src in crops.get(k, []):
                shutil.copy2(src, root / split / cname / f"{k.replace('/', '_')}_{src.name}")


def write_list(path, imgs):
    path.write_text("\n".join(p.as_posix() for p in imgs) + "\n", encoding="utf-8")


def write_yaml(path, train, val, test):
    data = {"path": C.CLEAN_DATA_DIR.as_posix(), "train": train, "val": val, "test": test, "names": C.DET_NAMES}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main():
    build_clean_dataset()
    pool = images("train") + images("val")
    test = images("test")
    print(f"[풀] train+val {len(pool)}장 / test(보존) {len(test)}장")
    fold_of, y = make_folds(pool)
    crops = crop_all(pool + test)
    test_keys = [key_of(p) for p in test]

    if C.FOLDS_DIR.exists():
        shutil.rmtree(C.FOLDS_DIR)
    for k in range(C.N_FOLDS):
        d = C.FOLDS_DIR / f"fold{k}"
        d.mkdir(parents=True)
        tr = [p for p, f in zip(pool, fold_of) if f != k]
        va = [p for p, f in zip(pool, fold_of) if f == k]
        write_list(d / "train.txt", tr)
        write_list(d / "val.txt", va)
        write_yaml(d / "data.yaml", (d / "train.txt").as_posix(), (d / "val.txt").as_posix(), "images/test")
        write_cls_dir(d / "cls", {"train": [key_of(p) for p in tr], "val": [key_of(p) for p in va],
                                  "test": test_keys}, crops)

    # 최종 학습용: 풀 전체 (val은 모니터링용으로 풀 자체를 사용, 에폭 수는 CV 결과로 결정)
    write_list(C.WORK_DIR / "all.txt", pool)
    write_yaml(C.DET_YAML, (C.WORK_DIR / "all.txt").as_posix(), (C.WORK_DIR / "all.txt").as_posix(), "images/test")
    pool_keys = [key_of(p) for p in pool]
    write_cls_dir(C.CLS_DATA_DIR, {"train": pool_keys, "val": pool_keys, "test": test_keys}, crops)

    (C.FOLDS_DIR / "folds.json").write_text(json.dumps(
        [{"path": str(p), "fold": int(f), "label": l} for p, f, l in zip(pool, fold_of, y)],
        ensure_ascii=False, indent=1), encoding="utf-8")
    n = Counter(c for v in crops.values() for c, _ in v)
    print(f"[cls] 크롭 {dict(n)}  → fold별 {C.FOLDS_DIR}/fold*/cls , 전체 {C.CLS_DATA_DIR}")


if __name__ == "__main__":
    main()
