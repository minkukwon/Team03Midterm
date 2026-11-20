"""
YOLO11m 학습 (RTX 5070 Ti 16GB 기준 설정)

사용법
  python train_yolo11m.py --data <data.yaml 경로> [--epochs 100] [--batch 16] [--project <결과 폴더>]
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)   # 5070 Ti 16GB: yolo11m@640 에서 여유 있게 들어감
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="yolo11m")
    ap.add_argument("--workers", type=int, default=4)  # Windows 에서는 너무 크면 오히려 느려짐
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        patience=30,          # 30 epoch 동안 개선이 없으면 조기 종료
        cos_lr=True,
        close_mosaic=10,
        cache="ram",          # 데이터가 작아서 RAM 캐시가 빠름
        amp=True,
        seed=42,
        deterministic=True,
        project=args.project,
        name=args.name,
        plots=True,
    )


if __name__ == "__main__":
    main()
