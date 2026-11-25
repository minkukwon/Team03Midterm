"""프로젝트 공통 설정 (경로, 클래스, 최종 판정 임계값)."""
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent              # code/yolo
PROJECT_DIR = CODE_DIR.parent.parent                    # Team03Midterm
DATASET_DIR = PROJECT_DIR / "labeldFiled"               # 원본 라벨링 데이터셋
RUNS_DIR = CODE_DIR / "runs"                            # 학습 결과
WORK_DIR = CODE_DIR / "work"                            # 생성되는 중간 데이터
CLEAN_DATA_DIR = WORK_DIR / "det_data"                  # 중복 제거된 탐지 데이터셋
DET_YAML = WORK_DIR / "data_det.yaml"                  # 실제 경로로 생성되는 탐지용 yaml
CLS_DATA_DIR = WORK_DIR / "cls_data"                    # 분류용 크롭 데이터셋 (최종 학습용)
FOLDS_DIR = WORK_DIR / "folds"                          # 5-fold 분할 (fold*/data.yaml, fold*/cls)
N_FOLDS = 5
MODELS_DIR = CODE_DIR / "models"                        # 최종 가중치 보관
DET_WEIGHTS = MODELS_DIR / "det_best.pt"
CLS_WEIGHTS = MODELS_DIR / "cls_best.pt"
FOLD_MODELS_DIR = MODELS_DIR / "folds"                  # fold별 가중치 (out-of-fold 예측용)
DECISION_PARAMS = MODELS_DIR / "decision_params.json"   # 최적화된 판단 파라미터

# ── 클래스 (labeldFiled/data.yaml 기준) ───────────────
DET_NAMES = {0: "AIGlasses", 1: "AIGlasses_Lens", 2: "NormalGlasses", 3: "NoGlasses"}
CLS_AI, CLS_LENS, CLS_NORMAL, CLS_NONE = 0, 1, 2, 3
GLASSES_CLASSES = (CLS_AI, CLS_NORMAL)

# ── 최종 판정 임계값 (요구사항) ───────────────────────
AI_THRESHOLD = 0.8       # 점수 >= 0.8  → AI 안경
NORMAL_THRESHOLD = 0.5   # 점수 <= 0.5  → AI 안경 아님
                         # 0.5 < 점수 < 0.8 → AI 안경 의심

LABEL_AI = "AI 안경"
LABEL_SUSPECT = "AI 안경 의심"
LABEL_NORMAL = "AI 안경 아님"
LABEL_RETAKE = "재촬영 필요"


def verdict(score):
    """통합 점수 → 최종 판정 문자열."""
    if score is None:
        return LABEL_RETAKE
    if score >= AI_THRESHOLD:
        return LABEL_AI
    if score <= NORMAL_THRESHOLD:
        return LABEL_NORMAL
    return LABEL_SUSPECT


# 기본 판단 파라미터 (optimize_decision.py 실행 전 사용)
DEFAULT_PARAMS = {
    "det_conf": 0.25,      # YOLO 탐지 신뢰도 기준 (README 초기 후보)
    "w_cls": 0.6,          # 안경 점수 = w_cls*분류모델 + (1-w_cls)*탐지모델
    "lens_weight": 0.5,    # 렌즈 탐지 시 점수 상향 비중
    "lens_conf": 0.25,     # 렌즈로 인정할 최소 신뢰도
    "fusion": "cond",      # mean | max | weighted | cond
    "w_front": 0.5,        # weighted / cond 에서 정면 가중치
}
