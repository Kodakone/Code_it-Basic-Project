import gc
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm
from ultralytics import YOLO

# ==========================================
# ⚙️ [설정] 경로 및 환경 변수
# ==========================================
load_dotenv(override=True)

# 프로젝트 루트 계산
project_root = Path(__file__).resolve().parent.parent

# 텍스트 파일에서 카테고리 ID 로드
TARGET_TXT_PATH = project_root / "bin" / "real_class_ids.txt"

if TARGET_TXT_PATH.exists():
    with open(TARGET_TXT_PATH, "r") as f:
        # 각 줄을 읽어서 공백 제거 후 숫자가 있는 경우만 정수로 변환하여 리스트 생성
        real_class_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
    print(
        f"✅ {len(real_class_ids)}개의 카테고리를 {TARGET_TXT_PATH.name}에서 로드했습니다."
    )
else:
    real_class_ids = []
    print(f"🚨 {TARGET_TXT_PATH} 파일을 찾을 수 없습니다!")

# .env에서 최신 모델 이름 가져오기
target_name = os.getenv("TARGET_MODEL_NAME")

# ★ 파인튜닝 모델 우선 로드, 없으면 일반 학습 모델 로드
ft_path = (
    project_root / "runs" / "fine_tuning" / f"{target_name}_FT" / "weights" / "best.pt"
)
train_path = project_root / "runs" / "train" / f"{target_name}" / "weights" / "best.pt"

if ft_path.exists():
    MODEL_PATH = str(ft_path)
    print(f"✅ 최종 추론에 파인튜닝 모델을 사용합니다: {MODEL_PATH}")
elif train_path.exists():
    MODEL_PATH = str(train_path)
    print(f"⚠️ 파인튜닝 모델이 없어 일반 학습 모델을 사용합니다: {MODEL_PATH}")
else:
    MODEL_PATH = None
    print("🚨 모델 파일을 찾을 수 없습니다! 학습을 먼저 완료해주세요.")

# 테스트 이미지 경로 및 결과 저장 경로
TEST_IMG_DIR = project_root / "data" / "raw" / "test_images"
RESULT_DIR = project_root / "runs" / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)  # 결과 폴더 자동 생성

OUTPUT_CSV = f"submission_{target_name}_WBF.csv"


# ==========================================
# 🛠️ 헬퍼 함수
# ==========================================
def extract_number_from_filename(filename):
    numbers = re.findall(r"\d+", filename)
    if numbers:
        return str(int("".join(numbers)))
    return filename


def run_submission_final():
    if not MODEL_PATH:
        return

    model = YOLO(MODEL_PATH)
    image_files = [
        f for f in os.listdir(TEST_IMG_DIR) if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    if not image_files:
        return print(f"🚨 이미지 폴더가 비어있습니다: {TEST_IMG_DIR}")

    results_list = []
    annotation_id_counter = 1

    print(f"🚀 {len(image_files)}장 처리 시작 (WBF + TTA 적용)...")

    for fname in tqdm(image_files):
        path = os.path.join(TEST_IMG_DIR, fname)

        # 1. 모델 추론 (TTA 적용)
        preds = model.predict(
            path,
            conf=0.001,  # 낮은 점수도 일단 다 뽑음 (WBF에서 걸러짐)
            iou=0.85,  # NMS를 느슨하게 해서 박스를 많이 확보
            augment=True,  # TTA(Test Time Augmentation)로 정확도 극대화
            imgsz=640,
            verbose=False,
        )

        r = preds[0]
        h, w = r.orig_shape

        # 2. WBF 적용 (여러 박스를 하나로 융합)
        if len(r.boxes) > 0:
            boxes_in = [r.boxes.xyxyn.cpu().numpy().tolist()]
            scores_in = [r.boxes.conf.cpu().numpy().tolist()]
            labels_in = [r.boxes.cls.cpu().numpy().tolist()]

            boxes, scores, labels = weighted_boxes_fusion(
                boxes_in,
                scores_in,
                labels_in,
                weights=None,
                iou_thr=0.4,
                skip_box_thr=0.01,
            )
        else:
            boxes, scores, labels = [], [], []

        # 3. 데이터 저장 형식 변환
        image_id_str = extract_number_from_filename(fname)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = (
                boxes[i][0] * w,
                boxes[i][1] * h,
                boxes[i][2] * w,
                boxes[i][3] * h,
            )

            results_list.append(
                {
                    "annotation_id": annotation_id_counter,
                    "image_id": image_id_str,
                    "category_id": real_class_ids[int(labels[i])],
                    "bbox_x": float(x1),
                    "bbox_y": float(y1),
                    "bbox_w": float(x2 - x1),
                    "bbox_h": float(y2 - y1),
                    "score": float(scores[i]),
                }
            )
            annotation_id_counter += 1

        del preds, r
        if annotation_id_counter % 500 == 0:
            gc.collect()

    # 4. CSV 저장
    df = pd.DataFrame(results_list)
    target_columns = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ]

    if not df.empty:
        df = df[target_columns]
        output_path = RESULT_DIR / OUTPUT_CSV
        df.to_csv(output_path, index=False)
        print(f"\n✨ 제출 파일 생성 완료: {output_path}")
    else:
        print("⚠️ 검출 결과가 없어 CSV를 생성하지 않았습니다.")


if __name__ == "__main__":
    run_submission_final()
