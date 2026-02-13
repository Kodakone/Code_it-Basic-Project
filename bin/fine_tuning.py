import os
from pathlib import Path

from dotenv import load_dotenv
from ultralytics import YOLO


def run_fine_tuning():
    # 0. 매 실행 시마다 .env 최신 상태 로드 (train.py가 수정한 값을 반영)
    load_dotenv(override=True)

    # 0. 환경 변수 및 경로 설정
    base_path = os.getenv("YOLO_PATH")  # .env에 등록된 데이터셋 경로
    target_name = os.getenv("TARGET_MODEL_NAME")  # 이름 지정

    if not base_path or not target_name:
        print("❌ .env에서 YOLO_PATH 또는 TARGET_MODEL_NAME을 찾을 수 없습니다!")
        return

    # 프로젝트 루트(Basic Project) 경로 계산 - 현재 파일이 bin 폴더 안에 있다면 .parent.parent가 루트.
    project_root = Path(__file__).resolve().parent.parent

    # 1. 베이스 모델 경로 (절대 경로로 변경) - 기존 train 결과물 중 best.pt를 가져오기
    base_model_path = (
        project_root / "runs" / "train" / target_name / "weights" / "best.pt"
    )

    if not base_model_path.exists():
        print(f"🚨 베이스 모델을 찾을 수 없습니다: {base_model_path}")
        print("💡 train.py를 먼저 실행하여 학습을 완료했는지 확인하세요.")
        return

    # 모델 로드
    model = YOLO(base_model_path)
    print(f"🔄 베이스 모델 로드 완료: {base_model_path}")
    yaml_file = Path(base_path) / "dataset.yaml"  # 데이터셋 YAML 경로

    # 2. 파인튜닝
    print(f"🚀 파인튜닝 시작: {target_name} -> {target_name}_FT")

    model.train(
        data=str(yaml_file),
        epochs=20,  # Train: 60 + Fine_Tuning: 20 ~ 40
        imgsz=640,  # img size
        batch=8,  # batch (train.py와 동일하게)
        # 파인튜닝 옵션 - (기존 학습 지식 보호 선에서)
        optimizer="AdamW",
        lr0=0.00005,  # 학습률 아주 낮게
        lrf=0.01,
        cos_lr=True,
        # 학습 도중 이미지를 섞거나(Mosaic) 자르는(Mixup) 기능은 끕니다.
        # 온전한 알약 모양을 그대로 보여주는 게 중요합니다.
        mosaic=0.0,  # 👈 끄기!
        mixup=0.0,  # 👈 끄기!
        copy_paste=0.0,  # 👈 끄기!
        # [Output 설정] 결과를 따로 저장하는 부분
        # project: '.../Basic Project/runs/fine_tuning' 폴더로 고정
        project=str(project_root / "runs" / "fine_tuning"),  # 결과가 저장될 상위 폴더
        name=f"{target_name}_FT",  # ★ 이 이름으로 새 폴더 생성 (기존 이름과 겹치지 않게)
        exist_ok=True,  # 폴더가 이미 있어도 덮어쓰기/이어서 저장
    )


if __name__ == "__main__":
    run_fine_tuning()
