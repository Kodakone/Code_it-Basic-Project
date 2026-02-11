import os
from pathlib import Path

from dotenv import load_dotenv, set_key  # set_key 추가
from ultralytics import YOLO


def train_single_model():
    load_dotenv()

    # 0. path 불러오기
    base_path = os.getenv("YOLO_PATH")

    if not base_path:
        print("❌ .env 파일에서 YOLO_PATH를 찾을 수 없습니다!")
        return

    yaml_file = Path(base_path) / "dataset.yaml"

    # 1. 모델 리스트 가져오기
    project_root = Path(base_path).parent.parent
    model_dir = project_root / "model"
    model_files = list(model_dir.glob("*.pt"))

    if not model_files:
        print(f"🚨 {model_dir}에 .pt 파일이 없습니다!")
        return

    # 2. 사용자 선택 인터페이스
    print("\n--- [학습 가능한 모델 목록] ---")
    for i, f in enumerate(model_files):
        print(f"[{i}] {f.name}")

    try:
        choice = int(
            input("\n👉 학습할 모델 번호를 입력하세요: ")
        )  # yolov11[m] 쓸거니, 1번 입력.
        selected_model = model_files[choice]
    except (ValueError, IndexError):
        print("❌ 잘못된 번호입니다. 프로그램을 종료합니다.")
        return

    # 저장될 폴더 이름 미리 정의
    custom_name = f"trained_{selected_model.stem}"  # 접두사 수정해서 다르게 적용 가능
    print(f"\n🚀 선택된 모델 [{selected_model.name}] 학습을 시작합니다.")

    # 3. 학습 실행
    model = YOLO(str(selected_model))
    save_dir = project_root / "runs" / "train"

    model.train(
        data=str(yaml_file),
        epochs=60,  # epoch 60
        imgsz=640,
        batch=8,  # local에 맞게 조정
        device=0,  # local에 맞게 조정 (GPU 번호)
        workers=2,  # local에 맞게 조정
        augment=True,
        project=str(save_dir),
        name=custom_name,  # 1번 작업 후, 선두 이름 변경해야 따로 저장
        exist_ok=True,  # 기존 폴더가 있어도 그 안에 덮어쓰거나 이어서 저장
    )

    # 4. 학습 완료 후 .env 파일 업데이트
    env_path = project_root / ".env"
    if env_path.exists():
        # TARGET_MODEL_NAME 항목을 방금 학습한 폴더명으로 업데이트
        set_key(str(env_path), "TARGET_MODEL_NAME", custom_name)
        print(f'\n✅ .env 업데이트 완료: TARGET_MODEL_NAME="{custom_name}"')
    else:
        print("\n⚠️ .env 파일을 찾을 수 없어 업데이트를 건너뜁니다.")


if __name__ == "__main__":
    train_single_model()
