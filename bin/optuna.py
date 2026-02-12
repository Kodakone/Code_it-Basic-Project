import os
from pathlib import Path

from dotenv import load_dotenv
from ultralytics import YOLO


def optuna_model():
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
    print(f"\n🚀 선택된 모델 [{selected_model.name}] 최적화를 시작합니다.")

    # 3. 학습 실행
    model = YOLO(str(selected_model))
    save_dir = project_root / "runs" / "optuna"

    # 하이퍼파라미터 튜닝 실행
    # 이 코드가 실행되면 Optuna가 알아서 cls, box, lr 등을 조정하며 반복 학습합니다.
    model.tune(
        data=str(yaml_file),
        epochs=30,  # 각 시도당 학습 에포크 (시간 절약을 위해 약간 줄임)
        iterations=50,  # 총 몇 번의 조합을 시도할 것인가 (최소 30회 이상 추천)
        imgsz=640,  # 안전한 640 유지
        project=str(save_dir),
        name="pill_optuna",
        use_ray=False,  # 단일 GPU라면 False
    )


if __name__ == "__main__":
    optuna_model()
