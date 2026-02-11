import os
from pathlib import Path

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO


def run_test():
    # 0. 최신 .env 상태 로드
    load_dotenv(override=True)

    # 환경 변수 불러오기
    target_name = os.getenv("TARGET_MODEL_NAME")

    # 프로젝트 루트 및 경로 설정
    project_root = Path(__file__).resolve().parent.parent

    # 1. 테스트할 모델 경로 결정
    ft_model_path = (
        project_root
        / "runs"
        / "fine_tuning"
        / f"{target_name}_FT"
        / "weights"
        / "best.pt"
    )
    train_model_path = (
        project_root / "runs" / "train" / f"{target_name}" / "weights" / "best.pt"
    )

    if ft_model_path.exists():
        model_path = ft_model_path
        print(f"✅ 파인튜닝된 모델 로드: {model_path}")
    elif train_model_path.exists():
        model_path = train_model_path
        print(f"⚠️ 일반 학습 모델 로드: {model_path}")
    else:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        return

    model = YOLO(str(model_path))

    # 2. 테스트 이미지 샘플링
    test_image_dir = project_root / "data" / "raw" / "test_images"
    all_test_images = list(test_image_dir.glob("*.png"))

    if not all_test_images:
        print(f"❌ 이미지 없음: {test_image_dir}")
        return

    selected_images = all_test_images[:5]
    print(f"🔍 {len(selected_images)}장 테스트 및 팝업 출력 시작...")

    # 3. 예측 실행
    results = model.predict(
        source=[str(p) for p in selected_images],
        conf=0.25,
        iou=0.45,
        save=False,
        verbose=False,
    )

    # 4. 결과 화면 출력 (OpenCV 팝업 창)
    for result in results:
        file_name = os.path.basename(result.path)

        # 박스가 그려진 이미지를 numpy 배열(BGR)로 가져옴
        im_array = result.plot()
        window_name = f"YOLOv11 Result: {file_name}"

        # 크기 조절 로직
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        display_width = 600
        display_height = int(im_array.shape[0] * (display_width / im_array.shape[1]))
        cv2.resizeWindow(window_name, display_width, display_height)
        cv2.imshow(window_name, im_array)

        print(f"🖼️ 확인 중: {file_name} (창을 닫으려면 아무 키나 누르세요)")

        # 사용자가 아무 키나 누를 때까지 대기 (0은 무한 대기)
        _ = cv2.waitKey(0)

        # 현재 창 닫기 (다음 이미지를 위해)
        cv2.destroyWindow(window_name)

    # 모든 처리가 끝나면 모든 창 닫기
    cv2.destroyAllWindows()
    print("\n✅ 모든 이미지 확인이 완료되었습니다.")


if __name__ == "__main__":
    run_test()
