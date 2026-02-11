bin folder 사용법

.env 파일에 다음과 같이 2가지 추가. LOG_FILE_PATH와 별개 작동

```
# YOLO dataset 절대 경로 [본인 경로 맞게 수정]
YOLO_PATH = "D:/Code_it/Basic Project/data/yolo_dataset_aug"

# runs/ 폴더 내 target 이름
TARGET_MODEL_NAME='trained_yolo11m'

```

이후, 다음 순서대로 진행.

경로 체크(data_precheck.py)\
↓\
모델 훈련 train.py\
↓\
파인 튜닝 fine_tuning.py (필요시)\
↓\
모델 시각화 테스트 test.py\
↓\
모델 csv 파일 추출 extration.py
