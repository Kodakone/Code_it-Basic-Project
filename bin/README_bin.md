### bin folder(모델 훈련 및 테스트 실행 파일) 사용법

.env 파일에 다음과 같이 2가지 추가. LOG_FILE_PATH와 별개 작동

```
# YOLO dataset 절대 경로 [본인 경로에 맞게 수정!!]
YOLO_PATH = "D:/Code_it/Basic Project/data/yolo_dataset_aug"

# runs/ 폴더 내 target 이름
TARGET_MODEL_NAME='trained_yolo11m'

```

이후, 다음 순서대로 진행.

**경로 체크 (data_precheck.py)**\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↓\
**모델 훈련 (train.py)**\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↓\
**파인 튜닝(필요시) (fine_tuning.py)**\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↓\
**모델 시각화 테스트 test.py**\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↓\
**모델 csv 파일 추출 extration.py**

실행 중 이상 상황 생겼을 경우, **sanity_check.py**로 파악하기

최적 성능을 도출하는 Hyperparameter를 조사 및 사용하길 원하면 **optuna.py** 파일 실행.\
실행 후, runs/tune/pill_optuna에 best_hyperparameters.yaml 생성. (탐색 시간 오래 걸림)\
해당 파일에 도출되는 best Hyperparameter 조합 확인하여, 학습에 사용
