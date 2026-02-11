.env 파일 생성 후, # 개인 Dataset file 경로 지정하기

```
# raw data 절대 경로 [본인 경로 맞게 수정]
LOG_FILE_PATH="D:/Code_it/Basic Project/data/raw"

```

데이터 경로 load (dataset_load.py)\
↓\
데이터 매핑 mapping.py\
↓\
데이터 분할 & .yaml 생성 split_yolo\
↓
이후, augment/ 내 증강들 실행. (필요한 증강들 선택하여 사용. 순서 상관 X)
1. 전체 데이터 증강: run_general_augmenataion.py
2. 희소 class 데이터 증강: augmentation2.py
3. 희소 class 데이터 Copy-Paste: copy-paste_v5.py


데이터 증강 후, 모델 훈련 및 테스트 실행은 README_bin.md 참고
