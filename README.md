Code_it 7기 초급 프로젝트
=============

## **프로젝트 개요**
> 이미지 인식 기술을 헬스케어 분야에 접목

+ 유저가 모바일 App으로 자신이 복용중인 약 사진을 찍었을 때, 이미지 인식을 통해 해당 약에 대한 정보를 확인할 수 있는 모델을 만드는 미션
+ 또한, 유저의 건강 상태 및 함께 복용하면 안되는 약 등 헬스케어 정보를 유저들에게 제공

<img width="343" height="429" alt="image" src="https://github.com/user-attachments/assets/ad2c3445-4217-401c-b665-88b3e9f18f26" />

&nbsp;사진 속에 있는 최대 4개의 알약의 이름(클래스)과 위치(바운딩 박스)를 검출하는 모델을 구현 & 성능을 지속적으로 개선해나가는 것이 프로젝트 주 목표


모델 성능 검증은 코드잇 스프린트에서 별도로 세팅해둔 Kaggle에서 진행\
[https://www.kaggle.com/competitions/ai-07-object-detection/overview]

사용 Dataset\
[https://www.kaggle.com/competitions/ai-07-object-detection/data]


---------------------------------------

##  **폴더 구조**
> 핵심 Folder 구조 구성

```
BASIC PROJECT
├── augment/                      # 증강 3가지 실행 (전체 증강 / 희소 class 증강 / 희소 Class Copy-Paste)
│   ├── augmentation2.py            # 희소 class 증강
│   ├── copy-paste_v5.py            # 희소 Class Copy-Paste (augment_target.txt에서 목표 선정)
│   └── run_general_augmentation.py # 전체 증강
├── bin/                          # 모델 훈련 및 테스트 시행. 
│   ├── data_precheck.py            # 데이터 경로 check
│   ├── extraction.py               # 모델 csv 파일 추출 (real_class_ids 에서 카테고리 ID 로드)
│   ├── fine_tuning.py              # 파인 튜닝 시행 (Train 후 필요 시 사용)
│   ├── optuna.py                   # 최적 Hyperparameter 도출
│   ├── README_bin.md               # 모델 훈련 / 테스트 사용 설명서
│   ├── sanity_check.py             # 이상 상황 검사 
│   ├── test.py                     # 모델 시각화 테스트
│   └── train.py                    # 모델 훈련
├── data/                         # 데이터 모음 (git: x)
│   ├── raw/                        # 원본 data 기입
│   │    ├── train_images/             # train 이미지 (.png)
│   │    ├── train_annotations/        # train annotation (.json)
│   │    └── test_images/              # test 이미지 (.png)
│   ├── yolo_dataset/                # YOLO data 원본 (.ymal)                           < dataloader 파일 실행시 생성
│   └── yolo_dataset_aug/            # 증강 data 기입 (원본 + 전체 + 희소 + Copy_Paste)  < dataloader 파일 실행시 생성
├── dataloader/                   # 데이터 불러오기 관련 folder
│   ├── dataset_load.py              # .env 파일 경로 지정
│   ├── general_augmentation.py      # run_general_augmentation.py에서 데이터 끌어오는 용도. 실행 필요 X
│   ├── mapping.py                   # image & annotation 매핑 (gt)
│   ├── README_load.md               # dataloader 사용 설명서
│   └── split_yolo.py                # train / valid split & .ymal 변형 (gt)
├── model/                        # Model (YOLOv11 nano ~ large)
├── modelpt/                      # 실험 후 제출하였던 Model output
├── .gitignore
├── REAMME.py
└── ...

```

코드, 모델, 결과 등 주요 디렉토리 구조를 그림 또는 리스트 형태로 작성

---------------------------------------

## **실행 방법**

+ Required package version

> Python = 3.13.9\

torch>=	2.6.0
ultralytics>=	8.4.14
numpy>= 2.3.5	
pandas>= 3.0.0	
scikit-learn>= 1.8.0	
albumentations>= 2.0.8	
opencv-python>=	4.13.0.92	
tqdm>= 4.67.3	
iterative-stratification>= 0.1.9	
ensemble-boxes>= 1.0.9	

+ 해당 github fork 후,

data/ 폴더에 raw data 기입(프로젝트 원본 데이터) 이후, README_data.md 참고

.env 생성 후 Data 경로 본인 Local에 맞게 수정하기

이후, dataloader/ 폴더의 README_load.md 참고 후 file들 실행

그리고, augment/ 폴더 파일 3가지 실행하여 증강 실행. 

마지막으로, bin/ 폴더의 README_bin.md 참고하여 모델 훈련 및 테스트 실행 

---------------------------------------

- **모델 설명 및 결과**:
- [보고서]([https://github.com/유저/저장소/raw/main/manual.pdf?download=](https://github.com/Kodakone/Code_it-Basic-Project/edit/main/%EC%B4%88%EA%B8%89%20%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%20%EB%B3%B4%EA%B3%A0%EC%84%9C_3%ED%8C%80.pdf))
- [발표자료](https://github.com/유저/저장소/raw/main/manual.pdf?download=)

---------------------------------------
## **협업 내용**

> 팀원 역할

**Project Manager:고대권**
- 프로젝트의 협업 과정을 매니징하는 역할
- 애자일/스프린트 방식으로 단위를 분리하고 회의를 주도
- 협업 일지\
[https://github.com/Kodakone/Code_it_-/tree/main/협업일지(초급%20Project)]


**Data Engineer:곽민선, 이수민**
- 데이터 수집, 정제 및 전처리를 담당하고 데이터 파이프라인을 구축
- 협업 일지\
[[https://woolly-farm-a38.notion.site/_3-_-30627611ae6e803f84f7fcda0ceca9f7]](https://woolly-farm-a38.notion.site/_3-_-30627611ae6e803f84f7fcda0ceca9f7)\
[https://www.notion.so/3-2f7d6346486581c087b3e93e647580cd]

**Model Architect & Experimentation Lead:신민수, 윤성현**
- 딥러닝 모델을 설계하고 아키텍처를 결정
- 다양한 실험을 주도하고, 하이퍼파라미터 튜닝 및 모델 성능 평가를 담당
- 협업 일지\
[https://www.notion.so/Code_it-7-2f7a8c677cd18138bb2ae3fde4d54e96?source=copy_link
신현수님의 워크스페이스 on Notion]\
[https://www.notion.so/2f7e2626fac580fda0d4e3c2407f94eb?source=copy_link]


> 협업 과정

Weekdays
+ AM 9:00 ~ PM 6:00 Project 수행
+ PM 6:00 ~ PM 7:00 협업 일지 작성
+ PM 7:00 Pull Request Merge 수행

---------------------------------------
