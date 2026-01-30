import os
import glob
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env에서 환경변수 가져오기 
base_path = os.getenv("LOG_FILE_PATH") # env 경로 중 택 1

# 경로 연결 - Train / Test / Annotation
train_path = os.path.join(base_path, "train_images")  
test_path = os.path.join(base_path, "test_images")
anno_path = os.path.join(base_path, 'train_annotations')

# Annotation file 1개씩 불러오기
json_pattern = os.path.join(
    anno_path,
    '*_json',   # K-xxxx_json
    'K-*',      # 내부 K-xxxx
    '*.json'    # json 파일
)

anno_files = glob.glob(json_pattern)                                                # 파일 찾기
anno_files = sorted(anno_files, key=lambda x: os.path.basename(os.path.dirname(x))) # 이름순 정렬된 Annotation List

# 결과 경로 출력
print(f"Dataset 경로 확인: {base_path}")

for i, f in enumerate(anno_files[:1]):      # 앞에 1개만 출력
    print(f"Annotation {i+1}번째 경로: {f}")  # 1개씩 가져와서 쓸 수 있음

