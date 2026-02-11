# 라벨

import glob
import os

from dotenv import load_dotenv

# 증강된 라벨 폴더 경로 (.env 로드)
load_dotenv()
base_path = os.getenv("YOLO_PATH")
LBL_DIR = os.path.join(base_path, "labels", "train")

# 파일 목록 가져오기
txt_files = glob.glob(os.path.join(LBL_DIR, "*.txt"))

print(f"📂 탐색 경로: {LBL_DIR}")
print(f"📄 총 라벨 파일 수: {len(txt_files)}")

if txt_files:
    # 첫 2개 파일만 내용 읽어보기
    for f in txt_files[:2]:
        print(f"\n--- {os.path.basename(f)} ---")
        with open(f, "r") as file:
            content = file.read()
            if not content.strip():
                print("⚠️ [비어있음] 내용이 없습니다!")
            else:
                print(content)
else:
    print("🚨 라벨 파일이 하나도 없습니다!")
