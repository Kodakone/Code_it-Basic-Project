import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env에서 환경변수 가져오기
data_dir = os.getenv("DATA_DIR")
log_file_path = os.getenv("LOG_FILE_PATH")
db_backup_path = os.getenv("DB_BACKUP_PATH")

# 경로 연결 예시
user_file = os.path.join(data_dir, "user_data.csv")

print("DATA_DIR:", data_dir)
print("파일 경로:", user_file)
print("로그 파일 경로:", log_file_path)
print("DB 백업 경로:", db_backup_path)
