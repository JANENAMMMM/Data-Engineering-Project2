import os
from dotenv import load_dotenv

load_dotenv()

# R2
R2_ENDPOINT_URL   = os.environ["R2_ENDPOINT_URL"]
R2_ACCESS_KEY_ID  = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET         = os.environ["R2_BUCKET"]
R2_PUBLIC_URL     = os.environ["R2_PUBLIC_URL"]
R2_BUCKET_PREFIX  = "image"

# Gemini
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL   = "gemini-2.5-flash-lite"  # 또는 "gemini-2.5-flash" 등 원하는 모델로 변경 가능

# Label Studio
LS_URL        = os.environ["LS_URL"]
LS_API_TOKEN  = os.environ["LS_API_TOKEN"]
LS_PROJECT_ID = int(os.environ["LS_PROJECT_ID"])

# Turso / SQLite (optional)
DB_URL = os.environ.get("DB_URL")
DB_ACCESS_TOKEN = os.environ.get("DB_ACCESS_TOKEN")

# 도메인 상수
CLOTHING_TYPES = ['상의', '하의', '아우터', '원피스']
