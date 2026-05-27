import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    's3',
    endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
)

BUCKET = 'project2'
LABEL_PREFIX = 'labeling/'  # 본인 R2 구조에 맞게

local_dir = Path("./labels")
local_dir.mkdir(exist_ok=True)

# JSON 파일 목록 가져오기
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=BUCKET, Prefix=LABEL_PREFIX):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.endswith('.json'):
            local_path = local_dir / Path(key).name
            s3.download_file(BUCKET, key, str(local_path))

print("Labels downloaded")