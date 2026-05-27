import os
import boto3, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    's3',
    endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
)
BUCKET = os.environ["R2_BUCKET"]

# R2 최상위 폴더 구조 확인
print("=== R2 최상위 폴더 목록 ===")
resp = s3.list_objects_v2(Bucket=BUCKET, Delimiter='/', MaxKeys=30)
for p in resp.get('CommonPrefixes', []):
    print(" ", p['Prefix'])

# image/ 하위 폴더 확인
print("\n=== image/ 하위 폴더 목록 ===")
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix='image/', Delimiter='/', MaxKeys=30)
for p in resp.get('CommonPrefixes', []):
    print(" ", p['Prefix'])

# 첫 번째 스타일 폴더 안 파일 5개 샘플
print("\n=== 실제 이미지 파일명 샘플 (첫 폴더 5개) ===")
resp2 = s3.list_objects_v2(Bucket=BUCKET, Prefix='image/', MaxKeys=5)
for obj in resp2.get('Contents', []):
    print(" ", obj['Key'])

# JSON의 파일이름과 비교
print("\n=== JSON 파일이름 vs path.stem 비교 ===")
labels_dir = Path("./labels")
for p in list(labels_dir.glob("*.json"))[:3]:
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    original_name = data["데이터셋 정보"]["파일 이름"]
    stem_name = p.stem + ".jpg"
    style = data["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"].get("스타일", [{}])[0].get("스타일", "")
    print(f"  JSON 파일이름: {original_name}")
    print(f"  stem+.jpg:    {stem_name}")
    print(f"  스타일:        {style}")
    print()
