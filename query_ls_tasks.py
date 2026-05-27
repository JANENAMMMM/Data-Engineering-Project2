import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

LS_URL     = os.environ["LS_URL"]
API_TOKEN  = os.environ["LS_API_TOKEN"]
PROJECT_ID = int(os.environ["LS_PROJECT_ID"])

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "ngrok-skip-browser-warning": "true",  # ngrok 경고 페이지 스킵
}

# 비어있는 task ID 목록 (Label Studio task ID)
TARGET_IDS = [24310, 24316, 24375, 24486, 24555]

print("Label Studio task ID → file_id 매핑:")
print("-" * 50)

for task_id in TARGET_IDS:
    resp = requests.get(
        f"{LS_URL}/api/tasks/{task_id}",
        headers=headers
    )
    if resp.status_code == 200:
        data = resp.json()
        file_id    = data.get("data", {}).get("file_id", "N/A")
        item_info  = data.get("data", {}).get("item_info", "")
        image      = data.get("data", {}).get("image", "")
        print(f"  LS task {task_id} → file_id={file_id}")
        print(f"    item_info: '{item_info[:30]}...' " if item_info else f"    item_info: (비어있음)")
        print(f"    image: {image.split('/')[-1]}")
    else:
        print(f"  LS task {task_id} → 조회 실패 ({resp.status_code})")
    print()
