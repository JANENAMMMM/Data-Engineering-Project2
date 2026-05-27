import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LS_URL     = os.environ["LS_URL"]
API_TOKEN  = os.environ["LS_API_TOKEN"]
PROJECT_ID = int(os.environ["LS_PROJECT_ID"])

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
}


def test_auth():
    """인증 테스트"""
    resp = requests.get(f"{LS_URL}/api/projects/{PROJECT_ID}", headers=headers)
    print(f"인증 테스트: {resp.status_code}")
    if resp.status_code == 200:
        print(f"  프로젝트: {resp.json().get('title')}")
        return True
    else:
        print(f"  응답: {resp.text[:200]}")
        return False


def get_all_tasks():
    """프로젝트의 모든 task ID와 file_id 매핑 가져오기"""
    tasks_map = {}  # {file_id: ls_task_id}
    page = 1
    while True:
        resp = requests.get(
            f"{LS_URL}/api/tasks",
            headers=headers,
            params={"project": PROJECT_ID, "page": page, "page_size": 100}
        )
        if resp.status_code != 200:
            print(f"조회 실패: {resp.status_code}")
            break
        data = resp.json()
        task_list = data.get("tasks", [])
        if not task_list:
            break
        for t in task_list:
            fid = t.get("data", {}).get("file_id")
            if fid:
                tasks_map[int(fid)] = t["id"]
        print(f"  page {page}: {len(task_list)}개 조회")
        if not data.get("next"):
            break
        page += 1
    return tasks_map


def update_tasks(tasks_final_path: str):
    """tasks_final_2.json 기준으로 기존 LS task의 data 업데이트"""
    if not test_auth():
        print("인증 실패. API_TOKEN을 확인하세요.")
        return

    print("\nLS task 목록 가져오는 중...")
    ls_map = get_all_tasks()  # {file_id: ls_task_id}
    print(f"총 {len(ls_map)}개 task 매핑 완료")

    with open(tasks_final_path, encoding="utf-8") as f:
        new_tasks = json.load(f)

    updated, skipped = 0, 0
    for task in new_tasks:
        fid = task["data"]["file_id"]
        ls_id = ls_map.get(fid)
        if ls_id is None:
            skipped += 1
            continue

        resp = requests.patch(
            f"{LS_URL}/api/tasks/{ls_id}",
            headers=headers,
            json={"data": task["data"]}
        )
        if resp.status_code in (200, 201):
            updated += 1
            if updated % 100 == 0:
                print(f"  {updated}개 업데이트 완료...")
        else:
            print(f"  FAIL task {ls_id} (file_id={fid}): {resp.status_code}")

    print(f"\n완료: {updated}개 업데이트, {skipped}개 스킵(LS에 없음)")


if __name__ == "__main__":
    update_tasks("tasks_final_2.json")
