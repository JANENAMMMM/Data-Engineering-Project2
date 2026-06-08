import requests
from src.config import LS_URL, LS_API_TOKEN, LS_PROJECT_ID

HEADERS = {
    "Authorization": f"Bearer {LS_API_TOKEN}",
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
}


def get_project():
    return requests.get(f"{LS_URL}/api/projects/{LS_PROJECT_ID}", headers=HEADERS)


def get_all_tasks() -> dict:
    """file_id → ls_task_id 매핑 반환"""
    tasks_map = {}
    page = 1
    while True:
        resp = requests.get(
            f"{LS_URL}/api/tasks",
            headers=HEADERS,
            params={"project": LS_PROJECT_ID, "page": page, "page_size": 100},
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
        if not data.get("next"):
            break
        page += 1
    return tasks_map


def patch_task(ls_task_id: int, data: dict) -> requests.Response:
    return requests.patch(
        f"{LS_URL}/api/tasks/{ls_task_id}",
        headers=HEADERS,
        json={"data": data},
    )


def get_task(ls_task_id: int) -> requests.Response:
    return requests.get(f"{LS_URL}/api/tasks/{ls_task_id}", headers=HEADERS)
