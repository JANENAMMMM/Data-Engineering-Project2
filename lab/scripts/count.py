"""
count.py — 파일/task 수 카운트
실행:
  python scripts/count.py                     # labels 파일 수 + tasks_final.json task 수
  python scripts/count.py path/to/tasks.json  # 해당 JSON의 task 수만
"""
import sys
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
PIPELINE = _PROJECT_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE))

if len(sys.argv) > 1:
    json_path = Path(sys.argv[1])
    with open(json_path, encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"전체 task 수: {len(tasks)}  ({json_path.name})")
else:
    labels_dir = PIPELINE / "data" / "labels"
    label_count = sum(1 for p in labels_dir.rglob("*") if p.is_file())
    print(f"labels 폴더 파일 개수: {label_count}")

    tasks_path = PIPELINE / "output" / "tasks_final.json"
    if tasks_path.exists():
        with open(tasks_path, encoding="utf-8") as f:
            tasks = json.load(f)
        print(f"전체 task 수: {len(tasks)}  ({tasks_path.name})")
    else:
        print("output/tasks_final.json 없음")
