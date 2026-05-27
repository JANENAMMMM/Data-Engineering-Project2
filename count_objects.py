from pathlib import Path
import json

path = Path(__file__).parent / "tasks_final.json"

with open(path, encoding="utf-8") as f:
    tasks = json.load(f)

print(f"전체 task 수: {len(tasks)}")
