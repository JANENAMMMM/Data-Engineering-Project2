import json
from pathlib import Path

# pipeline/ 기준 경로 (os.chdir 적용 후 test 스크립트들이 여기에 씀)
pipeline_cache = Path(__file__).parent / "pipeline" / "output" / "indexed_ids.txt"
# 루트 기준 경로 (os.chdir 없이 실행된 경우)
root_cache = Path(__file__).parent / "output" / "indexed_ids.txt"

REMOVE_IDS = {"1028690", "1029079", "101858"}

for cache in [pipeline_cache, root_cache]:
    if not cache.exists():
        print(f"없음: {cache}")
        continue
    ids = set(cache.read_text(encoding="utf-8").splitlines()) - {""}
    before = len(ids)
    ids -= REMOVE_IDS
    cache.write_text("\n".join(sorted(ids)), encoding="utf-8")
    print(f"캐시 정리: {cache}  {before} → {len(ids)}건")

# manual_labels.jsonl 정리
labels_path = Path(__file__).parent / "pipeline" / "output" / "manual_labels.jsonl"
if labels_path.exists():
    lines = labels_path.read_text(encoding="utf-8").splitlines()
    kept, removed = [], 0
    for line in lines:
        if not line.strip():
            continue
        try:
            if json.loads(line).get("file_id") in REMOVE_IDS:
                removed += 1
                continue
        except Exception:
            pass
        kept.append(line)
    labels_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"manual_labels 정리: {removed}건 제거  ({labels_path})")
else:
    print(f"없음: {labels_path}")
