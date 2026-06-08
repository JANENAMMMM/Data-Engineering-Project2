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
    print(f"정리: {cache}  {before} → {len(ids)}건")
