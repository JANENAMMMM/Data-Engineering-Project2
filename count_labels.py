from pathlib import Path

labels_dir = Path(__file__).parent / "labels"
count = sum(1 for p in labels_dir.rglob("*") if p.is_file())
print(f"labels 폴더 파일 개수: {count}")
