# 출처:
#   upload_with_retry, key_exists: SeoJimin1234/DE_pj2_Storage/r2_upload.py 이식
#   upload_masking_data 패턴: tldusdmlskr/fashion-search annotation/VLM_captioning
#     polygon_masking/upload_masking_to_r2.py 이식
#   Prefect @task 래핑 + 통합: 신규 작성

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prefect import task

from src.config import R2_BUCKET, R2_PUBLIC_URL
from src.r2_client import get_r2_client


# ── 저수준 헬퍼 ──────────────────────────────────────────────────────────────

def key_exists(bucket: str, key: str) -> bool:
    """출처: SeoJimin1234/DE_pj2_Storage/r2_upload.py — key_exists()"""
    s3 = get_r2_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def upload_with_retry(
    data: bytes | Path,
    bucket: str,
    r2_key: str,
    retries: int = 4,
) -> bool:
    """
    출처: SeoJimin1234/DE_pj2_Storage/r2_upload.py — upload_with_retry() 이식
    변경: local_path(str) 대신 bytes 또는 Path 허용
    exponential backoff cap(10s): tldusdmlskr/fashion-search _upload_one() 참고
    """
    s3 = get_r2_client()
    for attempt in range(retries):
        try:
            body = data.read_bytes() if isinstance(data, Path) else data
            s3.put_object(Bucket=bucket, Key=r2_key, Body=body)
            return True
        except Exception as e:
            wait = min(2 ** attempt, 10)  # 출처: fashion-search _upload_one 패턴
            if attempt < retries - 1:
                print(f"  R2 재시도 ({attempt+1}/{retries}) {r2_key}: {e} — {wait}s 대기")
                time.sleep(wait)
            else:
                print(f"  R2 최종 실패: {r2_key}")
                return False


def _list_existing_remote_keys(bucket: str, prefix: str) -> set[str]:
    """
    출처: tldusdmlskr/fashion-search annotation/VLM_captioning
          polygon_masking/upload_masking_to_r2.py — _list_existing_remote_names() 이식
    """
    s3        = get_r2_client()
    paginator = s3.get_paginator("list_objects_v2")
    existing  = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            existing.add(obj["Key"])
    return existing


# ── Prefect Task ─────────────────────────────────────────────────────────────

@task(name="upload-masked-images-r2", retries=1)
def upload_masked_images(
    local_dir: str,
    remote_prefix: str = "masking_data/",
    workers: int = 16,
    manifest_path: str | None = None,
) -> dict[str, str]:
    """
    출처: tldusdmlskr/fashion-search annotation/VLM_captioning
          polygon_masking/upload_masking_to_r2.py — upload_masking_data() 이식
    manifest 기반 재개 가능 / ThreadPoolExecutor 16 workers / 500건마다 체크포인트
    반환: {파일명_stem: public_url}
    """
    local_dir_p = Path(local_dir)
    manifest_p  = Path(manifest_path) if manifest_path else local_dir_p / "upload_manifest.txt"

    # 완료 목록 로드 (출처: fashion-search manifest 패턴)
    done: set[str] = set()
    if manifest_p.exists():
        done = set(manifest_p.read_text(encoding="utf-8").splitlines())
        print(f"매니페스트: {len(done)}개 이미 업로드됨")

    all_files = sorted(local_dir_p.glob("*.jpg"))
    remaining = [f for f in all_files if f.name not in done]
    print(f"업로드 대상: {len(remaining)}개 / 전체 {len(all_files)}개")

    urls:      dict[str, str] = {}
    completed: list[str]      = []

    def _upload_one(file_path: Path) -> tuple[str, str | None]:
        """출처: fashion-search _upload_one() 이식"""
        r2_key = f"{remote_prefix}{file_path.name}"
        if key_exists(R2_BUCKET, r2_key):
            return file_path.stem, f"{R2_PUBLIC_URL}/{r2_key}"
        ok = upload_with_retry(file_path, R2_BUCKET, r2_key)
        return file_path.stem, (f"{R2_PUBLIC_URL}/{r2_key}" if ok else None)

    # ThreadPoolExecutor 16 workers (출처: DE_pj2_Storage MAX_WORKERS=16 / fashion-search workers=16)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (stem, url) in enumerate(pool.map(_upload_one, remaining), 1):
            if url:
                urls[stem] = url
                completed.append(remaining[i - 1].name)
            # 500건마다 매니페스트 저장 (출처: fashion-search 500건 체크포인트)
            if i % 500 == 0:
                with open(manifest_p, "a", encoding="utf-8") as mf:
                    mf.write("\n".join(completed) + "\n")
                completed = []
                print(f"  {i}/{len(remaining)} 완료")

    if completed:
        with open(manifest_p, "a", encoding="utf-8") as mf:
            mf.write("\n".join(completed) + "\n")

    print(f"R2 업로드 완료: {len(urls)}건 성공")
    return urls
