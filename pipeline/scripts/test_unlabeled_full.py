"""
Scenario B 통합 테스트 — 사람 어노테이션 포함 (mode='full')
==============================================================
전체 흐름:
  1. 이미지 3개를 data/inbox/unlabeled/ 에 복사
  2. ingest_unlabeled_flow(mode='full') 실행
       rembg 자동 마스킹 → R2 업로드 → Turso pending 등록
       → Gradio 라벨링 대기 (poll 10초) → 라벨 감지 후 재개
       → VLM 캡셔닝 → 분류기 → 임베딩 → Turso(complete) → Qdrant
  3. DB 결과 확인 (status=complete, label_color 채워졌는지)
  4. inbox/unlabeled 정리

사전 조건:
  - python flows/gradio_labeler.py 가 실행 중이어야 함 (localhost:7860)
  - 이 스크립트 실행 후 브라우저에서 3장 라벨링하면 자동 재개됨

실행:
  python -X utf8 scripts/test_unlabeled_full.py
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import os

PROJECT_ROOT = Path(__file__).parent.parent.parent   # Label Studio Project/
PIPELINE     = PROJECT_ROOT / "pipeline"
LAB          = PROJECT_ROOT / "lab"
sys.path.insert(0, str(PIPELINE))
os.chdir(PIPELINE)   # 모든 상대 경로(output/, data/)가 pipeline/ 기준으로 동작

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

SAMPLE_IMAGES = [
    LAB / "K_fashion 이미지 sample" / "원천데이터" / "원천데이터_1" / "레트로" / "1028690.jpg",
    LAB / "K_fashion 이미지 sample" / "원천데이터" / "원천데이터_1" / "레트로" / "1029079.jpg",
    LAB / "K_fashion 이미지 sample" / "원천데이터" / "원천데이터_1" / "로맨틱" / "101858.jpg",
]
INBOX_DIR    = PIPELINE / "data" / "inbox" / "unlabeled"
TEST_IDS     = {p.stem for p in SAMPLE_IMAGES}

PASS = "[OK]"
FAIL = "[FAIL]"
results: dict[str, str] = {}


def clean_test_artifacts() -> None:
    """이전 테스트 흔적 제거: indexed_ids.txt + manual_labels.jsonl + 마스크 아카이브에서 TEST_IDS 항목 삭제"""
    cache = PIPELINE / "output" / "indexed_ids.txt"
    if cache.exists():
        ids = {l for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()}
        before = len(ids)
        ids -= TEST_IDS
        cache.write_text("\n".join(sorted(ids)), encoding="utf-8")
        if before != len(ids):
            print(f"  캐시 정리: {before} → {len(ids)}건  ({before - len(ids)}개 제거)")

    labels = PIPELINE / "output" / "manual_labels.jsonl"
    if labels.exists():
        kept, removed = [], 0
        for line in labels.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("file_id") in TEST_IDS:
                    removed += 1
                    continue
            except Exception:
                pass
            kept.append(line)
        labels.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        if removed:
            print(f"  manual_labels 정리: {removed}건 제거")

    # 이전 실행에서 생성된 마스크 파일 정리 (R2 업로드 오염 방지)
    archive = PIPELINE / "data" / "masked_images_archive" / "inbox_unlabeled"
    if archive.exists():
        removed_masks = 0
        for fid in TEST_IDS:
            for f in archive.glob(f"{fid}_*.jpg"):
                f.unlink()
                removed_masks += 1
        if removed_masks:
            print(f"  마스크 아카이브 정리: {removed_masks}개 제거")


def setup_inbox() -> list[str]:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    for f in INBOX_DIR.glob("*.jpg"):
        f.unlink()
    copied = []
    for src in SAMPLE_IMAGES:
        if not src.exists():
            print(f"  [경고] 샘플 이미지 없음: {src.name}")
            continue
        shutil.copy2(src, INBOX_DIR / src.name)
        copied.append(src.stem)
        print(f"  복사: {src.name} → inbox/unlabeled/")
    return copied


def cleanup_inbox():
    for f in INBOX_DIR.glob("*.jpg"):
        f.unlink()
    print("  inbox/unlabeled/ 정리 완료")


def verify_db(file_ids: list[str]) -> None:
    print("\n─── DB 결과 확인 ───")
    try:
        from libsql import connect
        from src.config import DB_URL, DB_ACCESS_TOKEN

        conn = connect(DB_URL, auth_token=DB_ACCESS_TOKEN, _uri=True)
        all_ok = True
        for fid in file_ids:
            rows = conn.execute(
                "SELECT file_id, category, status, label_color, dense_caption, pattern_size "
                "FROM fashion_items WHERE file_id = ?",
                (fid,)
            ).fetchall()
            if rows:
                r = rows[0]
                status_ok = r[2] == "complete"
                label_ok  = bool(r[3])
                mark = PASS if (status_ok and label_ok) else FAIL
                if not (status_ok and label_ok):
                    all_ok = False
                print(f"  {mark} {r[0]} | cat={r[1]} | status={r[2]} | color={r[3]} | ps={r[5]}")
                print(f"         dense_caption: {(r[4] or '')[:80]}...")
            else:
                print(f"  {FAIL} {fid} — DB에 없음")
                all_ok = False
        conn.close()
        results["db_verify"] = f"{PASS if all_ok else FAIL} {len(file_ids)}건 — status=complete, label_color 확인"
    except Exception as e:
        print(f"  {FAIL} DB 조회 오류: {e}")
        results["db_verify"] = f"{FAIL} {e}"


async def main():
    import time
    t0 = time.time()

    print("=" * 60)
    print("Scenario B 테스트 (mode=full) — 사람 어노테이션 포함")
    print("  rembg → pending 등록 → Gradio 라벨링 → 완전 색인")
    print("=" * 60)

    print("\n─── 이전 테스트 흔적 정리 ───")
    clean_test_artifacts()

    print("\n─── inbox 준비 ───")
    file_ids = setup_inbox()
    if not file_ids:
        print(f"{FAIL} 복사할 이미지 없음.")
        return
    print(f"  투입 이미지: {file_ids}")
    results["setup"] = f"{PASS} {len(file_ids)}건 inbox 투입"

    print("\n─── ingest_unlabeled_flow (mode=full, poll=10초) ───")
    print("  ★ 지금 http://localhost:7860 에서 이미지 3장을 라벨링하세요!")
    print("  ★ 라벨링 완료 후 flow가 자동으로 재개됩니다.\n")

    try:
        from flows.ingest_unlabeled import ingest_unlabeled_flow
        await ingest_unlabeled_flow(
            image_dir=str(INBOX_DIR),
            mode="full",
            concurrency=3,
            label_poll_interval=10,   # 테스트용: 10초마다 확인 (기본 300초)
        )
        results["flow"] = f"{PASS} 완료"
    except Exception as e:
        print(f"  {FAIL} flow 오류: {e}")
        results["flow"] = f"{FAIL} {e}"
        import traceback; traceback.print_exc()

    verify_db(file_ids)

    print("\n─── 정리 ───")
    cleanup_inbox()
    results["cleanup"] = PASS

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Scenario B (full) 테스트 완료 ({elapsed:.1f}초)")
    print("=" * 60)
    for step, status in results.items():
        print(f"  {step:<20} {status}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
