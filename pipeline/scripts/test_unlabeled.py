"""
Scenario B 통합 테스트 — 라벨 없는 이미지 처리 (mode='vlm_only')
=================================================================
Gradio 대기 없이 즉시 처리되는 vlm_only 모드로 전체 흐름 검증.

흐름:
  1. K_fashion sample에서 이미지 3개를 data/inbox/unlabeled/ 에 복사
  2. ingest_unlabeled_flow(mode='vlm_only') 실행
        rembg 자동 마스킹 → R2 업로드 → VLM 캡셔닝
        → 분류기 추론 → 임베딩 → Turso(vlm_only) → Qdrant
  3. DB 결과 확인
  4. inbox/unlabeled 정리

실행:
  python -X utf8 scripts/test_unlabeled.py
"""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import os

PROJECT_ROOT = Path(__file__).parent.parent.parent   # Label Studio Project/
PIPELINE     = PROJECT_ROOT / "pipeline"
LAB          = PROJECT_ROOT / "lab"
sys.path.insert(0, str(PIPELINE))
os.chdir(PIPELINE)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── 설정 ──────────────────────────────────────────────────────────────────────
# 라벨 없는 샘플 — 라벨링 테스트(1016530/53/618)와 겹치지 않는 이미지 사용
SAMPLE_IMAGES = [
    LAB / "K_fashion 이미지 sample" / "원천데이터" / "원천데이터_1" / "레트로" / "1028690.jpg",
    LAB / "K_fashion 이미지 sample" / "원천데이터" / "원천데이터_1" / "레트로" / "1029079.jpg",
    LAB / "K_fashion 이미지 sample" / "원천데이터" / "원천데이터_1" / "로맨틱" / "101858.jpg",
]
INBOX_DIR = PIPELINE / "data" / "inbox" / "unlabeled"

PASS = "[OK]"
FAIL = "[FAIL]"

results: dict[str, str] = {}


# ── 전처리: inbox 에 이미지 복사 ─────────────────────────────────────────────
def setup_inbox() -> list[str]:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)

    # 이미 있는 파일 초기화 (이전 테스트 잔여 방지)
    for f in INBOX_DIR.glob("*.jpg"):
        f.unlink()

    copied = []
    for src in SAMPLE_IMAGES:
        if not src.exists():
            print(f"  [경고] 샘플 이미지 없음, 건너뜀: {src.name}")
            continue
        dst = INBOX_DIR / src.name
        shutil.copy2(src, dst)
        copied.append(src.stem)
        print(f"  복사: {src.name} → inbox/unlabeled/")

    return copied


# ── 후처리: inbox 정리 ────────────────────────────────────────────────────────
def cleanup_inbox():
    for f in INBOX_DIR.glob("*.jpg"):
        f.unlink()
    print("  inbox/unlabeled/ 정리 완료")


# ── DB 결과 확인 ─────────────────────────────────────────────────────────────
def verify_db(file_ids: list[str]) -> None:
    print("\n─── DB 결과 확인 ───")
    try:
        from libsql import connect
        from src.config import DB_URL, DB_ACCESS_TOKEN

        conn = connect(DB_URL, auth_token=DB_ACCESS_TOKEN, _uri=True)
        for fid in file_ids:
            rows = conn.execute(
                "SELECT file_id, category, status, dense_caption, pattern_size FROM fashion_items WHERE file_id = ?",
                (fid,)
            ).fetchall()
            if rows:
                r = rows[0]
                print(f"  {PASS} {r[0]} | cat={r[1]} | status={r[2]} | ps={r[4]}")
                print(f"         dense_caption: {(r[3] or '')[:80]}...")
            else:
                print(f"  {FAIL} {fid} — DB에 없음")
        conn.close()
        results["db_verify"] = f"{PASS} {len(file_ids)}건 DB 확인"
    except Exception as e:
        print(f"  {FAIL} DB 조회 오류: {e}")
        results["db_verify"] = f"{FAIL} {e}"


# ── 메인 ─────────────────────────────────────────────────────────────────────
async def main():
    import time
    t0 = time.time()

    print("=" * 60)
    print("Scenario B 테스트 (mode=vlm_only)")
    print("  라벨 없음 → rembg 마스킹 → VLM → 분류기 → DB/Qdrant")
    print("=" * 60)

    # Step 1: inbox 준비
    print("\n─── inbox 준비 ───")
    file_ids = setup_inbox()
    if not file_ids:
        print(f"{FAIL} 복사할 이미지 없음. 경로 확인 필요.")
        return
    print(f"  투입 이미지: {file_ids}")
    results["setup"] = f"{PASS} {len(file_ids)}건 inbox 투입"

    # Step 2: ingest_unlabeled_flow 실행
    print("\n─── ingest_unlabeled_flow (mode=vlm_only) ───")
    try:
        from flows.ingest_unlabeled import ingest_unlabeled_flow
        await ingest_unlabeled_flow(
            image_dir=str(INBOX_DIR),
            mode="vlm_only",
            concurrency=3,
        )
        results["flow"] = f"{PASS} 완료"
    except Exception as e:
        print(f"  {FAIL} flow 오류: {e}")
        results["flow"] = f"{FAIL} {e}"
        import traceback; traceback.print_exc()

    # Step 3: DB 결과 확인
    verify_db(file_ids)

    # Step 4: inbox 정리
    print("\n─── 정리 ───")
    cleanup_inbox()
    results["cleanup"] = f"{PASS}"

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Scenario B 테스트 완료 ({elapsed:.1f}초)")
    print("=" * 60)
    for step, status in results.items():
        print(f"  {step:<20} {status}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
