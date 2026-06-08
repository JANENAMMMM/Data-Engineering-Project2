"""
K-Fashion 자동화 파이프라인 — Scenario B 라이브 데모
=====================================================
이 스크립트는 수업 시연용입니다.

[시나리오 B]: 라벨이 없는 새 패션 이미지가 들어왔을 때,
자동화 파이프라인이 어떻게 처리하는지 보여줍니다.

흐름:
  ① 이미지 Inbox 투입
  ② AI 자동 배경 제거 (rembg)
  ③ R2 클라우드 업로드 + DB pending 등록
  ④ 사람(라벨러)이 Gradio UI에서 마스킹 + 라벨 입력
  ⑤ 파이프라인 자동 재개: VLM 캡셔닝 → 분류 → 임베딩 → 저장

실행 방법 (pipeline/ 폴더 기준):
  [터미널 1] cd pipeline && python flows/gradio_labeler.py
  [터미널 2] cd pipeline && python demo_scenario_b.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

# pipeline/ 폴더가 작업 디렉토리이자 패키지 루트
ROOT = Path(__file__).parent.resolve()   # = .../pipeline/
os.chdir(ROOT)                            # 상대경로(data/, output/ 등)가 pipeline/ 기준으로 동작
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT.parent / ".env")         # .env 는 프로젝트 루트에 위치

LAB = ROOT.parent / "lab"

# ── 데모 설정 ──────────────────────────────────────────────────────────────
DEMO_IMAGES = [
    LAB / "K_fashion 이미지 sample" / "원천데이터" / "원천데이터_1" / "레트로" / "1028690.jpg",
    LAB / "K_fashion 이미지 sample" / "원천데이터" / "원천데이터_1" / "레트로" / "1029079.jpg",
    LAB / "K_fashion 이미지 sample" / "원천데이터" / "원천데이터_1" / "로맨틱" / "101858.jpg",
]
INBOX_DIR         = ROOT / "data" / "inbox" / "unlabeled"
LABEL_POLL_SEC    = 15   # 데모용: 15초마다 라벨링 완료 체크 (실제: 300초)
GRADIO_PORT       = 7860


# ── 출력 유틸 ──────────────────────────────────────────────────────────────

def header(title: str) -> None:
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")

def step(n: int, msg: str) -> None:
    print(f"\n  [{n}] {msg}")

def ok(msg: str) -> None:
    print(f"      ✓  {msg}")

def warn(msg: str) -> None:
    print(f"      ⚠  {msg}")

def info(msg: str) -> None:
    print(f"      →  {msg}")


# ── 준비 / 정리 ──────────────────────────────────────────────────────────────

def setup_inbox() -> list[str]:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    for f in INBOX_DIR.glob("*.jpg"):
        f.unlink()

    copied = []
    for src in DEMO_IMAGES:
        if not src.exists():
            warn(f"샘플 이미지 없음: {src.name}")
            continue
        shutil.copy2(src, INBOX_DIR / src.name)
        ok(f"투입: {src.name}")
        copied.append(src.stem)
    return copied


def clean_cache(file_ids: list[str]) -> None:
    cache_path = ROOT / "output" / "indexed_ids.txt"
    if not cache_path.exists():
        return
    ids = set(cache_path.read_text(encoding="utf-8").splitlines())
    ids -= set(file_ids)
    cache_path.write_text("\n".join(sorted(ids)), encoding="utf-8")
    info("indexed_ids.txt 에서 테스트 ID 제거 완료")


def cleanup_inbox() -> None:
    for f in INBOX_DIR.glob("*.jpg"):
        f.unlink()
    ok("inbox/unlabeled 정리 완료")


# ── DB 결과 확인 ───────────────────────────────────────────────────────────

def show_results(file_ids: list[str]) -> bool:
    try:
        from libsql import connect
        from src.config import DB_URL, DB_ACCESS_TOKEN

        conn = connect(DB_URL, auth_token=DB_ACCESS_TOKEN, _uri=True)
        print()
        print(f"  {'file_id':<12} {'category':<12} {'status':<14} {'color':<10} {'pattern_size'}")
        print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*10} {'-'*12}")

        all_ok = True
        for fid in file_ids:
            rows = conn.execute(
                "SELECT file_id, category, status, label_color, pattern_size "
                "FROM fashion_items WHERE file_id=?", (fid,)
            ).fetchall()
            if rows:
                r = rows[0]
                status_ok = r[2] == "complete"
                mark = "✓" if status_ok else "✗"
                if not status_ok:
                    all_ok = False
                print(f"  {mark} {r[0]:<12} {(r[1] or '-'):<12} {r[2]:<14} {(r[3] or '-'):<10} {r[4] or '-'}")
            else:
                print(f"  ✗ {fid:<12} — DB에 없음")
                all_ok = False

        conn.close()
        return all_ok
    except Exception as e:
        warn(f"DB 조회 오류: {e}")
        return False


# ── 메인 데모 ─────────────────────────────────────────────────────────────

async def run_demo() -> None:
    t0 = time.time()

    header("K-Fashion 자동화 파이프라인 — Scenario B 라이브 데모")
    print("""
  시나리오:
    새로운 K-Fashion 이미지 3장이 시스템에 유입됩니다.
    파이프라인이 자동으로 배경을 제거하고 클라우드에 업로드한 뒤,
    라벨러에게 알립니다. 라벨러가 Gradio UI에서 마스킹 + 라벨을
    입력하면 파이프라인이 재개되어 완전한 인덱스를 생성합니다.
""")

    # ① Inbox 준비
    step(1, "이미지 Inbox 투입")
    file_ids = setup_inbox()
    if not file_ids:
        warn("투입할 이미지가 없습니다. DEMO_IMAGES 경로를 확인하세요.")
        return
    info(f"총 {len(file_ids)}개 이미지: {file_ids}")

    # indexed_ids 캐시 정리 (재실행 시 중복 방지)
    clean_cache(file_ids)

    # ② 파이프라인 실행
    step(2, "파이프라인 시작 (mode='full', Gradio 대기 포함)")
    print(f"""
  ★★★ 지금 브라우저에서 http://localhost:{GRADIO_PORT} 를 여세요! ★★★
  각 이미지에서 의류 영역을 브러시로 칠하고 라벨을 입력한 뒤
  '저장 후 다음 이미지 →' 를 눌러주세요.
  {len(file_ids)}장 모두 완료하면 파이프라인이 자동으로 재개됩니다.
""")

    from flows.ingest_unlabeled import ingest_unlabeled_flow

    try:
        await ingest_unlabeled_flow(
            image_dir=str(INBOX_DIR),
            mode="full",
            concurrency=5,
            label_poll_interval=LABEL_POLL_SEC,
        )
        ok("파이프라인 완료")
    except Exception as e:
        warn(f"파이프라인 오류: {e}")
        import traceback; traceback.print_exc()
        return

    # ③ 결과 확인
    header("파이프라인 처리 결과")
    success = show_results(file_ids)

    # ④ 정리
    step(4, "Inbox 정리")
    cleanup_inbox()

    elapsed = time.time() - t0
    header(f"데모 완료 — 소요시간: {elapsed:.1f}초")
    if success:
        print("  모든 이미지가 성공적으로 처리되어 DB + Qdrant에 저장되었습니다.\n")
    else:
        print("  일부 항목이 처리되지 않았습니다. 로그를 확인하세요.\n")


if __name__ == "__main__":
    asyncio.run(run_demo())
