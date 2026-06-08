"""
inbox_watcher.py — 신규 데이터 자동 감지 & 파이프라인 트리거

실행:
  python deployments/inbox_watcher.py

동작:
  1분마다 inbox 폴더 스캔
  신규 파일 발견 → ingest_labeled_flow / ingest_unlabeled_flow 즉시 실행
  Prefect UI(http://localhost:4200)에서 실행 이력 확인 가능

inbox 폴더 구조:
  data/inbox/labeled/images/    ← 라벨 있는 신규 원본 이미지 (.jpg)
  data/inbox/labeled/labels/    ← 대응하는 라벨 JSON (파일명=file_id)
  data/inbox/unlabeled/         ← 라벨 없는 신규 이미지 (.jpg)
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from prefect import flow, get_run_logger, serve

from flows.ingest_labeled import ingest_labeled_flow
from flows.ingest_unlabeled import ingest_unlabeled_flow

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
LABELED_IMAGE_DIR = str(ROOT / "data" / "inbox" / "labeled" / "images")
LABELED_LABEL_DIR = str(ROOT / "data" / "inbox" / "labeled" / "labels")
UNLABELED_DIR     = str(ROOT / "data" / "inbox" / "unlabeled")

# 처리 완료 기록 파일 (DB 조회 없이 로컬에서만 관리)
SEEN_FILE = ROOT / "output" / ".inbox_seen.txt"


def _get_seen() -> set[str]:
    """처리 완료된 file_id 로드 (로컬 파일만 읽음, DB 조회 없음)"""
    if not SEEN_FILE.exists():
        return set()
    return {l.strip() for l in SEEN_FILE.read_text(encoding="utf-8").splitlines() if l.strip()}


def _mark_seen(file_ids: set[str]) -> None:
    """처리 완료 file_id 기록"""
    existing = _get_seen()
    all_seen = existing | file_ids
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text("\n".join(sorted(all_seen)), encoding="utf-8")


def _scan_labeled_inbox() -> tuple[set[str], set[str]]:
    """
    labeled inbox 스캔.
    이미지 + 라벨 JSON이 모두 있는 file_id만 '준비 완료'로 분류.
    반환: (준비완료 file_ids, 라벨 없는 이미지 file_ids)
    """
    img_ids   = {p.stem for p in Path(LABELED_IMAGE_DIR).glob("*.jpg")}
    label_ids = {p.stem for p in Path(LABELED_LABEL_DIR).glob("*.json")}
    ready     = img_ids & label_ids       # 이미지 + 라벨 모두 있음
    img_only  = img_ids - label_ids       # 이미지만 있고 라벨 없음 (라벨 아직 안 왔음)
    return ready, img_only


def _scan_unlabeled_inbox() -> set[str]:
    """unlabeled inbox 스캔: 이미지만 있는 file_id"""
    return {p.stem for p in Path(UNLABELED_DIR).glob("*.jpg")}


@flow(name="inbox-watcher", log_prints=True)
async def inbox_watcher_flow() -> None:
    """
    1분 주기로 inbox 폴더를 스캔하여 신규 데이터를 감지하고 적절한 flow를 트리거.
    Prefect serve()에 의해 스케줄 실행됨.
    """
    logger = get_run_logger()
    seen   = _get_seen()

    # ── Labeled inbox 스캔 ───────────────────────────────────────────────────
    Path(LABELED_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    Path(LABELED_LABEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(UNLABELED_DIR).mkdir(parents=True, exist_ok=True)

    ready_ids, img_only_ids = _scan_labeled_inbox()
    new_labeled = ready_ids - seen

    if new_labeled:
        logger.info(
            f"[Labeled] 신규 {len(new_labeled)}건 감지 "
            f"(이미지+라벨 모두 준비됨) → ingest_labeled_flow 실행"
        )
        await ingest_labeled_flow(
            image_dir=LABELED_IMAGE_DIR,
            label_dir=LABELED_LABEL_DIR,
        )
        _mark_seen(new_labeled)
    else:
        logger.info(f"[Labeled] 신규 없음 (seen={len(seen)}, img_only={len(img_only_ids)})")

    # ── Unlabeled inbox 스캔 ──────────────────────────────────────────────────
    new_unlabeled = _scan_unlabeled_inbox() - seen

    if new_unlabeled:
        logger.info(
            f"[Unlabeled] 신규 {len(new_unlabeled)}건 감지 "
            "→ ingest_unlabeled_flow 실행"
        )
        await ingest_unlabeled_flow(
            image_dir=UNLABELED_DIR,
            mode="full",
        )
        _mark_seen(new_unlabeled)
    else:
        logger.info("[Unlabeled] 신규 없음")

    if not new_labeled and not new_unlabeled:
        logger.info("inbox 변경 없음. 다음 스캔까지 대기.")


if __name__ == "__main__":
    # Prefect serve — 1분마다 inbox_watcher_flow 실행
    # 같은 프로세스에서 ingest flows도 함께 서빙 (Prefect UI에서 수동 실행 가능)
    print("=" * 55)
    print("K-Fashion 파이프라인 자동화 시작")
    print(f"  감시 폴더 (labeled) : {LABELED_IMAGE_DIR}")
    print(f"  감시 폴더 (unlabeled): {UNLABELED_DIR}")
    print("  Prefect UI          : http://localhost:4200")
    print("  스캔 주기           : 60초")
    print("=" * 55)

    serve(
        # inbox 감시 — 60초마다 자동 실행
        inbox_watcher_flow.to_deployment(
            name="inbox-watcher",
            interval=timedelta(seconds=60),
            description="inbox 폴더 감시 → 신규 파일 시 자동 적재",
        ),
        # 수동 실행용 deployments (Prefect UI 또는 CLI에서 트리거)
        ingest_labeled_flow.to_deployment(
            name="ingest-labeled",
            description="라벨 있는 신규 데이터 전체 파이프라인 (수동 실행)",
        ),
        ingest_unlabeled_flow.to_deployment(
            name="ingest-unlabeled",
            description="라벨 없는 데이터: rembg → Gradio 라벨링 → 적재 (수동 실행)",
        ),
    )
