"""
deploy.py — 모든 Prefect flow를 work pool에 등록

inbox_watcher.py 방식(serve)과 달리 이 방식은
별도 Prefect 서버 + 워커가 실행 중일 때 사용.

사용법:
  # 1. Prefect 서버 시작 (별도 터미널)
  prefect server start

  # 2. 워커 시작 (별도 터미널, 풀 없으면 자동 생성)
  prefect worker start --pool "local-process" --type process

  # 3. 이 스크립트로 모든 flow 등록
  python deployments/deploy.py

  # 4. 이후 CLI 또는 UI에서 실행
  prefect deployment run "ingest-labeled/ingest-labeled-prod"

  # inbox_watcher 단순 실행 방식 (서버 불필요, 권장)
  python deployments/inbox_watcher.py
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from flows.ingest_labeled import ingest_labeled_flow
from flows.ingest_unlabeled import ingest_unlabeled_flow
from flows.retrain_classifier import retrain_classifier_flow
from deployments.inbox_watcher import inbox_watcher_flow

WORK_POOL = "local-process"


def deploy_all() -> None:
    print(f"Prefect flow 등록 중 (work pool: {WORK_POOL})...")

    deployments = [
        # inbox 감시 — 60초 주기
        inbox_watcher_flow.deploy(
            name="inbox-watcher-prod",
            work_pool_name=WORK_POOL,
            interval=timedelta(seconds=60),
            description="inbox 폴더 60초 주기 감시 → 신규 파일 자동 적재",
        ),
        # 라벨 있는 데이터 수동 실행
        ingest_labeled_flow.deploy(
            name="ingest-labeled-prod",
            work_pool_name=WORK_POOL,
            description="라벨+이미지 신규 데이터 완전 파이프라인 (10,000건↑ 재학습 포함)",
        ),
        # 라벨 없는 데이터
        ingest_unlabeled_flow.deploy(
            name="ingest-unlabeled-prod",
            work_pool_name=WORK_POOL,
            description="라벨 없는 이미지: rembg → Gradio 라벨링 대기 → 적재",
        ),
        # 분류기 재학습 (수동 트리거용)
        retrain_classifier_flow.deploy(
            name="retrain-classifier-prod",
            work_pool_name=WORK_POOL,
            description="FashionAttributeClassifier LS 어노테이션 기반 재학습",
        ),
    ]

    print("\n등록 완료:")
    for d in deployments:
        print(f"  - {d}")

    print("\n실행 명령어 예시:")
    print("  prefect deployment run 'ingest-labeled/ingest-labeled-prod'")
    print("  prefect deployment run 'retrain-classifier/retrain-classifier-prod'")
    print("\nPrefect UI: http://localhost:4200")


if __name__ == "__main__":
    deploy_all()
