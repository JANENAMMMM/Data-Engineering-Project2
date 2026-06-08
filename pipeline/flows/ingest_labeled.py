# 출처: 신규 작성 — Scenario A Prefect flow (라벨 있는 신규 데이터 자동 처리)
# 수정: ES → Qdrant, E5/FashionCLIP → marqo-fashionSigLIP, masked_by_id 버그 수정

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from prefect import flow, get_run_logger

from flows.shared.tasks_caption import run_vlm_caption_batch, build_index_fields_batch
from src.caption import _parse_item_info
from src.label_parser import _parse_json_file
from flows.shared.tasks_classify import classify_batch
from flows.shared.tasks_embed import embed_images_siglip, embed_texts_siglip
from flows.shared.tasks_mask import mask_images_local
from flows.shared.tasks_qdrant import upsert_qdrant
from flows.shared.tasks_r2 import upload_masked_images
from flows.shared.tasks_rdb import (
    ensure_table,
    filter_already_indexed,
    upsert_rdb,
)

RETRAIN_THRESHOLD = 10_000  # 이 건수 이상 신규 적재 시 분류기 자동 재학습


@flow(name="ingest-labeled", log_prints=True)
async def ingest_labeled_flow(
    image_dir: str = "data/inbox/labeled/images",
    label_dir: str = "data/inbox/labeled/labels",
    masked_output_dir: str = "data/masked_images_archive/inbox",
    classifier_model: str = "models/fashion_classifier_v5.pt",
    concurrency: int = 50,
) -> None:
    """
    Scenario A: 라벨(JSON) + 이미지 동시 도착 시 완전 자동 처리
    실행: prefect deployment run ingest-labeled/default
    """
    logger = get_run_logger()
    run_id = str(uuid.uuid4())[:8]

    # ── Step 1: 테이블 준비 + 중복 제거 ──────────────────────────────────
    ensure_table()
    all_ids = [p.stem for p in Path(image_dir).glob("*.jpg")]
    if not all_ids:
        logger.info(f"image_dir에 이미지 없음: {image_dir}")
        return

    new_ids = filter_already_indexed(all_ids)
    if not new_ids:
        logger.info("모두 기처리됨. 종료.")
        return

    logger.info(f"신규 {len(new_ids)}건 처리 시작 (run_id={run_id})")

    # ── Step 2: 폴리곤 마스킹 ────────────────────────────────────────────
    # 출처: 현재 프로젝트 src/masking.py 래핑
    masked_results = mask_images_local(image_dir, label_dir, masked_output_dir)
    if not masked_results:
        logger.warning("마스킹 결과 없음. 라벨 JSON 확인 필요.")
        return

    # (file_id, category) → masked_result 매핑 [버그 수정: 이전엔 file_id만으로 매핑]
    masked_by_key = {(r["file_id"], r["category"]): r for r in masked_results}

    # ── Step 3: R2 업로드 ────────────────────────────────────────────────
    # 출처: fashion-search upload_masking_to_r2.py + DE_pj2_Storage 이식
    r2_urls = upload_masked_images(
        local_dir=masked_output_dir,
        remote_prefix="masking_data/",
        manifest_path=f"output/r2_manifest_{run_id}.txt",
    )

    # ── Step 4: VLM 캡셔닝 (외부 터미널 권장) ────────────────────────────
    # 출처: 현재 프로젝트 src/caption.py 래핑
    caption_path = f"output/captions_{run_id}.jsonl"

    _KOR_TO_EN = {"상의": "top", "하의": "bottom", "아우터": "outerwear", "원피스": "dress"}
    label_hints: dict[str, str] = {}
    for json_path in Path(label_dir).glob("**/*.json"):
        try:
            file_id, entry = _parse_json_file(json_path)
            for item in entry.get("items", []):
                item_info = item.get("item_info", "")
                kor_type = ""
                for ln in item_info.strip().splitlines():
                    if ln.startswith("타입:"):
                        kor_type = ln.split(":", 1)[1].strip().split("/")[0].strip()
                        break
                cat_en = _KOR_TO_EN.get(kor_type, "")
                if not cat_en:
                    continue
                parsed  = _parse_item_info(item_info)
                kor_cat = parsed.get("카테고리", "")
                parts   = [f"대분류: {kor_type}"]
                if kor_cat:
                    parts.append(f"카테고리: {kor_cat}")
                label_hints[f"{file_id}_{cat_en}"] = "\n".join(parts)
        except Exception:
            pass
    logger.info(f"label_hints 구성: {len(label_hints)}개 항목")

    await run_vlm_caption_batch(masked_output_dir, caption_path, concurrency=concurrency,
                                label_hints=label_hints or None)

    # ── Step 5: 인덱스 필드 생성 (dense_caption, flat_tags) ──────────────
    # 출처: 현재 프로젝트 src/caption.py (build_dense_caption, build_flat_tags)
    index_rows = build_index_fields_batch(caption_path, label_dir)

    # image_url + run_id + _masked_path 보강
    for row in index_rows:
        fid      = row["file_id"]
        cat      = row.get("category", "")
        stem     = f"{fid}_{cat}"
        row["image_url"]       = r2_urls.get(stem, "")
        row["pipeline_run_id"] = run_id
        # (file_id, category) 키로 정확한 masked_path 참조
        masked = masked_by_key.get((fid, cat))
        row["_masked_path"] = masked["output_path"] if masked else ""

    # ── Step 6: 분류기 추론 (패턴위치/크기/끝단마감 + 하의 전용 속성) ────────
    # 출처: fashion-search classifier-v4/classifier/inference.py 래핑
    masked_paths    = [row["_masked_path"] for row in index_rows]
    is_bottom_flags = [row.get("category") == "bottom" for row in index_rows]
    clf_results     = classify_batch(
        masked_paths,
        model_path=classifier_model,
        bottom_model_path="models/fashion_classifier_bottom.pt",
        is_bottom=is_bottom_flags,
    )

    for row, clf in zip(index_rows, clf_results):
        if clf:
            row["pattern_position"]  = clf.get("pattern_position", [])
            row["pattern_size"]      = clf.get("pattern_size", "")
            row["trim"]              = clf.get("trim", "")
            row["bottom_length"]     = clf.get("bottom_length", "")
            row["bottom_waist_rise"] = clf.get("bottom_waist_rise", "")

    # ── Step 7: 이미지 임베딩 (marqo-fashionSigLIP, 768d) ────────────────
    # 출처: 신규 작성 (사용자 제공 코드 기반)
    image_vecs = await asyncio.to_thread(embed_images_siglip, masked_paths)

    # ── Step 8: RDB 적재 (Turso) ─────────────────────────────────────────
    # 출처: DE_pj2_Storage 패턴 이식
    for row in index_rows:
        row.pop("_masked_path", None)  # 내부 참조용 필드 제거

    upsert_rdb(index_rows, status="complete")

    # ── Step 9: Qdrant 적재 (visual 컬렉션, 768d) ────────────────────────
    # 출처: 신규 작성 (사용자 제공 Qdrant 코드 기반)
    upsert_qdrant(index_rows, image_vecs)

    n_indexed = len(index_rows)
    logger.info(f"완료: {n_indexed}건 색인 (run_id={run_id})")

    # ── Step 10: 분류기 자동 재학습 트리거 ───────────────────────────────────
    # 신규 적재량 >= RETRAIN_THRESHOLD(10,000)이면 LS 어노테이션 기반 재학습
    if n_indexed >= RETRAIN_THRESHOLD:
        logger.info(
            f"신규 적재 {n_indexed}건 >= {RETRAIN_THRESHOLD}건 임계값"
            " → 분류기 자동 재학습 시작"
        )
        from flows.retrain_classifier import retrain_classifier_flow
        # 신규 항목 목록을 records로 전달 → train_classifier_v4 로직으로 저신뢰도 샘플 추출
        retrain_records = [
            {"image_id": r["file_id"], "image_url": r.get("image_url", ""),
             "folder": r.get("category", "")}
            for r in index_rows if r.get("image_url")
        ]
        success = retrain_classifier_flow(
            model_path=classifier_model,
            records=retrain_records,
            update_qdrant=True,
            qdrant_update_limit=min(n_indexed, 5000),
        )
        if success:
            logger.info("분류기 재학습 완료. 새 모델 적용됨.")
        else:
            logger.warning("분류기 재학습 건너뜀 (LS 어노테이션 부족).")
