# 출처: 신규 작성 — Scenario B Prefect flow
# (라벨 없는 신규 데이터: rembg 자동 마스킹 → Gradio 수동 라벨링 → 파이프라인 재개)
# 수정: ES → Qdrant, E5/FashionCLIP → marqo-fashionSigLIP

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from prefect import flow, get_run_logger

from flows.shared.tasks_caption import run_vlm_caption_batch
from flows.shared.tasks_classify import classify_batch
from flows.shared.tasks_embed import embed_images_siglip
from flows.shared.tasks_mask import auto_mask_rembg
from flows.shared.tasks_qdrant import upsert_qdrant
from flows.shared.tasks_r2 import upload_masked_images
from flows.shared.tasks_rdb import (
    ensure_table,
    filter_already_indexed,
    upsert_rdb,
)

from src.caption import build_dense_caption, build_flat_tags

MANUAL_LABELS_PATH = Path("output/manual_labels.jsonl")
MASKED_OUTPUT_DIR  = "data/masked_images_archive/inbox_unlabeled"


def _load_manual_labels(file_ids: list[str]) -> dict[str, dict]:
    """output/manual_labels.jsonl에서 완료된 file_id 라벨 로드"""
    if not MANUAL_LABELS_PATH.exists():
        return {}
    target = set(file_ids)
    result: dict[str, dict] = {}
    with open(MANUAL_LABELS_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                fid = row.get("file_id", "")
                if fid in target:
                    result[fid] = row
            except Exception:
                pass
    return result


def _poll_for_labels(
    file_ids: list[str],
    poll_interval: int = 300,
    timeout_hours: int = 48,
) -> dict[str, dict]:
    """Gradio UI 완료 감지 (로컬 파일 폴링, DB 쿼리 없음)"""
    deadline = time.time() + timeout_hours * 3600
    target   = set(file_ids)

    while time.time() < deadline:
        done      = _load_manual_labels(file_ids)
        remaining = target - set(done.keys())
        print(f"라벨링 대기: {len(done)}/{len(target)} 완료 ({len(remaining)}개 남음)")
        if not remaining:
            return done
        time.sleep(poll_interval)

    return _load_manual_labels(file_ids)  # timeout 시 완료된 것만 반환


def _manual_to_existing(label: dict) -> dict:
    """Gradio 출력 dict → _parse_item_info 결과와 동일한 형태로 변환"""
    return {
        "카테고리":  label.get("카테고리"),
        "색상":     label.get("색상"),
        "서브색상":  label.get("서브색상"),
        "소재":     label.get("소재", []),
        "핏":       label.get("핏"),
        "기장":     label.get("기장"),
        "소매기장":  label.get("소매기장"),
        "넥라인":   label.get("넥라인"),
        "디테일":   label.get("디테일", []),
        "프린트":   label.get("프린트", []),
    }


@flow(name="ingest-unlabeled", log_prints=True)
async def ingest_unlabeled_flow(
    image_dir: str = "data/inbox/unlabeled",
    classifier_model: str = "models/fashion_classifier_v5.pt",
    mode: str = "full",
    concurrency: int = 50,
    label_poll_interval: int = 300,
) -> None:
    """
    Scenario B: 라벨 없는 이미지 처리
    mode='full'     → Gradio UI 라벨링 완료 대기 후 완전 색인 (권장)
    mode='vlm_only' → VLM 출력만으로 즉시 부분 색인 (긴급 적재)

    실행 전 Gradio UI 별도 실행 필요:
      python flows/gradio_labeler.py
    """
    logger = get_run_logger()
    run_id = str(uuid.uuid4())[:8]

    # ── Step 1: 신규 파일 감지 (캐시 파일 기반, DB 쿼리 없음) ───────────
    ensure_table()
    all_ids = [p.stem for p in Path(image_dir).glob("*.jpg")]
    if not all_ids:
        logger.info(f"image_dir에 이미지 없음: {image_dir}")
        return

    new_ids = filter_already_indexed(all_ids)
    if not new_ids:
        logger.info("모두 기처리됨. 종료.")
        return

    new_paths = [str(Path(image_dir) / f"{fid}.jpg") for fid in new_ids]
    logger.info(f"신규 {len(new_ids)}건 처리 시작 (mode={mode}, run_id={run_id})")

    # ── Step 2: rembg 자동 마스킹 ───────────────────────────────────────
    # 출처: 신규 작성 (rembg 라이브러리)
    masked_results = auto_mask_rembg(new_paths, MASKED_OUTPUT_DIR)
    masked_by_id   = {r["file_id"]: r for r in masked_results}

    # ── Step 3: R2 업로드 + pending 상태 RDB 등록 ────────────────────────
    r2_urls = upload_masked_images(
        local_dir=MASKED_OUTPUT_DIR,
        remote_prefix="masking_data/unlabeled/",
        manifest_path=f"output/r2_manifest_unlabeled_{run_id}.txt",
    )

    # pending 상태 등록 (단순 메타데이터만, SELECT 없음)
    pending_rows = [
        {
            "file_id":        r["file_id"],
            "category":       r["category"],
            "image_url":      r2_urls.get(Path(r["output_path"]).stem, ""),
            "pipeline_run_id": run_id,
        }
        for r in masked_results
    ]
    upsert_rdb(pending_rows, status="pending_label")

    # ── Step 4a: VLM-Only 모드 ───────────────────────────────────────────
    if mode == "vlm_only":
        logger.info("VLM-Only 모드: 구조화 라벨 없이 즉시 색인")
        caption_path = f"output/captions_unlabeled_{run_id}.jsonl"
        await run_vlm_caption_batch(MASKED_OUTPUT_DIR, caption_path, concurrency=concurrency)

        index_rows = _build_rows_from_caption(caption_path, masked_by_id, r2_urls, run_id, labeled_map=None)
        clf_results = classify_batch([r.get("_masked_path", "") for r in index_rows], model_path=classifier_model)
        _merge_clf(index_rows, clf_results)

        image_vecs = await asyncio.to_thread(embed_images_siglip, [r.pop("_masked_path", "") for r in index_rows])
        upsert_rdb(index_rows, status="vlm_only")
        upsert_qdrant(index_rows, image_vecs)
        logger.info(f"VLM-Only 완료: {len(index_rows)}건 (run_id={run_id})")
        return

    # ── Step 4b: Full 모드 — Gradio 라벨링 대기 ──────────────────────────
    logger.info("Gradio UI(http://localhost:7860)에서 라벨링 완료 후 자동 재개")
    labeled_map = await asyncio.to_thread(_poll_for_labels, new_ids, label_poll_interval, 48)

    if not labeled_map:
        logger.warning("라벨링 완료 항목 없음. 종료.")
        return

    logger.info(f"라벨링 완료: {len(labeled_map)}건 — 파이프라인 재개")

    # ── Step 5: VLM 캡셔닝 ───────────────────────────────────────────────
    caption_path = f"output/captions_unlabeled_{run_id}.jsonl"
    await run_vlm_caption_batch(MASKED_OUTPUT_DIR, caption_path, concurrency=concurrency)

    # ── Step 6~9: 인덱스 필드 생성 → 분류 → 임베딩 → 적재 ──────────────
    index_rows = _build_rows_from_caption(caption_path, masked_by_id, r2_urls, run_id, labeled_map)
    clf_results = classify_batch([r.get("_masked_path", "") for r in index_rows], model_path=classifier_model)
    _merge_clf(index_rows, clf_results)

    masked_paths = [r.pop("_masked_path", "") for r in index_rows]
    image_vecs   = await asyncio.to_thread(embed_images_siglip, masked_paths)

    upsert_rdb(index_rows, status="complete")
    upsert_qdrant(index_rows, image_vecs)
    logger.info(f"Scenario B 완료: {len(index_rows)}건 (run_id={run_id})")


def _build_rows_from_caption(
    caption_path: str,
    masked_by_id: dict,
    r2_urls: dict,
    run_id: str,
    labeled_map: dict | None,
) -> list[dict]:
    """caption JSONL → index_rows 구성 (Gradio 라벨 또는 VLM-only)"""
    rows = []
    with open(caption_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("error"):
                    continue
                fid = row["file_id"]

                vlm = {
                    "category":      row.get("caption_category", ""),
                    "micro_details": row.get("caption_micro_details", []),
                    "mood_and_tpo":  row.get("mood_and_tpo", []),
                }

                existing = None
                if labeled_map and fid in labeled_map:
                    existing = _manual_to_existing(labeled_map[fid])
                    # 구조화 라벨 필드 보강
                    row["label_color"]    = existing.get("색상")
                    row["label_sub_color"]= existing.get("서브색상")
                    row["label_material"] = existing.get("소재", [])
                    row["label_fit"]      = existing.get("핏")
                    row["label_length"]   = existing.get("기장")
                    row["label_sleeve"]   = existing.get("소매기장")
                    row["label_neckline"] = existing.get("넥라인")
                    row["label_detail"]   = existing.get("디테일", [])
                    row["label_print"]    = existing.get("프린트", [])

                row["dense_caption"]   = build_dense_caption(vlm, existing)
                row["flat_tags"]       = build_flat_tags(vlm, existing)
                row["pipeline_run_id"] = run_id

                masked = masked_by_id.get(fid)
                cat    = masked["category"] if masked else "unknown"
                row["category"]    = cat
                row["image_url"]   = r2_urls.get(f"{fid}_{cat}", "")
                row["_masked_path"]= masked["output_path"] if masked else ""

                rows.append(row)
            except Exception as e:
                print(f"row 구성 실패: {e}")
    return rows


def _merge_clf(index_rows: list[dict], clf_results: list[dict]) -> None:
    """분류기 결과를 index_rows에 in-place 병합"""
    for row, clf in zip(index_rows, clf_results):
        if clf:
            row["pattern_position"] = clf.get("pattern_position", [])
            row["pattern_size"]     = clf.get("pattern_size", "")
            row["trim"]             = clf.get("trim", "")
