# 출처: 신규 작성 — Scenario B Prefect flow
# (라벨 없는 신규 데이터: rembg 자동 마스킹 → Gradio 수동 라벨링 → 파이프라인 재개)
# 수정: ES → Qdrant, E5/FashionCLIP → marqo-fashionSigLIP

from __future__ import annotations

import asyncio
import json
import shutil
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

CLOTHING_TYPE_TO_CATEGORY = {
    "상의": "top", "하의": "bottom", "아우터": "outerwear", "원피스": "dress",
}

_SEP = "═" * 62

def _banner(step: int | str, title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  STEP {step}  |  {title}")
    print(_SEP)

def _fmt_elapsed(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}분 {s:02d}초" if m else f"{s:02d}초"


def _load_manual_labels(file_ids: list[str]) -> dict[str, list[dict]]:
    """
    output/manual_labels.jsonl에서 file_id별 모든 항목 로드.
    다중 항목 지원: 한 이미지에 여러 번 라벨링된 경우 모두 수집.
    반환: {file_id: [item1, item2, ...]}
    """
    if not MANUAL_LABELS_PATH.exists():
        return {}
    target = set(file_ids)
    result: dict[str, list[dict]] = {}
    with open(MANUAL_LABELS_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                fid = row.get("file_id", "")
                if fid in target:
                    result.setdefault(fid, []).append(row)
            except Exception:
                pass
    return result


def _poll_for_labels(
    file_ids: list[str],
    poll_interval: int = 300,
    timeout_hours: int = 48,
) -> dict[str, list[dict]]:
    """Gradio UI 완료 감지 (로컬 파일 폴링, DB 쿼리 없음)"""
    deadline = time.time() + timeout_hours * 3600
    target   = set(file_ids)
    t_start  = time.time()

    print(f"\n  [Gradio 라벨링 대기]  총 {len(target)}건")
    for fid in sorted(target):
        print(f"    ⋯ {fid}")
    print(f"  poll 간격: {poll_interval}초 | timeout: {timeout_hours}h")

    cycle = 0
    while time.time() < deadline:
        cycle += 1
        done      = _load_manual_labels(file_ids)
        remaining = target - set(done.keys())
        elapsed   = _fmt_elapsed(time.time() - t_start)

        print(f"\n  ──── 폴링 #{cycle}  ({elapsed} 경과) ────")
        print(f"  완료 {len(done)}/{len(target)}건  |  대기 {len(remaining)}건")
        for fid in sorted(done.keys()):
            items = done[fid]
            types = [i.get("타입", "?") for i in items]
            cats  = [i.get("카테고리", "") for i in items]
            detail_parts = [f"{t}({c})" if c else t for t, c in zip(types, cats)]
            print(f"    ✔ {fid}  →  {', '.join(detail_parts)}")
        for fid in sorted(remaining):
            print(f"    ⋯ {fid}  (대기 중)")

        if not remaining:
            print(f"\n  ★ 전체 라벨링 완료! ({elapsed} 소요, 항목 {sum(len(v) for v in done.values())}개)")
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

    실행 전 Gradio UI 별도 실행 필요 (full mode):
      python flows/gradio_labeler.py
    """
    logger = get_run_logger()
    run_id = str(uuid.uuid4())[:8]
    t_flow_start = time.time()

    print(f"\n{'★'*62}")
    print(f"  Scenario B  |  mode={mode}  |  run_id={run_id}")
    print(f"{'★'*62}")

    # ── Step 1: 신규 파일 감지 (캐시 파일 기반, DB 쿼리 없음) ───────────
    _banner(1, "신규 이미지 감지 (캐시 기반, DB 쿼리 없음)")
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
    print(f"  전체 {len(all_ids)}건 중 신규 {len(new_ids)}건 처리")
    for fid in new_ids:
        print(f"    ▸ {fid}.jpg")
    logger.info(f"신규 {len(new_ids)}건 처리 시작 (mode={mode}, run_id={run_id})")

    # ── Step 2: rembg 자동 마스킹 ───────────────────────────────────────
    _banner(2, "rembg 자동 배경 제거 마스킹")
    t2 = time.time()
    masked_results = auto_mask_rembg(new_paths, MASKED_OUTPUT_DIR)
    masked_by_id   = {r["file_id"]: r for r in masked_results}
    print(f"  완료: {len(masked_results)}건  ({_fmt_elapsed(time.time()-t2)} 소요)")
    for r in masked_results:
        print(f"    ✔ {r['file_id']}  →  {Path(r['output_path']).name}")

    # ── Step 3: R2 업로드 + pending 상태 RDB 등록 ────────────────────────
    _banner(3, "Cloudflare R2 업로드  +  Turso DB pending_label 등록")
    t3 = time.time()
    r2_manifest = f"output/r2_manifest_unlabeled_{run_id}.txt"
    r2_urls = upload_masked_images(
        local_dir=MASKED_OUTPUT_DIR,
        remote_prefix="masking_data/unlabeled/",
        manifest_path=r2_manifest,
    )
    print(f"  R2 업로드 완료: {len(r2_urls)}건  ({_fmt_elapsed(time.time()-t3)} 소요)")
    for stem, url in r2_urls.items():
        print(f"    ▸ {stem}  →  {url}")

    pending_rows = [
        {
            "file_id":         r["file_id"],
            "category":        r["category"],
            "image_url":       r2_urls.get(Path(r["output_path"]).stem, ""),
            "pipeline_run_id": run_id,
        }
        for r in masked_results
    ]
    print(f"\n  [DB] {len(pending_rows)}건 → status='pending_label' 등록 중...")
    upsert_rdb(pending_rows, status="pending_label")
    print(f"  DB 상태: (없음) → pending_label  ★ Gradio 라벨링 대기 시작")

    # ── Step 4a: VLM-Only 모드 ───────────────────────────────────────────
    if mode == "vlm_only":
        logger.info("VLM-Only 모드: 구조화 라벨 없이 즉시 색인")
        caption_path = f"output/captions_unlabeled_{run_id}.jsonl"
        await run_vlm_caption_batch(MASKED_OUTPUT_DIR, caption_path, concurrency=concurrency)

        index_rows      = _build_rows_from_caption(caption_path, masked_by_id, r2_urls, run_id,
                                                   label_by_key=None)
        is_bottom_flags = [r.get("category") == "bottom" for r in index_rows]
        clf_results     = classify_batch(
            [r.get("_masked_path", "") for r in index_rows],
            model_path=classifier_model,
            bottom_model_path="models/fashion_classifier_bottom.pt",
            is_bottom=is_bottom_flags,
        )
        _merge_clf(index_rows, clf_results)

        image_vecs = await asyncio.to_thread(embed_images_siglip,
                                             [r.pop("_masked_path", "") for r in index_rows])
        upsert_rdb(index_rows, status="vlm_only")
        upsert_qdrant(index_rows, image_vecs)
        logger.info(f"VLM-Only 완료: {len(index_rows)}건 (run_id={run_id})")
        return

    # ── Step 4b: Full 모드 — Gradio 라벨링 대기 ──────────────────────────
    _banner("4b", "Gradio 수동 라벨링 대기  (http://localhost:7860)")
    print("  ★ 브라우저에서 이미지를 라벨링하면 자동으로 파이프라인이 재개됩니다.")
    t4 = time.time()
    labeled_map = await asyncio.to_thread(_poll_for_labels, new_ids, label_poll_interval, 48)

    if not labeled_map:
        logger.warning("라벨링 완료 항목 없음. 종료.")
        return

    labeled_items_total = sum(len(v) for v in labeled_map.values())
    print(f"\n  파이프라인 재개! 라벨링 소요: {_fmt_elapsed(time.time()-t4)}")
    print(f"  이미지 {len(labeled_map)}건 / 항목 {labeled_items_total}개 — 수집된 라벨:")
    for fid, items in sorted(labeled_map.items()):
        for item in items:
            typ  = item.get("타입", "?")
            cat  = item.get("카테고리", "-")
            col  = item.get("색상", "-")
            fit  = item.get("핏", "-")
            mat  = ", ".join(item.get("소재", [])) or "-"
            print(f"    ✔ {fid}  {typ}({cat})  색상={col}  핏={fit}  소재={mat}")
    logger.info(f"라벨링 완료: 이미지 {len(labeled_map)}건, 항목 {labeled_items_total}개 — 파이프라인 재개")

    # ── Step 5: 사용자 마스크 준비 및 R2 재업로드 ────────────────────────
    _banner(5, "사용자 마스크 정리  +  R2 재업로드")
    temp_caption_dir = Path(f"data/temp_caption_{run_id}")
    temp_caption_dir.mkdir(parents=True, exist_ok=True)

    item_tuples: list[tuple[str, dict]] = []  # (dest_stem, label_item)
    for orig_fid, items in labeled_map.items():
        for item in items:
            cat_en    = CLOTHING_TYPE_TO_CATEGORY.get(item.get("타입", ""), "unknown")
            dest_name = f"{orig_fid}_{cat_en}.jpg"
            dest_path = temp_caption_dir / dest_name

            mp = item.get("masked_path", "")
            src = Path(mp) if mp and Path(mp).exists() else None

            if src is None:
                # 사용자 마스크 없음 → rembg 결과 사용
                fallback = masked_by_id.get(orig_fid, {}).get("output_path", "")
                src = Path(fallback) if fallback and Path(fallback).exists() else None

            if src:
                shutil.copy2(src, dest_path)
                item_tuples.append((dest_path.stem, item))
                print(f"    ▸ {src.name}  →  {dest_path.name}")
            else:
                logger.warning(f"마스크 파일 없음 → 스킵: {orig_fid} / {cat_en}")

    if not item_tuples:
        logger.warning("처리 가능한 마스크 없음. 종료.")
        shutil.rmtree(temp_caption_dir, ignore_errors=True)
        return

    # temp dir의 user-drawn 마스크도 R2에 업로드 (동일 remote prefix, manifest 재사용)
    t5r2 = time.time()
    new_r2_urls = upload_masked_images(
        local_dir=str(temp_caption_dir),
        remote_prefix="masking_data/unlabeled/",
        manifest_path=r2_manifest,
    )
    r2_urls.update(new_r2_urls)
    print(f"  R2 재업로드: {len(new_r2_urls)}건  ({_fmt_elapsed(time.time()-t5r2)} 소요)")

    # label_by_key: {stem: label_item} 빠른 조회용
    label_by_key: dict[str, dict] = {stem: item for stem, item in item_tuples}

    # ── Step 6: VLM 캡셔닝 (user-drawn masks only) ────────────────────────
    _banner(6, "VLM 캡셔닝  (marqo-fashionSigLIP → GPT-4o)")
    caption_path = f"output/captions_unlabeled_{run_id}.jsonl"

    label_hints: dict[str, str] = {}
    for orig_fid, items in labeled_map.items():
        for item in items:
            kor_type = item.get("타입", "")
            kor_cat  = item.get("카테고리", "")
            cat_en   = CLOTHING_TYPE_TO_CATEGORY.get(kor_type, "")
            if not cat_en:
                continue
            parts = [f"대분류: {kor_type}"]
            if kor_cat:
                parts.append(f"카테고리: {kor_cat}")
            label_hints[f"{orig_fid}_{cat_en}"] = "\n".join(parts)
    print(f"  label_hints {len(label_hints)}개 항목 전달 (VLM에 사전 정보 제공)")
    for k, v in label_hints.items():
        print(f"    {k}:  {v.replace(chr(10), ' / ')}")

    t6 = time.time()
    await run_vlm_caption_batch(str(temp_caption_dir), caption_path, concurrency=concurrency,
                                label_hints=label_hints or None)
    shutil.rmtree(temp_caption_dir, ignore_errors=True)
    print(f"  VLM 캡셔닝 완료  ({_fmt_elapsed(time.time()-t6)} 소요)  →  {caption_path}")

    # ── Step 7: 인덱스 필드 구성 ──────────────────────────────────────────
    _banner(7, "인덱스 필드 구성  (VLM 출력 + 수동 라벨 병합)")
    index_rows = _build_rows_from_caption(caption_path, masked_by_id, r2_urls, run_id,
                                          label_by_key=label_by_key)
    print(f"  구성된 행: {len(index_rows)}건")
    for r in index_rows:
        fid   = r.get("file_id", "?")
        cat   = r.get("category", "?")
        col   = r.get("label_color", "-")
        cap   = (r.get("dense_caption") or "")[:70]
        tags  = (r.get("flat_tags") or "")[:60]
        print(f"    ▸ {fid}_{cat}  색상={col}")
        print(f"      dense_caption: {cap}...")
        print(f"      flat_tags: {tags}...")

    # ── Step 8: 패턴·핏 분류기 ────────────────────────────────────────────
    _banner(8, "패션 분류기  (pattern / fit / trim)")
    t8 = time.time()
    is_bottom_flags = [r.get("category") == "bottom" for r in index_rows]
    clf_results     = classify_batch(
        [r.get("_masked_path", "") for r in index_rows],
        model_path=classifier_model,
        bottom_model_path="models/fashion_classifier_bottom.pt",
        is_bottom=is_bottom_flags,
    )
    _merge_clf(index_rows, clf_results)
    print(f"  분류 완료  ({_fmt_elapsed(time.time()-t8)} 소요)")
    for r, clf in zip(index_rows, clf_results):
        fid = r.get("file_id", "?")
        cat = r.get("category", "?")
        if clf:
            ps  = clf.get("pattern_size", "-")
            pp  = clf.get("pattern_position", [])
            tr  = clf.get("trim", "-")
            bl  = clf.get("bottom_length", "")
            print(f"    ✔ {fid}_{cat}  패턴크기={ps}  트림={tr}"
                  + (f"  하의기장={bl}" if bl else ""))

    # ── Step 9: 이미지 임베딩 ─────────────────────────────────────────────
    _banner(9, "이미지 임베딩  (marqo-fashionSigLIP  768d)")
    t9 = time.time()
    masked_paths = [r.pop("_masked_path", "") for r in index_rows]
    image_vecs   = await asyncio.to_thread(embed_images_siglip, masked_paths)
    print(f"  임베딩 완료: {len(image_vecs)}건  ({_fmt_elapsed(time.time()-t9)} 소요)")
    for path, vec in zip(masked_paths, image_vecs):
        norm = sum(v*v for v in vec) ** 0.5
        print(f"    ▸ {Path(path).name}  dim={len(vec)}  norm={norm:.4f}")

    # ── Step 10a: Turso DB 적재 (pending_label → complete) ───────────────
    _banner("10a", "Turso DB 적재  pending_label  →  complete")
    print(f"  [{len(index_rows)}건] DB 상태 변경: pending_label → complete")
    t10a = time.time()
    upsert_rdb(index_rows, status="complete")
    print(f"  DB 적재 완료  ({_fmt_elapsed(time.time()-t10a)} 소요)")

    # ── Step 10b: Qdrant 벡터 색인 ───────────────────────────────────────
    _banner("10b", "Qdrant 벡터 색인  (visual 컬렉션)")
    t10b = time.time()
    upsert_qdrant(index_rows, image_vecs)
    print(f"  Qdrant 색인 완료  ({_fmt_elapsed(time.time()-t10b)} 소요)")

    # ── 최종 요약 ─────────────────────────────────────────────────────────
    elapsed_total = _fmt_elapsed(time.time() - t_flow_start)
    print(f"\n{'★'*62}")
    print(f"  Scenario B 완료!  {len(index_rows)}건  /  총 소요: {elapsed_total}")
    print(f"{'★'*62}")
    print(f"  {'file_id':<14} {'category':<12} {'status':<14} {'color':<12} {'pattern_size'}")
    print(f"  {'-'*58}")
    for r in index_rows:
        fid = r.get("file_id", "?")
        cat = r.get("category", "?")
        col = r.get("label_color", "-")
        ps  = r.get("pattern_size", "-")
        print(f"  {fid:<14} {cat:<12} {'complete':<14} {col:<12} {ps}")
    print()
    logger.info(f"Scenario B 완료: {len(index_rows)}건 (run_id={run_id})")


def _build_rows_from_caption(
    caption_path: str,
    masked_by_id: dict,
    r2_urls: dict,
    run_id: str,
    label_by_key: dict[str, dict] | None,
) -> list[dict]:
    """
    caption JSONL → index_rows 구성
    label_by_key: {"{file_id}_{category}": label_item} — full mode
                  None → VLM-only mode
    파일명 규칙: {file_id}_{category}.jpg → file_id, category 파싱
    """
    rows = []
    with open(caption_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("error"):
                    continue
                fid      = row["file_id"]       # e.g., "1028690"
                category = row.get("category", "unknown")  # e.g., "top"
                stem     = f"{fid}_{category}"  # e.g., "1028690_top"

                vlm = {
                    "category":      row.get("caption_category", ""),
                    "micro_details": row.get("caption_micro_details", []),
                    "mood_and_tpo":  row.get("mood_and_tpo", []),
                }

                existing = None
                if label_by_key is not None:
                    label_item = label_by_key.get(stem)
                    if label_item:
                        existing                  = _manual_to_existing(label_item)
                        row["label_color"]        = existing.get("색상")
                        row["label_sub_color"]    = existing.get("서브색상")
                        row["label_material"]     = existing.get("소재", [])
                        row["label_fit"]          = existing.get("핏")
                        row["label_length"]       = existing.get("기장")
                        row["label_sleeve"]       = existing.get("소매기장")
                        row["label_neckline"]     = existing.get("넥라인")
                        row["label_detail"]       = existing.get("디테일", [])
                        row["label_print"]        = existing.get("프린트", [])
                        masked_path               = label_item.get("masked_path") or \
                                                    masked_by_id.get(fid, {}).get("output_path", "")
                    else:
                        masked_path = masked_by_id.get(fid, {}).get("output_path", "")
                else:
                    # VLM-only: use rembg masked path
                    masked_path = masked_by_id.get(fid, {}).get("output_path", "")

                row["dense_caption"]   = build_dense_caption(vlm, existing)
                row["flat_tags"]       = build_flat_tags(vlm, existing)
                row["pipeline_run_id"] = run_id
                row["category"]        = category
                row["image_url"]       = r2_urls.get(stem, r2_urls.get(f"{fid}_unknown", ""))
                row["_masked_path"]    = masked_path

                rows.append(row)
            except Exception as e:
                print(f"row 구성 실패: {e}")
    return rows


def _merge_clf(index_rows: list[dict], clf_results: list[dict]) -> None:
    """분류기 결과를 index_rows에 in-place 병합"""
    for row, clf in zip(index_rows, clf_results):
        if clf:
            row["pattern_position"]  = clf.get("pattern_position", [])
            row["pattern_size"]      = clf.get("pattern_size", "")
            row["trim"]              = clf.get("trim", "")
            row["bottom_length"]     = clf.get("bottom_length", "")
            row["bottom_waist_rise"] = clf.get("bottom_waist_rise", "")
