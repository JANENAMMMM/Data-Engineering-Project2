# 출처: 현재 프로젝트 src/caption.py → Prefect @task 래핑
# 래핑 구조: 신규 작성

from __future__ import annotations

import json
from pathlib import Path

from prefect import task

from src.caption import (
    batch_from_dir_async,
    build_dense_caption,
    build_flat_tags,
    _parse_item_info,
)
from src.label_parser import _parse_json_file


@task(name="vlm-caption-batch", retries=1, retry_delay_seconds=60)
async def run_vlm_caption_batch(
    images_dir: str,
    out_path: str,
    concurrency: int = 50,
) -> str:
    """
    출처: 현재 프로젝트 src/caption.py → batch_from_dir_async() Prefect 래핑
    반환: out_path (완료된 JSONL 경로)
    주의: VLM 배치는 수 시간 소요 → Claude Code 내부 실행 금지, 외부 터미널 권장
    """
    await batch_from_dir_async(images_dir, out_path, concurrency=concurrency)
    return out_path


@task(name="build-index-fields")
def build_index_fields_batch(
    caption_jsonl_path: str,
    label_dir: str | None = None,
) -> list[dict]:
    """
    출처:
      - 현재 프로젝트 src/caption.py → build_dense_caption, build_flat_tags, _parse_item_info
      - 현재 프로젝트 src/label_parser.py → _parse_json_file
    caption JSONL + (선택) 라벨 dir → dense_caption, flat_tags 추가된 row 목록 반환
    """
    label_index: dict[str, list] = {}
    if label_dir:
        for json_path in Path(label_dir).glob("**/*.json"):
            try:
                file_id, entry = _parse_json_file(json_path)
                label_index[str(file_id)] = entry.get("items", [])
            except Exception:
                pass

    rows: list[dict] = []
    with open(caption_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("error"):
                    continue

                vlm = {
                    "category":      row.get("caption_category", ""),
                    "micro_details": row.get("caption_micro_details", []),
                    "mood_and_tpo":  row.get("mood_and_tpo", []),
                }

                existing = None
                items = label_index.get(str(row["file_id"]), [])
                if items:
                    existing = _parse_item_info(items[0].get("item_info", ""))

                row["dense_caption"] = build_dense_caption(vlm, existing)
                row["flat_tags"]     = build_flat_tags(vlm, existing)

                # 구조화 라벨 필드 보강
                if existing:
                    row.setdefault("label_color",    existing.get("색상"))
                    row.setdefault("label_sub_color", existing.get("서브색상"))
                    row.setdefault("label_material",  existing.get("소재", []))
                    row.setdefault("label_fit",       existing.get("핏"))
                    row.setdefault("label_length",    existing.get("기장"))
                    row.setdefault("label_sleeve",    existing.get("소매기장"))
                    row.setdefault("label_neckline",  existing.get("넥라인"))
                    row.setdefault("label_detail",    existing.get("디테일", []))
                    row.setdefault("label_print",     existing.get("프린트", []))

                rows.append(row)
            except Exception as e:
                print(f"인덱스 필드 생성 실패: {e} | {line[:60]}")

    print(f"인덱스 필드 생성 완료: {len(rows)}건")
    return rows
