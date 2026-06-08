# 출처: 신규 작성
# Elasticsearch 대체 — Qdrant 클라우드 프리티어 사용
# 컬렉션: "visual" (768d COSINE) — marqo-fashionSigLIP 이미지 벡터
# 동일 임베딩 공간이므로 텍스트 쿼리도 이미지 컬렉션에서 크로스 모달 검색 가능
# 참고: 사용자 제공 Qdrant 코드 기반

from __future__ import annotations

import os
from prefect import task
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
)

VISUAL_COLLECTION = "visual"
VECTOR_DIM        = 768      # marqo-fashionSigLIP 출력 차원

# 카테고리 → 점 ID 오프셋 (결정적 고유 ID 생성용)
_CAT_OFFSET = {"top": 0, "bottom": 1, "outerwear": 2, "dress": 3, "unknown": 4}


def _make_point_id(file_id: str, category: str) -> int:
    """file_id + category → 결정적 uint64 ID (충돌 위험 사실상 없음)"""
    try:
        base = int(file_id)
    except ValueError:
        base = abs(hash(file_id)) % 10_000_000
    return base * 10 + _CAT_OFFSET.get(category, 5)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


def ensure_qdrant_collection() -> None:
    """
    출처: 신규 작성 (사용자 제공 컬렉션 생성 코드 참고)
    visual 컬렉션 없으면 생성 + flat_tags 전문 검색 인덱스 추가
    """
    client     = get_qdrant_client()
    existing   = {c.name for c in client.get_collections().collections}

    if VISUAL_COLLECTION not in existing:
        client.create_collection(
            collection_name=VISUAL_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Qdrant 컬렉션 생성: {VISUAL_COLLECTION} (dim={VECTOR_DIM})")

        # flat_tags 필드에 전문 검색 인덱스 (키워드 검색 보조)
        client.create_payload_index(
            collection_name=VISUAL_COLLECTION,
            field_name="flat_tags",
            field_schema=TextIndexParams(
                type=PayloadSchemaType.TEXT,
                tokenizer=TokenizerType.WHITESPACE,
            ),
        )
        print("flat_tags 전문 인덱스 생성 완료")
    else:
        print(f"Qdrant 컬렉션 이미 존재: {VISUAL_COLLECTION}")


@task(name="upsert-qdrant", retries=2, retry_delay_seconds=30)
def upsert_qdrant(
    rows: list[dict],
    image_vectors: list[list[float]],
    batch_size: int = 1000,
) -> int:
    """
    출처: 신규 작성 (사용자 제공 Qdrant 저장 코드 기반)
    rows: build_index_fields_batch() + classify_batch() 병합 결과
    image_vectors: embed_images_siglip() 출력 (768d)
    1000건씩 배치 upsert (사용자 제공 코드 동일 배치 크기)
    반환: 성공 건수
    """
    ensure_qdrant_collection()
    client = get_qdrant_client()

    points: list[PointStruct] = []
    total_uploaded = 0

    for i, (row, vec) in enumerate(zip(rows, image_vectors)):
        file_id  = str(row["file_id"])
        category = row.get("category", "unknown")

        point = PointStruct(
            id=_make_point_id(file_id, category),
            vector=vec,
            payload={
                "file_id":               file_id,
                "category":              category,
                "image_url":             row.get("image_url", ""),
                "flat_tags":             row.get("flat_tags", ""),
                "dense_caption":         row.get("dense_caption", ""),
                "caption_category":      row.get("caption_category", ""),
                "caption_micro_details": row.get("caption_micro_details", []),
                "mood_and_tpo":          row.get("mood_and_tpo", []),
                "pattern_position":      row.get("pattern_position", []),
                "pattern_size":          row.get("pattern_size", ""),
                "trim":                  row.get("trim", ""),
                "bottom_length":         row.get("bottom_length", "") or "",
                "bottom_waist_rise":     row.get("bottom_waist_rise", "") or "",
                # 구조화 라벨
                "label_color":     row.get("label_color"),
                "label_sub_color": row.get("label_sub_color"),
                "label_material":  row.get("label_material", []),
                "label_fit":       row.get("label_fit"),
                "label_length":    row.get("label_length"),
                "label_sleeve":    row.get("label_sleeve"),
                "label_neckline":  row.get("label_neckline"),
                "label_detail":    row.get("label_detail", []),
                "label_print":     row.get("label_print", []),
            },
        )
        points.append(point)

        # 1000건마다 배치 upsert (사용자 제공 코드 동일 배치 크기)
        if len(points) == batch_size:
            client.upsert(collection_name=VISUAL_COLLECTION, points=points)
            total_uploaded += len(points)
            print(f"  Qdrant upsert: {total_uploaded}/{len(rows)}")
            points = []

    if points:
        client.upsert(collection_name=VISUAL_COLLECTION, points=points)
        total_uploaded += len(points)

    print(f"Qdrant 적재 완료: {total_uploaded}건")
    return total_uploaded
