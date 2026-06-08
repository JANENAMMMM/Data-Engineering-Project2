# 출처:
#   load_cache, append_to_cache, build_cache_from_db:
#     SeoJimin1234/DE_pj2_Storage/db_ingest.py 이식
#   배치커밋(100건) + threading.Lock + auto-reconnect:
#     SeoJimin1234/DE_pj2_Storage/db_ingest.py 이식
#   fashion_items 테이블 스키마: 신규 설계

from __future__ import annotations

import json
import threading
from pathlib import Path

from prefect import task

from src.config import DB_URL, DB_ACCESS_TOKEN

CACHE_FILE = Path("output/indexed_ids.txt")
BATCH_SIZE = 100     # 출처: SeoJimin1234/DE_pj2_Storage/db_ingest.py BATCH_SIZE=100
db_lock    = threading.Lock()  # 출처: DE_pj2_Storage threading.Lock 패턴

# ── Turso 과금 방지 원칙 ────────────────────────────────────────────────────
# 중복 확인은 반드시 로컬 캐시 파일(indexed_ids.txt)만 사용.
# DB SELECT는 build_cache_from_db() 한 번만 허용 (캐시 파일 없을 때만).
# 루프 안에서 DB SELECT 절대 금지.
# ─────────────────────────────────────────────────────────────────────────────


# ── 캐시 관리 (출처: SeoJimin1234/DE_pj2_Storage/db_ingest.py) ─────────────

def load_cache() -> set[str]:
    """
    출처: SeoJimin1234/DE_pj2_Storage/db_ingest.py — load_cache()
    로컬 파일만 읽음. DB 쿼리 없음 → Turso 과금 없음.
    """
    if not CACHE_FILE.exists():
        return set()
    return {l.strip() for l in CACHE_FILE.read_text(encoding="utf-8").splitlines() if l.strip()}


def append_to_cache(new_ids: list[str]) -> None:
    """출처: SeoJimin1234/DE_pj2_Storage/db_ingest.py — append_to_cache()"""
    if not new_ids:
        return
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(new_ids) + "\n")


def build_cache_from_db() -> set[str]:
    """
    출처: SeoJimin1234/DE_pj2_Storage/db_ingest.py — build_cache_from_db()
    SELECT 1회로 전체 file_id를 가져와 로컬 파일에 저장.
    이후 중복 확인은 로컬 파일만 사용 → DB read 요금 최소화.
    캐시 파일이 없을 때 1회만 호출할 것.
    """
    from libsql import connect
    print("DB에서 캐시 구성 중... (SELECT 1회 실행)")
    conn = connect(DB_URL, auth_token=DB_ACCESS_TOKEN, _uri=True)
    # SELECT file_id 1회 = 저장된 행 수만큼 row read 과금
    rows = conn.execute("SELECT file_id FROM fashion_items").fetchall()
    ids  = {r[0] for r in rows}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text("\n".join(sorted(ids)), encoding="utf-8")
    print(f"캐시 파일 저장 완료: {len(ids)}개 → {CACHE_FILE}")
    return ids


# ── DB 연결 ─────────────────────────────────────────────────────────────────

def get_conn():
    """출처: SeoJimin1234/DE_pj2_Storage/db_ingest.py — get_connection() 패턴 이식"""
    from libsql import connect
    return connect(DB_URL, auth_token=DB_ACCESS_TOKEN, _uri=True)


def ensure_table() -> None:
    """출처: 신규 작성 — fashion_items 테이블 DDL"""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fashion_items (
            file_id               TEXT PRIMARY KEY,
            category              TEXT NOT NULL,
            image_url             TEXT,
            label_color           TEXT,
            label_sub_color       TEXT,
            label_material        TEXT,
            label_fit             TEXT,
            label_length          TEXT,
            label_sleeve          TEXT,
            label_neckline        TEXT,
            label_detail          TEXT,
            label_print           TEXT,
            caption_category      TEXT,
            caption_micro_details TEXT,
            mood_and_tpo          TEXT,
            dense_caption         TEXT,
            flat_tags             TEXT,
            pattern_position      TEXT,
            pattern_size          TEXT,
            trim                  TEXT,
            bottom_length         TEXT,
            bottom_waist_rise     TEXT,
            status                TEXT DEFAULT 'complete',
            pipeline_run_id       TEXT,
            indexed_at            DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at            DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_status   ON fashion_items(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON fashion_items(category)")
    # 기존 테이블에 컬럼이 없으면 추가 (마이그레이션)
    for col, typedef in [("bottom_length", "TEXT"), ("bottom_waist_rise", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE fashion_items ADD COLUMN {col} {typedef}")
        except Exception:
            pass  # 이미 존재하면 무시
    conn.commit()
    print("fashion_items 테이블 준비 완료")


# ── Prefect Tasks ────────────────────────────────────────────────────────────

@task(name="filter-already-indexed")
def filter_already_indexed(file_ids: list[str]) -> list[str]:
    """
    출처: SeoJimin1234/DE_pj2_Storage/db_ingest.py — 캐시 기반 중복 제거 패턴
    DB SELECT 절대 없음. 로컬 캐시 파일만 읽음.
    캐시 파일 없으면 build_cache_from_db() 1회 호출 후 이후 파일만 사용.
    """
    if not CACHE_FILE.exists():
        # 캐시 파일 없을 때만 DB SELECT 1회 허용
        print(f"캐시 파일 없음 → DB에서 1회 구성 (이후 DB 조회 없음)")
        build_cache_from_db()

    cached  = load_cache()  # 로컬 파일 read만
    new_ids = [fid for fid in file_ids if fid not in cached]
    print(f"중복 제거: {len(file_ids)}개 → {len(new_ids)}개 신규 (캐시 {len(cached)}개)")
    return new_ids


@task(name="upsert-rdb", retries=2, retry_delay_seconds=30)
def upsert_rdb(rows: list[dict], status: str = "complete") -> None:
    """
    출처:
      배치커밋(100건) + threading.Lock:
        SeoJimin1234/DE_pj2_Storage/db_ingest.py 이식
      auto-reconnect ("stream not found"):
        SeoJimin1234/DE_pj2_Storage/db_ingest.py run() 패턴 이식
      INSERT OR REPLACE + 스키마:
        신규 설계
    """
    conn      = get_conn()
    cur       = conn.cursor()
    batch_ids: list[str] = []
    errors    = 0

    for i, row in enumerate(rows):
        try:
            with db_lock:
                cur.execute("""
                    INSERT OR REPLACE INTO fashion_items (
                        file_id, category, image_url,
                        label_color, label_sub_color, label_material,
                        label_fit, label_length, label_sleeve, label_neckline,
                        label_detail, label_print,
                        caption_category, caption_micro_details, mood_and_tpo,
                        dense_caption, flat_tags,
                        pattern_position, pattern_size, trim,
                        bottom_length, bottom_waist_rise,
                        status, pipeline_run_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    row.get("file_id"),
                    row.get("category", ""),
                    row.get("image_url"),
                    row.get("label_color"),
                    row.get("label_sub_color"),
                    json.dumps(row.get("label_material") or [],  ensure_ascii=False),
                    row.get("label_fit"),
                    row.get("label_length"),
                    row.get("label_sleeve"),
                    row.get("label_neckline"),
                    json.dumps(row.get("label_detail") or [],    ensure_ascii=False),
                    json.dumps(row.get("label_print") or [],     ensure_ascii=False),
                    row.get("caption_category"),
                    json.dumps(row.get("caption_micro_details") or [], ensure_ascii=False),
                    json.dumps(row.get("mood_and_tpo") or [],          ensure_ascii=False),
                    row.get("dense_caption"),
                    row.get("flat_tags"),
                    json.dumps(row.get("pattern_position") or [], ensure_ascii=False),
                    row.get("pattern_size"),
                    row.get("trim"),
                    row.get("bottom_length") or None,
                    row.get("bottom_waist_rise") or None,
                    status,
                    row.get("pipeline_run_id"),
                ))
                batch_ids.append(str(row["file_id"]))

            # 100건마다 커밋 (출처: DE_pj2_Storage BATCH_SIZE=100)
            if (i + 1) % BATCH_SIZE == 0:
                conn.commit()
                append_to_cache(batch_ids)
                print(f"  DB 커밋: {i + 1}/{len(rows)}")
                batch_ids = []

        except Exception as e:
            err_str = str(e).lower()
            # auto-reconnect (출처: DE_pj2_Storage "stream not found" 패턴)
            if "stream not found" in err_str:
                print("  Turso 연결 재시도...")
                conn = get_conn()
                cur  = conn.cursor()
            else:
                print(f"  DB 오류 [{row.get('file_id')}]: {e}")
                errors += 1

    if batch_ids:
        conn.commit()
        append_to_cache(batch_ids)

    print(f"RDB 적재 완료: {len(rows) - errors}건 성공, {errors}건 실패")
