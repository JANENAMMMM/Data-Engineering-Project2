"""
전체 파이프라인 통합 테스트 — test_data/ 의 8개 이미지 사용
Prefect 서버 없이 asyncio.run()으로 직접 실행

실행:
  python scripts/test_pipeline.py

단계:
  1. 폴리곤 마스킹 (src/masking.py)
  2. VLM 캡셔닝 (src/caption.py)
  3. 인덱스 필드 생성 (dense_caption, flat_tags)
  4. 이미지 임베딩 (marqo-fashionSigLIP 768d)
  5. 분류기 추론 (모델 없으면 빈 결과)
  6. Turso RDB 적재
  7. Qdrant 적재
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
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
# K_fashion 이미지 sample 폴더 (라벨링데이터/{스타일}/, 원천데이터/원천데이터_1/{스타일}/)
SAMPLE_ROOT  = LAB / "K_fashion 이미지 sample"
LABEL_ROOT   = SAMPLE_ROOT / "라벨링데이터"
IMAGE_ROOT   = SAMPLE_ROOT / "원천데이터" / "원천데이터_1"
OUTPUT_DIR   = PIPELINE / "output" / "test_run_sample"
MASKED_DIR   = OUTPUT_DIR / "masked"
CAPTION_PATH = str(OUTPUT_DIR / "captions_test.jsonl")
MODEL_PATH        = str(PIPELINE / "models" / "fashion_classifier_v5.pt")
BOTTOM_MODEL_PATH = str(PIPELINE / "models" / "fashion_classifier_bottom.pt")
TEST_LIMIT   = 10   # 스타일별로 골고루 10개

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASKED_DIR.mkdir(parents=True, exist_ok=True)

PASS = "[OK]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results: dict[str, str] = {}


# ════════════════════════════════════════════════════════════════════════════════
# Step 1: 폴리곤 마스킹
# ════════════════════════════════════════════════════════════════════════════════
def _collect_sample_pairs(limit: int) -> list[tuple[Path, Path]]:
    """
    K_fashion 이미지 sample 폴더에서 (json_path, img_path) 쌍 수집.
    스타일 폴더별 균등 샘플링.
    """
    pairs: list[tuple[Path, Path]] = []
    style_dirs = sorted(LABEL_ROOT.iterdir())
    per_style  = max(1, limit // len(style_dirs))

    for style_dir in style_dirs:
        style_name = style_dir.name
        img_dir    = IMAGE_ROOT / style_name
        json_files = sorted(style_dir.glob("*.json"))

        count = 0
        for json_path in json_files:
            if count >= per_style:
                break
            img_path = img_dir / f"{json_path.stem}.jpg"
            if not img_path.exists():
                continue
            pairs.append((json_path, img_path))
            count += 1

    return pairs[:limit]


def step1_masking() -> list[dict]:
    print("\n─── Step 1: 폴리곤 마스킹 ───")
    from src.masking import load_masked_items

    KOR_TO_EN = {"상의": "top", "하의": "bottom", "아우터": "outerwear", "원피스": "dress"}
    masked_results = []
    MASKED_DIR.mkdir(parents=True, exist_ok=True)

    pairs = _collect_sample_pairs(TEST_LIMIT)
    print(f"  수집된 (이미지, 라벨) 쌍: {len(pairs)}개")

    for json_path, img_path in pairs:
        try:
            items = load_masked_items(json_path, img_path)
            for cat_kor, _idx, masked_img in items:
                cat_en   = KOR_TO_EN.get(cat_kor, cat_kor)
                file_id  = json_path.stem
                out_name = f"{file_id}_{cat_en}.jpg"
                out_path = MASKED_DIR / out_name
                masked_img.save(out_path, "JPEG", quality=95)
                masked_results.append({
                    "file_id":     file_id,
                    "category":    cat_en,
                    "output_path": str(out_path),
                })
                print(f"  {PASS} {out_name}")
        except Exception as e:
            print(f"  {FAIL} {json_path.name}: {e}")

    results["step1_masking"] = f"{PASS} {len(masked_results)}개 마스킹 완료"
    return masked_results


# ════════════════════════════════════════════════════════════════════════════════
# Step 2: VLM 캡셔닝
# ════════════════════════════════════════════════════════════════════════════════
async def step2_captioning() -> list[dict]:
    print("\n─── Step 2: VLM 캡셔닝 (Gemini) ───")
    from src.caption import batch_from_dir_async

    if Path(CAPTION_PATH).exists():
        print(f"  체크포인트 발견 → 재사용: {CAPTION_PATH}")
    else:
        await batch_from_dir_async(
            str(MASKED_DIR),
            CAPTION_PATH,
            limit=TEST_LIMIT,
            concurrency=5,
        )

    captions = []
    with open(CAPTION_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if not row.get("error"):
                    captions.append(row)
            except Exception:
                pass

    ok = len(captions)
    results["step2_captioning"] = f"{PASS if ok else FAIL} {ok}개 캡션 생성"
    print(f"  → {ok}개 성공")
    if captions:
        sample = captions[0]
        print(f"  샘플: {sample['file_id']} | {sample.get('caption_category')} | {sample.get('mood_and_tpo')}")
    return captions


# ════════════════════════════════════════════════════════════════════════════════
# Step 3: 인덱스 필드 생성
# ════════════════════════════════════════════════════════════════════════════════
def step3_index_fields(captions: list[dict]) -> list[dict]:
    print("\n─── Step 3: 인덱스 필드 생성 (dense_caption, flat_tags) ───")
    from src.caption import build_dense_caption, build_flat_tags, _parse_item_info
    from src.label_parser import _parse_json_file

    label_index: dict[str, list] = {}
    for json_path in LABEL_ROOT.glob("**/*.json"):
        try:
            file_id, entry = _parse_json_file(json_path)
            label_index[str(file_id)] = entry.get("items", [])
        except Exception:
            pass

    index_rows = []
    for row in captions:
        vlm = {
            "category":      row.get("caption_category", ""),
            "micro_details": row.get("caption_micro_details", []),
            "mood_and_tpo":  row.get("mood_and_tpo", []),
        }
        items    = label_index.get(str(row["file_id"]), [])
        existing = _parse_item_info(items[0].get("item_info", "")) if items else None

        row["dense_caption"] = build_dense_caption(vlm, existing)
        row["flat_tags"]     = build_flat_tags(vlm, existing)
        row["pipeline_run_id"] = "test_run"

        if existing:
            row.update({
                "label_color":    existing.get("색상"),
                "label_sub_color":existing.get("서브색상"),
                "label_material": existing.get("소재", []),
                "label_fit":      existing.get("핏"),
                "label_length":   existing.get("기장"),
                "label_sleeve":   existing.get("소매기장"),
                "label_neckline": existing.get("넥라인"),
                "label_detail":   existing.get("디테일", []),
                "label_print":    existing.get("프린트", []),
            })

        index_rows.append(row)
        print(f"  {PASS} {row['file_id']} | flat_tags: {row['flat_tags'][:60]}...")

    results["step3_index_fields"] = f"{PASS} {len(index_rows)}개 인덱스 필드 생성"
    return index_rows


# ════════════════════════════════════════════════════════════════════════════════
# Step 4: 이미지 임베딩 (marqo-fashionSigLIP)
# ════════════════════════════════════════════════════════════════════════════════
def step4_embedding(masked_results: list[dict], index_rows: list[dict]) -> tuple[list, list]:
    print("\n─── Step 4: 이미지 임베딩 (marqo-fashionSigLIP 768d) ───")
    try:
        import torch
        import open_clip
        from PIL import Image

        print("  모델 로드 중... (최초 실행 시 다운로드 수 분 소요)")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)
        print(f"  디바이스: {device}")

        # 마스킹 결과와 index_rows 매핑
        masked_by_key = {(r["file_id"], r["category"]): r["output_path"] for r in masked_results}

        image_vecs = []
        valid_rows = []
        for row in index_rows:
            key  = (row["file_id"], row.get("category", ""))
            path = masked_by_key.get(key)
            if not path or not Path(path).exists():
                continue
            try:
                img    = Image.open(path).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    vec = model.encode_image(tensor, normalize=True)
                image_vecs.append(vec.cpu().squeeze().tolist())
                valid_rows.append(row)
                print(f"  {PASS} {row['file_id']}_{row.get('category')} → dim={len(image_vecs[-1])}")
            except Exception as e:
                print(f"  {FAIL} {row['file_id']}: {e}")

        results["step4_embedding"] = f"{PASS} {len(image_vecs)}개 이미지 임베딩 완료 (768d)"
        return valid_rows, image_vecs

    except ImportError as e:
        print(f"  {SKIP} 패키지 미설치: {e}")
        results["step4_embedding"] = f"{SKIP} 패키지 미설치 → 건너뜀"
        return index_rows, []


# ════════════════════════════════════════════════════════════════════════════════
# Step 5: 분류기 추론
# ════════════════════════════════════════════════════════════════════════════════
def step5_classifier(masked_results: list[dict], index_rows: list[dict]) -> list[dict]:
    print("\n─── Step 5: 분류기 추론 (패턴위치/크기/끝단마감 + 하의 전용) ───")
    if not Path(MODEL_PATH).exists():
        print(f"  {SKIP} 모델 파일 없음: {MODEL_PATH}")
        print(f"       (학습된 모델 필요. 빈 결과로 계속 진행)")
        results["step5_classifier"] = f"{SKIP} 모델 파일 없음"
        return index_rows

    try:
        from src.classifier.inference import FashionClassifier
        from PIL import Image

        bmp = BOTTOM_MODEL_PATH if Path(BOTTOM_MODEL_PATH).exists() else None
        if not bmp:
            print(f"  [경고] bottom 모델 없음: {BOTTOM_MODEL_PATH} → 하의 추론 비활성")

        masked_by_key = {(r["file_id"], r["category"]): r["output_path"] for r in masked_results}
        clf    = FashionClassifier(MODEL_PATH, bottom_model_path=bmp)
        images, is_bottom_flags, rows = [], [], []
        for row in index_rows:
            path = masked_by_key.get((row["file_id"], row.get("category", "")))
            if path and Path(path).exists():
                images.append(Image.open(path).convert("RGB"))
                is_bottom_flags.append(row.get("category") == "bottom")
                rows.append(row)

        clf_results = clf.predict_batch(images, is_bottom=is_bottom_flags)
        for row, clf_res in zip(rows, clf_results):
            if clf_res:
                row["pattern_position"]  = clf_res.get("pattern_position", [])
                row["pattern_size"]      = clf_res.get("pattern_size", "")
                row["trim"]              = clf_res.get("trim", "")
                row["bottom_length"]     = clf_res.get("bottom_length", "")
                row["bottom_waist_rise"] = clf_res.get("bottom_waist_rise", "")
                extra = ""
                if row.get("category") == "bottom":
                    extra = f" len={row['bottom_length']} waist={row['bottom_waist_rise']}"
                print(f"  {PASS} {row['file_id']}: pp={row['pattern_position']} ps={row['pattern_size']} trim={row['trim']}{extra}")

        results["step5_classifier"] = f"{PASS} {len(rows)}개 분류 완료"
    except Exception as e:
        print(f"  {FAIL} 분류기 오류: {e}")
        results["step5_classifier"] = f"{FAIL} 오류: {e}"

    return index_rows


# ════════════════════════════════════════════════════════════════════════════════
# Step 6: Turso RDB 적재
# ════════════════════════════════════════════════════════════════════════════════
def step6_rdb(index_rows: list[dict]) -> None:
    print("\n─── Step 6: Turso RDB 적재 ───")
    try:
        from libsql import connect
        from src.config import DB_URL, DB_ACCESS_TOKEN
        import json as _json

        conn = connect(DB_URL, auth_token=DB_ACCESS_TOKEN, _uri=True)

        # 테이블 생성
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fashion_items (
                file_id TEXT PRIMARY KEY, category TEXT NOT NULL,
                image_url TEXT, label_color TEXT, label_sub_color TEXT,
                label_material TEXT, label_fit TEXT, label_length TEXT,
                label_sleeve TEXT, label_neckline TEXT, label_detail TEXT,
                label_print TEXT, caption_category TEXT,
                caption_micro_details TEXT, mood_and_tpo TEXT,
                dense_caption TEXT, flat_tags TEXT,
                pattern_position TEXT, pattern_size TEXT, trim TEXT,
                status TEXT DEFAULT 'complete', pipeline_run_id TEXT,
                indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        ok = 0
        for row in index_rows:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO fashion_items (
                        file_id, category, image_url,
                        label_color, label_sub_color, label_material,
                        label_fit, label_length, label_sleeve, label_neckline,
                        label_detail, label_print,
                        caption_category, caption_micro_details, mood_and_tpo,
                        dense_caption, flat_tags,
                        pattern_position, pattern_size, trim,
                        status, pipeline_run_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    row.get("file_id"), row.get("category", ""), row.get("image_url"),
                    row.get("label_color"), row.get("label_sub_color"),
                    _json.dumps(row.get("label_material") or [], ensure_ascii=False),
                    row.get("label_fit"), row.get("label_length"),
                    row.get("label_sleeve"), row.get("label_neckline"),
                    _json.dumps(row.get("label_detail") or [], ensure_ascii=False),
                    _json.dumps(row.get("label_print") or [], ensure_ascii=False),
                    row.get("caption_category"),
                    _json.dumps(row.get("caption_micro_details") or [], ensure_ascii=False),
                    _json.dumps(row.get("mood_and_tpo") or [], ensure_ascii=False),
                    row.get("dense_caption"), row.get("flat_tags"),
                    _json.dumps(row.get("pattern_position") or [], ensure_ascii=False),
                    row.get("pattern_size"), row.get("trim"),
                    "test_complete", row.get("pipeline_run_id"),
                ))
                ok += 1
            except Exception as e:
                print(f"  {FAIL} {row.get('file_id')}: {e}")

        conn.commit()
        conn.close()
        print(f"  {PASS} {ok}건 Turso 저장 완료")
        results["step6_rdb"] = f"{PASS} {ok}건 저장"

    except Exception as e:
        print(f"  {FAIL} Turso 연결 실패: {e}")
        results["step6_rdb"] = f"{FAIL} {e}"


# ════════════════════════════════════════════════════════════════════════════════
# Step 7: Qdrant 적재
# ════════════════════════════════════════════════════════════════════════════════
def step7_qdrant(index_rows: list[dict], image_vecs: list[list[float]]) -> None:
    print("\n─── Step 7: Qdrant 적재 ───")
    if not image_vecs:
        print(f"  {SKIP} 임베딩 없음 (Step 4 실패 또는 건너뜀)")
        results["step7_qdrant"] = f"{SKIP} 임베딩 없음"
        return

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        import os

        client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )

        # 컬렉션 생성 (없으면)
        existing = {c.name for c in client.get_collections().collections}
        if "visual" not in existing:
            client.create_collection(
                collection_name="visual",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            print("  컬렉션 'visual' 생성")

        _CAT_OFFSET = {"top": 0, "bottom": 1, "outerwear": 2, "dress": 3, "unknown": 4}
        points = []
        for row, vec in zip(index_rows, image_vecs):
            fid = row["file_id"]
            cat = row.get("category", "unknown")
            try:
                base    = int(fid)
            except ValueError:
                base    = abs(hash(fid)) % 10_000_000
            point_id = base * 10 + _CAT_OFFSET.get(cat, 5)

            points.append(PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "file_id":          fid,
                    "category":         cat,
                    "flat_tags":        row.get("flat_tags", ""),
                    "dense_caption":    row.get("dense_caption", ""),
                    "caption_category": row.get("caption_category", ""),
                    "mood_and_tpo":     row.get("mood_and_tpo", []),
                    "image_url":        row.get("image_url", ""),
                },
            ))

        client.upsert(collection_name="visual", points=points)
        count = client.get_collection("visual").points_count
        print(f"  {PASS} {len(points)}건 업로드 완료 (컬렉션 총 {count}개)")
        results["step7_qdrant"] = f"{PASS} {len(points)}건 업로드"

    except ImportError:
        print(f"  {SKIP} qdrant-client 미설치")
        results["step7_qdrant"] = f"{SKIP} qdrant-client 미설치"
    except Exception as e:
        print(f"  {FAIL} Qdrant 오류: {e}")
        results["step7_qdrant"] = f"{FAIL} {e}"


# ════════════════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════════════════
async def main():
    t0 = time.time()
    total_imgs = len(list(IMAGE_ROOT.glob("**/*.jpg")))
    print("=" * 60)
    print("K-Fashion 파이프라인 통합 테스트")
    print(f"입력: {SAMPLE_ROOT.name} (전체 {total_imgs}개 중 {TEST_LIMIT}개 샘플)")
    print("=" * 60)

    masked_results = step1_masking()
    if not masked_results:
        print(f"\n{FAIL} 마스킹 결과 없음. test_data 확인 필요.")
        return

    captions   = await step2_captioning()
    if not captions:
        print(f"\n{FAIL} 캡션 없음. GEMINI_API_KEY 확인 필요.")
        return

    index_rows = step3_index_fields(captions)
    index_rows, image_vecs = step4_embedding(masked_results, index_rows)
    index_rows = step5_classifier(masked_results, index_rows)
    step6_rdb(index_rows)
    step7_qdrant(index_rows, image_vecs)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"테스트 완료 ({elapsed:.1f}초)")
    print("=" * 60)
    for step, status in results.items():
        print(f"  {step:<25} {status}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
