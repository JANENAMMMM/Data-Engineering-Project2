import json
import pandas as pd
from pathlib import Path

LABELS_DIR = Path(__file__).parent / "labels"
CSV_PATH   = Path(__file__).parent / "sample_1500(2).csv"
OUT_PATH   = Path(__file__).parent / "tasks_final_2.json"
CACHE_PATH = Path(__file__).parent / "json_index_cache.json"

# ── 1. JSON 인덱스 로드 (캐시 있으면 재사용) ─────────────────────────────────
if CACHE_PATH.exists():
    print("캐시 로드 중...")
    with open(CACHE_PATH, encoding="utf-8") as f:
        json_index = {int(k): v for k, v in json.load(f).items()}
    print(f"캐시 로드 완료: {len(json_index)}개")
else:
    print("JSON 인덱싱 중... (최초 1회, 이후 캐시 사용)")
    json_index = {}

    for path in LABELS_DIR.glob("**/*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        img_info    = data["이미지 정보"]
        detail_info = data["데이터셋 정보"]["데이터셋 상세설명"]
        labeling    = detail_info["라벨링"]
        rect_coords = detail_info["렉트좌표"]
        file_id     = data["데이터셋 정보"]["파일 번호"]
        iw, ih      = img_info["이미지 너비"], img_info["이미지 높이"]

        items = []
        for ct in ['상의', '하의', '아우터', '원피스']:
            for i, item in enumerate(labeling.get(ct, [{}])):
                # 키 자체가 없거나 값이 None/빈 dict인 경우 스킵
                if not item or not any(item.values()):
                    continue
                rects = rect_coords.get(ct, [{}])
                rect  = rects[i] if i < len(rects) else {}
                bbox  = None
                if rect.get("X좌표") is not None:
                    bbox = [rect["X좌표"], rect["Y좌표"], rect["가로"], rect["세로"]]

                # 카테고리 없으면 의류 타입명으로 대체
                category  = item.get("카테고리") or ct
                color_str = item.get("색상") or "-"
                if item.get("서브색상"):
                    color_str += f" / {item['서브색상']}"

                items.append({
                    "clothing_type": ct,
                    "item_info": "\n".join([
                        f"타입: {ct} / {category}",
                        f"컬러: {color_str}",
                        f"핏: {item.get('핏') or '-'}",
                        f"기장: {item.get('기장') or '-'}",
                        f"소재: {', '.join(item.get('소재', [])) or '-'}",
                        f"디테일: {', '.join(item.get('디테일', [])) or '-'}",
                        f"프린트: {', '.join(item.get('프린트', [])) or '-'}",
                    ]),
                    "bbox": bbox,
                })

        json_index[file_id] = {"items": items, "iw": iw, "ih": ih}

    # 캐시 저장
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(json_index, f, ensure_ascii=False)
    print(f"JSON 인덱싱 완료 및 캐시 저장: {len(json_index)}개")

# ── 2. CSV 읽기 ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print(f"CSV 행 수: {len(df)}")

# ── 3. task 생성 ─────────────────────────────────────────────────────────────
tasks = []
missing = 0

for _, row in df.iterrows():
    file_id   = int(row["image_id"])
    image_url = row["image_url"]
    pp_pred   = row.get("pp_pred", "")
    ps_pred   = row.get("ps_pred", "")
    trim_pred = row.get("trim_pred", "")

    meta = json_index.get(file_id)
    if meta is None:
        missing += 1
        item_info   = f"file_id: {file_id}"
        predictions = []
    else:
        iw, ih = meta["iw"], meta["ih"]
        if meta["items"]:
            best      = meta["items"][0]
            item_info = best.get("item_info", "")
            bbox      = best.get("bbox")
        else:
            # 원본 JSON에 라벨링 없는 케이스 → 폴백
            item_info = f"스타일: {row.get('folder', '-')}\n(원본 속성 정보 없음)"
            bbox      = None

        predictions = []
        if bbox and iw and ih:
            predictions = [{
                "model_version": "metadata_bbox",
                "result": [{
                    "from_name": "bbox_display",
                    "to_name":   "image",
                    "type":      "rectanglelabels",
                    "value": {
                        "x":      (bbox[0] / iw) * 100,
                        "y":      (bbox[1] / ih) * 100,
                        "width":  (bbox[2] / iw) * 100,
                        "height": (bbox[3] / ih) * 100,
                        "rectanglelabels": ["현재 라벨링 대상"],
                    }
                }]
            }]

    tasks.append({
        "data": {
            "image":     image_url,
            "item_info": item_info,
            "file_id":   file_id,
            "pp_pred":   str(pp_pred),
            "ps_pred":   str(ps_pred),
            "trim_pred": str(trim_pred),
        },
        "predictions": predictions,
    })

# ── 4. 저장 ──────────────────────────────────────────────────────────────────
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"\n→ {OUT_PATH.name} 저장 완료 ({len(tasks)}개 task)")
if missing:
    print(f"  ⚠ JSON 매칭 실패: {missing}개 (image_id 불일치)")
print(json.dumps(tasks[0], ensure_ascii=False, indent=2))
