import json
import random
from pathlib import Path
from urllib.parse import quote

LABELS_DIR = Path(__file__).parent / "labels"
CLOTHING_TYPES = ['상의', '하의', '아우터', '원피스']
R2_PUBLIC_URL = "https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev"
R2_BUCKET_PREFIX = "image"  # 버킷(project2) 안의 실제 폴더

rows = []
files = list(LABELS_DIR.glob("**/*.json"))
random.seed(42)
random.shuffle(files)

for path in files:
    if len(rows) >= 10:
        break
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    img_info    = data["이미지 정보"]
    detail_info = data["데이터셋 정보"]["데이터셋 상세설명"]
    labeling    = detail_info["라벨링"]
    rect_coords = detail_info["렉트좌표"]
    file_id     = data["데이터셋 정보"]["파일 번호"]
    file_name   = f"{file_id}.jpg"  # R2 파일명은 파일번호.jpg 형식
    style       = labeling.get("스타일", [{}])[0].get("스타일", "")
    iw          = img_info["이미지 너비"]
    ih          = img_info["이미지 높이"]

    for ct in CLOTHING_TYPES:
        items = labeling.get(ct, [{}])
        rects = rect_coords.get(ct, [{}])
        for i, item in enumerate(items):
            if not item.get("카테고리"):
                continue
            rect = rects[i] if i < len(rects) else {}
            bbox = None
            if rect.get("X좌표") is not None:
                bbox = [rect["X좌표"], rect["Y좌표"], rect["가로"], rect["세로"]]
            rows.append({
                "file_id":       file_id,
                "file_name":     file_name,
                "style":         style,
                "clothing_type": ct,
                "category":      item.get("카테고리"),
                "color":         item.get("색상"),
                "sub_color":     item.get("서브색상"),
                "fit":           item.get("핏"),
                "length":        item.get("기장"),
                "materials":     item.get("소재", []),
                "details":       item.get("디테일", []),
                "prints":        item.get("프린트", []),
                "bbox":          bbox,
                "iw": iw, "ih": ih,
            })
        if len(rows) >= 10:
            break

tasks = []
for row in rows:
    bbox, iw, ih = row["bbox"], row["iw"], row["ih"]
    color_str = row["color"] or "-"
    if row["sub_color"]:
        color_str += f" / {row['sub_color']}"

    item_info = "\n".join([
        f"타입: {row['clothing_type']} / {row['category']}",
        f"컬러: {color_str}",
        f"핏: {row['fit'] or '-'}",
        f"기장: {row['length'] or '-'}",
        f"소재: {', '.join(row['materials']) if row['materials'] else '-'}",
        f"디테일: {', '.join(row['details']) if row['details'] else '-'}",
        f"프린트: {', '.join(row['prints']) if row['prints'] else '-'}",
    ])

    predictions = []
    if bbox and iw and ih:
        predictions = [{
            "model_version": "metadata_bbox",
            "result": [{
                "from_name": "bbox_display",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x":      (bbox[0] / iw) * 100,
                    "y":      (bbox[1] / ih) * 100,
                    "width":  (bbox[2] / iw) * 100,
                    "height": (bbox[3] / ih) * 100,
                    "rectanglelabels": ["현재 라벨링 대상"],
                }
            }]
        }]

    image_url = f"{R2_PUBLIC_URL}/{R2_BUCKET_PREFIX}/{quote(row['style'])}/{quote(row['file_name'])}"
    tasks.append({
        "data": {
            "image":         image_url,
            "item_info":     item_info,
            "file_id":       int(row["file_id"]),
            "clothing_type": row["clothing_type"],
            "category":      row["category"],
        },
        "predictions": predictions,
    })

with open("tasks_test5.json", "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"tasks_test5.json 저장 완료 ({len(tasks)}개)")
print(json.dumps(tasks[0], ensure_ascii=False, indent=2))
