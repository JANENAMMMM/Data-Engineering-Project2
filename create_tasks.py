import os
import json
import pandas as pd
from pathlib import Path
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

LABELS_DIR = Path(__file__).parent / "labels"
CLOTHING_TYPES = ['상의', '하의', '아우터', '원피스']

# ★ 여기 두 줄만 본인 값으로 채우세요
R2_PUBLIC_URL    = os.environ["R2_PUBLIC_URL"]
R2_BUCKET_PREFIX = "image"


def load_labels():
    rows = []
    for path in LABELS_DIR.glob("**/*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        img_info    = data["이미지 정보"]
        detail_info = data["데이터셋 정보"]["데이터셋 상세설명"]
        labeling    = detail_info["라벨링"]
        rect_coords = detail_info["렉트좌표"]

        file_id   = data["데이터셋 정보"]["파일 번호"]
        file_name = f"{file_id}.jpg"  # R2 파일명은 파일번호.jpg 형식
        style     = labeling.get("스타일", [{}])[0].get("스타일", "")
        iw        = img_info["이미지 너비"]
        ih        = img_info["이미지 높이"]

        for ct in CLOTHING_TYPES:
            items = labeling.get(ct, [{}])
            rects = rect_coords.get(ct, [{}])

            for i, item in enumerate(items):
                if not item.get("카테고리"):  # 빈 객체 스킵
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
                    "image_width":   iw,
                    "image_height":  ih,
                })

    return pd.DataFrame(rows)


def sample_diverse(df, n_total=1500, color_cap=50):
    samples = []
    present_types = [ct for ct in CLOTHING_TYPES if ct in df["clothing_type"].values]
    n_per_type = n_total // len(present_types)

    for ct in present_types:
        type_df = df[df["clothing_type"] == ct]
        n_per_cat = max(1, n_per_type // type_df["category"].nunique())

        for cat in type_df["category"].unique():
            cat_df = type_df[type_df["category"] == cat]
            capped = (
                cat_df.groupby("color", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), color_cap), random_state=42))
            )
            samples.append(capped.sample(min(len(capped), n_per_cat), random_state=42))

    result = pd.concat(samples).drop_duplicates(subset=["file_id", "clothing_type"])
    return result.sample(frac=1, random_state=42).reset_index(drop=True)


def make_tasks(df):
    tasks = []
    for _, row in df.iterrows():
        bbox = row["bbox"]
        iw, ih = row["image_width"], row["image_height"]

        color_str = row["color"] or "-"
        if row.get("sub_color"):
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

        image_url = f"{R2_PUBLIC_URL}/{R2_BUCKET_PREFIX}/{row['style']}/{row['file_name']}"
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

    return tasks


if __name__ == "__main__":
    print("JSON 파일 로딩 중...")
    df = load_labels()
    print(f"전체 라벨 수: {len(df)}")
    print(f"\n[clothing_type 분포]\n{df['clothing_type'].value_counts()}")
    print(f"\n[category 분포]\n{df['category'].value_counts()}")

    sampled = sample_diverse(df, n_total=1500, color_cap=50)
    print(f"\n샘플링 결과: {len(sampled)}개")
    print(f"\n[샘플 clothing_type 분포]\n{sampled['clothing_type'].value_counts()}")
    print(f"\n[샘플 category 분포]\n{sampled['category'].value_counts()}")

    tasks = make_tasks(sampled)

    with open("tasks2.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\n→ tasks.json 저장 완료 ({len(tasks)}개 task)")
    print(f"\n[첫 번째 task 미리보기]")
    print(json.dumps(tasks[0], ensure_ascii=False, indent=2))
