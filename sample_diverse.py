import json
import pandas as pd
from pathlib import Path

LABELS_DIR = Path("./labels")
CLOTHING_TYPES = ['상의', '하의', '아우터', '원피스']


def load_labels():
    rows = []
    for path in LABELS_DIR.glob("**/*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        labeling = data["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"]
        file_id = data["데이터셋 정보"]["파일 번호"]
        file_name = data["데이터셋 정보"]["파일 이름"]

        for ct in CLOTHING_TYPES:
            items = labeling.get(ct, [{}])
            for item in items:
                if not item.get("카테고리"):  # 빈 객체({}) 스킵
                    continue
                rows.append({
                    "file_id": file_id,
                    "file_name": file_name,
                    "clothing_type": ct,
                    "category": item.get("카테고리"),
                    "color": item.get("색상"),
                    "sub_color": item.get("서브색상"),
                    "fit": item.get("핏"),
                    "length": item.get("기장"),
                })

    return pd.DataFrame(rows)


def sample_diverse(df, n_total=1500, color_cap=50):
    samples = []
    clothing_types = [ct for ct in CLOTHING_TYPES if ct in df["clothing_type"].values]
    n_per_type = n_total // len(clothing_types)

    for ct in clothing_types:
        type_df = df[df["clothing_type"] == ct]
        n_cats = type_df["category"].nunique()
        n_per_cat = max(1, n_per_type // n_cats)

        for cat in type_df["category"].unique():
            cat_df = type_df[type_df["category"] == cat]

            # color cap: 특정 색상이 너무 많지 않도록 제한
            capped = (
                cat_df.groupby("color", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), color_cap), random_state=42))
            )

            samples.append(capped.sample(min(len(capped), n_per_cat), random_state=42))

    result = pd.concat(samples).drop_duplicates(subset=["file_id", "clothing_type"])
    return result.sample(frac=1, random_state=42).reset_index(drop=True)


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

    sampled.to_csv("sampled_labels.csv", index=False, encoding="utf-8-sig")
    print("\n→ sampled_labels.csv 저장 완료")
