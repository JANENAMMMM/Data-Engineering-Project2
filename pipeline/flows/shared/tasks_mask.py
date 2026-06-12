# 출처:
#   mask_images_local: 현재 프로젝트 src/masking.py → load_masked_items() 래핑
#   auto_mask_rembg: 신규 작성 (rembg 라이브러리, Scenario B용)
#   KOR_TO_EN 매핑 참고: tldusdmlskr/fashion-search annotation/VLM_captioning
#     polygon_masking/mask_from_r2.py

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from prefect import task

from src.masking import load_masked_items
from src.label_parser import _parse_json_file

KOR_TO_EN = {"상의": "top", "하의": "bottom", "아우터": "outerwear", "원피스": "dress"}


@task(name="mask-images-local", retries=1)
def mask_images_local(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    crop: bool = True,
) -> list[dict]:
    """
    출처: 현재 프로젝트 src/masking.py — load_masked_items() Prefect 래핑
    label_dir 하위 JSON 폴리곤 → 로컬 이미지 마스킹 → output_dir 저장
    반환: [{"file_id": str, "category": str, "output_path": str}]
    """
    image_dir_p  = Path(image_dir)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for json_path in sorted(Path(label_dir).glob("**/*.json")):
        try:
            file_id, _ = _parse_json_file(json_path)
            img_path   = image_dir_p / f"{file_id}.jpg"
            if not img_path.exists():
                continue

            # load_masked_items: 현재 프로젝트 src/masking.py
            # 반환: [(category_kor, polygon_idx, masked_PIL), ...]
            masked_items = load_masked_items(json_path, img_path, crop=crop)
            for cat_kor, _idx, masked_img in masked_items:
                cat_en   = KOR_TO_EN.get(cat_kor, cat_kor)
                out_name = f"{file_id}_{cat_en}.jpg"
                out_path = output_dir_p / out_name
                masked_img.save(out_path, "JPEG", quality=95)
                results.append({
                    "file_id":     str(file_id),
                    "category":    cat_en,
                    "output_path": str(out_path),
                })
        except Exception as e:
            print(f"마스킹 실패 {json_path.name}: {e}")

    print(f"폴리곤 마스킹 완료: {len(results)}개")
    return results


@task(name="auto-mask-rembg", retries=1)
def auto_mask_rembg(image_paths: list[str], output_dir: str) -> list[dict]:
    """
    출처: 신규 작성 (rembg 라이브러리 — pip install rembg)
    Scenario B: 폴리곤 없이 DNN 배경 제거 → 흰 배경(255,255,255) 합성
    반환: [{"file_id": str, "category": "unknown", "output_path": str}]
    """
    try:
        from rembg import remove
    except ImportError:
        raise RuntimeError("rembg 미설치. 'pip install rembg' 실행 필요.")

    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    print(f"\n  [rembg]  {len(image_paths)}건 배경 제거 시작  →  {output_dir}")
    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        try:
            print(f"  ▸ {img_path.stem}  배경 제거 중...", end="", flush=True)
            removed = remove(img_path.read_bytes())
            fg      = Image.open(io.BytesIO(removed)).convert("RGBA")
            bg      = Image.new("RGBA", fg.size, (255, 255, 255, 255))
            bg.paste(fg, mask=fg.split()[3])
            result_img = bg.convert("RGB")

            out_path = output_dir_p / f"{img_path.stem}_unknown.jpg"
            result_img.save(out_path, "JPEG", quality=95)
            results.append({
                "file_id":     img_path.stem,
                "category":    "unknown",
                "output_path": str(out_path),
            })
            w, h = result_img.size
            print(f"  ✔  {out_path.name}  ({w}×{h})")
        except Exception as e:
            print(f"  ✗")
            print(f"  rembg 실패 {img_path.name}: {e}")

    print(f"\n  [rembg]  자동 마스킹 완료: {len(results)}개")
    return results
