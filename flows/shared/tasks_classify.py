# 출처: tldusdmlskr/fashion-search classifier-v4/classifier/inference.py
#   FashionClassifier 클래스 → src/classifier/inference.py 로 복사
# Prefect @task 래핑: 신규 작성

from __future__ import annotations

from pathlib import Path

from PIL import Image
from prefect import task

_classifier = None  # 프로세스 내 싱글턴 캐시


def _get_classifier(model_path: str):
    global _classifier
    if _classifier is None:
        from src.classifier.inference import FashionClassifier
        _classifier = FashionClassifier(model_path)
    return _classifier


@task(name="classify-fashion-attributes", retries=1)
def classify_batch(
    masked_paths: list[str],
    model_path: str = "models/fashion_classifier_v5.pt",
    batch_size: int = 32,
) -> list[dict]:
    """
    출처: tldusdmlskr/fashion-search classifier-v4/classifier/inference.py
          FashionClassifier.predict_batch() 호출
    모델 파일 없으면 경고 후 빈 결과 반환 (파이프라인 중단 없음)
    반환: [{"pattern_position": [...], "pattern_size": str,
             "trim": str, "confidence": {...}}, ...]
    """
    _EMPTY = {"pattern_position": [], "pattern_size": "unknown",
              "trim": "unknown", "confidence": {}}

    if not Path(model_path).exists():
        print(f"[경고] 분류기 모델 없음: {model_path} → 빈 결과 반환")
        return [_EMPTY.copy() for _ in masked_paths]

    clf    = _get_classifier(model_path)
    images: list[Image.Image | None] = []

    for p in masked_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"이미지 로드 실패 {p}: {e}")
            images.append(None)

    valid_images  = [img for img in images if img is not None]
    valid_indices = [i for i, img in enumerate(images) if img is not None]

    raw = clf.predict_batch(valid_images, batch_size=batch_size)

    # None 슬롯 복원
    results: list[dict] = [_EMPTY.copy() for _ in masked_paths]
    for idx, res in zip(valid_indices, raw):
        if res:
            results[idx] = res

    return results
