# 출처: tldusdmlskr/fashion-search classifier-v4/classifier/inference.py
#   FashionClassifier 클래스 → src/classifier/inference.py 로 복사
# Prefect @task 래핑: 신규 작성
# 변경: BottomClassifier 지원 추가 (bottom_model_path, is_bottom 파라미터)

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image
from prefect import task

_classifier = None  # 프로세스 내 싱글턴 캐시
_classifier_key: tuple = (None, None)  # (model_path, bottom_model_path)


def _get_classifier(model_path: str, bottom_model_path: str | None = None):
    global _classifier, _classifier_key
    key = (model_path, bottom_model_path)
    if _classifier is None or _classifier_key != key:
        from src.classifier.inference import FashionClassifier
        _classifier     = FashionClassifier(model_path, bottom_model_path=bottom_model_path)
        _classifier_key = key
    return _classifier


@task(name="classify-fashion-attributes", retries=1)
def classify_batch(
    masked_paths: list[str],
    model_path: str = "models/fashion_classifier_v5.pt",
    bottom_model_path: str = "models/fashion_classifier_bottom.pt",
    is_bottom: Union[bool, list] = False,
    batch_size: int = 32,
) -> list[dict]:
    """
    출처: tldusdmlskr/fashion-search classifier-v4/classifier/inference.py
          FashionClassifier.predict_batch() 호출
    모델 파일 없으면 경고 후 빈 결과 반환 (파이프라인 중단 없음)
    is_bottom: bool(전체 동일) 또는 list[bool](이미지별). category=="bottom"인 경우 True.
    반환: [{"pattern_position": [...], "pattern_size": str, "trim": str,
             "bottom_length": str, "bottom_waist_rise": str,  # is_bottom=True 시만
             "confidence": {...}}, ...]
    """
    _EMPTY = {
        "pattern_position": [], "pattern_size": "unknown",
        "trim": "unknown", "bottom_length": "", "bottom_waist_rise": "",
        "confidence": {},
    }

    if not Path(model_path).exists():
        print(f"[경고] 분류기 모델 없음: {model_path} → 빈 결과 반환")
        return [_EMPTY.copy() for _ in masked_paths]

    # bottom 모델 없으면 경고만 출력하고 계속 (pp/ps/trim은 정상 동작)
    _bmp = bottom_model_path if (bottom_model_path and Path(bottom_model_path).exists()) else None
    if bottom_model_path and not _bmp:
        print(f"[경고] bottom 모델 없음: {bottom_model_path} → 하의 추론 비활성")

    clf    = _get_classifier(model_path, _bmp)
    images: list[Image.Image | None] = []

    for p in masked_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"이미지 로드 실패 {p}: {e}")
            images.append(None)

    valid_images  = [img for img in images if img is not None]
    valid_indices = [i for i, img in enumerate(images) if img is not None]

    # is_bottom 리스트 정렬 (None 슬롯 제외)
    if isinstance(is_bottom, bool):
        valid_is_bottom = [is_bottom] * len(valid_images)
    else:
        valid_is_bottom = [is_bottom[i] for i in valid_indices]

    raw = clf.predict_batch(valid_images, is_bottom=valid_is_bottom, batch_size=batch_size)

    # None 슬롯 복원
    results: list[dict] = [_EMPTY.copy() for _ in masked_paths]
    for idx, res in zip(valid_indices, raw):
        if res:
            results[idx] = res

    return results
