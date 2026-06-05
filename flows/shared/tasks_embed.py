# 출처: 신규 작성
# 이미지/텍스트 임베딩 모두 hf-hub:Marqo/marqo-fashionSigLIP (768d) 사용
# 동일 임베딩 공간 → 이미지-텍스트 크로스 모달 검색 가능
# 참고: 사용자 제공 임베딩 코드 (open_clip 기반)

from __future__ import annotations

import torch
import open_clip
from pathlib import Path
from PIL import Image
from prefect import task

MODEL_NAME = "hf-hub:Marqo/marqo-fashionSigLIP"
EMBED_DIM  = 768

_model      = None
_preprocess = None
_tokenizer  = None


def _get_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        print(f"[Embed] {MODEL_NAME} 로드 중...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
        _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        _model.eval()
        print("[Embed] 로드 완료")
    return _model, _preprocess, _tokenizer


@task(name="embed-images-siglip", retries=1)
def embed_images_siglip(image_paths: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    출처: 신규 작성 (사용자 제공 임베딩 코드 기반)
    hf-hub:Marqo/marqo-fashionSigLIP image encoder → 768d 코사인 정규화
    이미지 로드 실패 시 흰 이미지(224×224)로 대체
    """
    model, preprocess, _ = _get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    all_vecs: list[list[float]] = []

    for start in range(0, len(image_paths), batch_size):
        batch = image_paths[start:start + batch_size]
        tensors: list[torch.Tensor] = []

        for p in batch:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (255, 255, 255))
            tensors.append(preprocess(img))

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            vecs = model.encode_image(batch_tensor, normalize=True)

        all_vecs.extend(vecs.cpu().tolist())
        done = min(start + batch_size, len(image_paths))
        print(f"이미지 임베딩: {done}/{len(image_paths)}", end="\r")

    print()
    return all_vecs


@task(name="embed-texts-siglip", retries=1)
def embed_texts_siglip(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    """
    출처: 신규 작성
    hf-hub:Marqo/marqo-fashionSigLIP text encoder → 768d 코사인 정규화
    이미지 벡터와 동일한 임베딩 공간 → Qdrant visual 컬렉션에 텍스트로 검색 가능
    """
    model, _, tokenizer = _get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    all_vecs: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch  = texts[start:start + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            vecs = model.encode_text(tokens, normalize=True)
        all_vecs.extend(vecs.cpu().tolist())

    return all_vecs
