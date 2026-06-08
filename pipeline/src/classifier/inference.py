# 출처: tldusdmlskr/fashion-search classifier-v4/classifier/inference.py
# 변경: BottomClassifier 추가, FashionClassifier에 bottom_model_path 파라미터 추가

import io
import torch
import torch.nn as nn
import requests
from PIL import Image
from typing import Union


# ── 레이블 정의 ──────────────────────────────────────────────────────────────

PP_CLASSES     = ["none", "allover", "front", "back", "hem", "upper", "side", "sleeve"]
PS_CLASSES     = ["none", "tiny", "medium", "large"]
TRIM_CLASSES   = ["plain", "banded", "mixed", "rolled", "unknown"]
LENGTH_CLASSES = ["발목", "미디", "초숏", "숏", "판별불가", "카프리", "맥시", "버뮤다"]
WAIST_CLASSES  = ["하이웨이스트", "normal", "판별불가"]

EMBED_DIM    = 768
PP_THRESHOLD = 0.5


# ── 모델 정의 ────────────────────────────────────────────────────────────────

class FashionAttributeClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        self.classifier_pp   = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(PP_CLASSES)))
        self.classifier_ps   = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, len(PS_CLASSES)))
        self.classifier_trim = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(TRIM_CLASSES)))

    def forward(self, images):
        with torch.no_grad():
            emb = self.clip.encode_image(images).float()
        return {
            "pp":   self.classifier_pp(emb),
            "ps":   self.classifier_ps(emb),
            "trim": self.classifier_trim(emb),
        }


class BottomClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        self.classifier_length = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(LENGTH_CLASSES)))
        self.classifier_waist  = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, len(WAIST_CLASSES)))

    def forward(self, images):
        with torch.no_grad():
            emb = self.clip.encode_image(images).float()
        return {
            "length": self.classifier_length(emb),
            "waist":  self.classifier_waist(emb),
        }


# ── 추론 클래스 ──────────────────────────────────────────────────────────────

class FashionClassifier:
    def __init__(self,
                 model_path: str,
                 bottom_model_path: str = None,
                 device: str = None):
        """
        model_path:        fashion_classifier_v5.pt 경로
        bottom_model_path: fashion_classifier_bottom.pt 경로 (없으면 하의 추론 비활성)
        device:            'cuda' / 'cpu' / None(자동)
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[FashionClassifier] device: {self.device}")

        print("[FashionClassifier] marqo-fashionSigLIP 로드 중...")
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )
        self.clip_model = clip_model.to(self.device)
        self.preprocess = preprocess

        self.model = FashionAttributeClassifier(self.clip_model).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[FashionClassifier] v5 로드 완료: {model_path}")

        self.bottom_model = None
        if bottom_model_path:
            self.bottom_model = BottomClassifier(self.clip_model).to(self.device)
            self.bottom_model.load_state_dict(torch.load(bottom_model_path, map_location=self.device))
            self.bottom_model.eval()
            print(f"[FashionClassifier] bottom 로드 완료: {bottom_model_path}")
        else:
            print("[FashionClassifier] bottom_model_path 미지정 → 하의 추론 비활성")

    def _load_image(self, source: Union[str, Image.Image]) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        resp = requests.get(source.strip(), timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def _postprocess_main(self, outputs: dict) -> dict:
        pp_probs   = torch.sigmoid(outputs["pp"])[0]
        ps_probs   = torch.softmax(outputs["ps"],   dim=1)[0]
        trim_probs = torch.softmax(outputs["trim"], dim=1)[0]
        pp_labels  = [PP_CLASSES[i] for i, v in enumerate(pp_probs) if v > PP_THRESHOLD] or ["none"]
        return {
            "pattern_position": pp_labels,
            "pattern_size":     PS_CLASSES[int(ps_probs.argmax())],
            "trim":             TRIM_CLASSES[int(trim_probs.argmax())],
            "confidence": {
                "pp":   round(float(pp_probs.max()), 4),
                "ps":   round(float(ps_probs.max()),  4),
                "trim": round(float(trim_probs.max()), 4),
            },
        }

    def _postprocess_bottom(self, outputs: dict) -> dict:
        len_probs   = torch.softmax(outputs["length"], dim=1)[0]
        waist_probs = torch.softmax(outputs["waist"],  dim=1)[0]
        return {
            "bottom_length":     LENGTH_CLASSES[int(len_probs.argmax())],
            "bottom_waist_rise": WAIST_CLASSES[int(waist_probs.argmax())],
            "confidence_bottom": {
                "length": round(float(len_probs.max()),   4),
                "waist":  round(float(waist_probs.max()), 4),
            },
        }

    def predict(self, source: Union[str, Image.Image], is_bottom: bool = False) -> dict:
        image  = self._load_image(source)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            result = self._postprocess_main(self.model(tensor))
            if is_bottom:
                if self.bottom_model is None:
                    print("[Warn] bottom_model이 로드되지 않아 하의 추론을 건너뜁니다.")
                else:
                    result.update(self._postprocess_bottom(self.bottom_model(tensor)))
        return result

    def predict_url(self, url: str, is_bottom: bool = False) -> dict:
        return self.predict(url, is_bottom=is_bottom)

    def predict_batch(self,
                      sources: list,
                      is_bottom: Union[bool, list] = False,
                      batch_size: int = 32) -> list[dict]:
        if isinstance(is_bottom, bool):
            is_bottom = [is_bottom] * len(sources)

        results = []
        for start in range(0, len(sources), batch_size):
            batch        = sources[start:start + batch_size]
            batch_bottom = is_bottom[start:start + batch_size]
            images, valid_flags = [], []

            for src, ib in zip(batch, batch_bottom):
                try:
                    img = self._load_image(src)
                    images.append(self.preprocess(img))
                    valid_flags.append(ib)
                except Exception as e:
                    print(f"[Warn] 로드 실패: {e}")
                    results.append(None)

            if not images:
                continue

            tensor = torch.stack(images).to(self.device)
            with torch.no_grad():
                main_out   = self.model(tensor)
                bottom_out = self.bottom_model(tensor) if self.bottom_model else None

            for i, ib in enumerate(valid_flags):
                single_main = {k: v[i:i+1] for k, v in main_out.items()}
                res = self._postprocess_main(single_main)
                if ib and bottom_out is not None:
                    single_bottom = {k: v[i:i+1] for k, v in bottom_out.items()}
                    res.update(self._postprocess_bottom(single_bottom))
                results.append(res)

            print(f"[Batch] {min(start + batch_size, len(sources))}/{len(sources)} 완료", end="\r")

        print()
        return results
