# src/ — 핵심 라이브러리

Flow에서 호출되는 순수 Python 모듈들입니다.  
Prefect 의존성이 없어 단독 테스트 및 재사용이 가능합니다.

---

## 모듈별 역할

### `config.py` — 환경변수 로드

`.env` 파일을 읽어 외부 서비스 자격증명을 상수로 노출합니다.

```python
from src.config import (
    R2_BUCKET, R2_PUBLIC_URL,   # Cloudflare R2
    GEMINI_API_KEY, GEMINI_MODEL,  # Google Gemini
    DB_URL, DB_ACCESS_TOKEN,       # Turso
    QDRANT_URL, QDRANT_API_KEY,    # Qdrant
)
```

> `LS_URL`, `LS_API_TOKEN`, `LS_PROJECT_ID` 는 Label Studio 연동 시에만 필요합니다.

---

### `caption.py` — Gemini VLM 캡셔닝

마스킹된 패션 이미지를 Gemini Flash에 보내 구조화된 캡션을 생성합니다.

**주요 출력 필드:**

| 필드 | 예시 | 용도 |
|---|---|---|
| `caption_category` | `"스트레이트 데님 팬츠"` | 서브카테고리 |
| `caption_micro_details` | `["생지", "핀턱"]` | taxonomy 미커버 디테일 |
| `mood_and_tpo` | `["오피스룩", "이지웨어"]` | TPO 검색 태그 |
| `dense_caption` | 전체 텍스트 | E5 임베딩 입력 |
| `flat_tags` | 공백 구분 토큰 | BM25 검색 입력 |

**비동기 배치 처리:**

```python
from src.caption import batch_from_dir_async

await batch_from_dir_async(
    images_dir="data/masked_images_archive/inbox_unlabeled",
    out_path="output/captions.jsonl",
    concurrency=50,   # Gemini Flash ~2000 RPM 기준
)
```

- JSONL 체크포인트: 이미 완료된 `(file_id, category)` 쌍은 자동 스킵
- 파일명 규칙: `{file_id}_{category}.jpg` → 메타데이터 자동 파싱

---

### `masking.py` — 폴리곤 마스킹

Label Studio JSON의 폴리곤 좌표를 읽어 PIL로 의류 영역만 추출합니다.

```python
from src.masking import load_masked_items, apply_polygon_mask

# JSON + 원본 이미지 → 마스킹 이미지 목록
masked_items = load_masked_items(json_path, img_path, crop=True)
# 반환: [(category_kor, polygon_idx, masked_PIL), ...]

# 폴리곤 좌표 직접 적용
result = apply_polygon_mask(image, points, crop=True, bg_color=(255, 255, 255))
```

---

### `classifier/` — FashionClassifier 추론

marqo-fashionSigLIP 백본 위에 올린 멀티헤드 분류기입니다.

**모델 버전:**

| 모델 파일 | 대상 | 출력 헤드 |
|---|---|---|
| `fashion_classifier_v5.pt` | 상의 / 아우터 / 원피스 | `pattern_position` (multilabel), `pattern_size` (softmax), `trim` (softmax) |
| `fashion_classifier_bottom.pt` | 하의 | `bottom_length` (8-class), `bottom_waist_rise` (3-class) |

```python
from flows.shared.tasks_classify import classify_batch

results = classify_batch(
    masked_paths=["data/masked_images_archive/.../1028690_top.jpg"],
    model_path="models/fashion_classifier_v5.pt",
    bottom_model_path="models/fashion_classifier_bottom.pt",
    is_bottom=[False],
)
# 반환: [{"pattern_position": [...], "pattern_size": "중간", "trim": "없음"}]
```

---

### `r2_client.py` — R2 boto3 클라이언트

Cloudflare R2 연결 클라이언트 (boto3 S3 호환 API).

```python
from src.r2_client import get_r2_client

s3 = get_r2_client()
```

---

### `label_parser.py` — Label Studio JSON 파서

Label Studio 내보내기 JSON에서 `file_id`와 라벨 항목을 추출합니다.

```python
from src.label_parser import _parse_json_file

file_id, entry = _parse_json_file(Path("data/labels/1028690.json"))
items = entry.get("items", [])  # 의류 항목 목록
```

---

### `task_builder.py` — Label Studio Task 생성기

R2 URL + 라벨 정보로 Label Studio import용 JSON task를 구성합니다.

---

## 백본 모델: marqo-fashionSigLIP

- **아키텍처**: ViT-B-16-SigLIP (open_clip 기반)
- **임베딩 차원**: 768d
- **용도**: 이미지 임베딩 + 텍스트 임베딩 (동일 임베딩 공간)
- **로드 방법** (`tasks_embed.py`):

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16-SigLIP",
    pretrained="webli",
)
tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
```

> 첫 실행 시 Hugging Face에서 모델 가중치를 자동 다운로드합니다 (~900MB).
