# flows/ — Prefect 오케스트레이션 Flows

각 flow는 파이프라인의 진입점입니다.  
`asyncio.run()` 으로 직접 실행하거나, `deployments/inbox_watcher.py` 가 자동 트리거합니다.

> **실행 위치**: 반드시 `pipeline/` 디렉토리를 작업 디렉토리로 사용하세요.  
> 상대 경로(`data/`, `output/`, `models/`)가 `pipeline/` 기준으로 해석됩니다.

---

## Flow 파일별 역할

### `ingest_labeled.py` — Scenario A

라벨(JSON) + 원본 이미지가 동시에 도착했을 때 실행합니다.

```
data/inbox/labeled/images/  ← 원본 JPG
data/inbox/labeled/labels/  ← 폴리곤 라벨 JSON (파일명 = file_id)
```

**처리 단계:**

```
폴리곤 마스킹  →  R2 업로드  →  VLM 캡셔닝  →  분류기 추론  →  임베딩  →  Turso + Qdrant
```

**주요 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `image_dir` | `data/inbox/labeled/images` | 원본 이미지 디렉토리 |
| `label_dir` | `data/inbox/labeled/labels` | JSON 라벨 디렉토리 |
| `classifier_model` | `models/fashion_classifier_v5.pt` | 분류기 모델 경로 |
| `concurrency` | `50` | Gemini API 동시 호출 수 |

**직접 실행 예시:**

```python
import asyncio
from flows.ingest_labeled import ingest_labeled_flow

asyncio.run(ingest_labeled_flow(
    image_dir="data/inbox/labeled/images",
    label_dir="data/inbox/labeled/labels",
))
```

---

### `ingest_unlabeled.py` — Scenario B

라벨 없이 이미지만 도착했을 때 실행합니다.  
두 가지 모드로 동작합니다.

```
data/inbox/unlabeled/  ← 라벨 없는 원본 JPG
```

**`mode="full"` (권장)** — Gradio 수동 라벨링 포함

```
rembg 배경제거  →  R2 업로드  →  pending 등록
    →  Gradio 라벨링 대기 (사용자 마스킹 + 라벨 입력)
    →  VLM 캡셔닝  →  분류기  →  임베딩  →  Turso(complete) + Qdrant
```

**`mode="vlm_only"`** — VLM 출력만으로 즉시 색인 (긴급 적재용)

```
rembg 배경제거  →  R2 업로드  →  VLM 캡셔닝  →  분류기  →  임베딩  →  Turso(vlm_only) + Qdrant
```

**주요 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `image_dir` | `data/inbox/unlabeled` | 이미지 디렉토리 |
| `mode` | `"full"` | `"full"` 또는 `"vlm_only"` |
| `label_poll_interval` | `300` | 라벨링 완료 체크 주기(초) |
| `concurrency` | `50` | Gemini 동시 호출 수 |

**색인 상태값 (Turso `fashion_items.status`):**

| 값 | 의미 |
|---|---|
| `pending_label` | 라벨링 대기 중 |
| `vlm_only` | VLM 출력만으로 색인됨 |
| `complete` | 수동 라벨 포함 완전 색인 |

---

### `gradio_labeler.py` — 수동 라벨링 UI

Scenario B의 `mode="full"` 에서 사용자가 라벨을 입력하는 Gradio 웹 UI입니다.

```bash
# pipeline/ 에서 실행
python flows/gradio_labeler.py
# → http://localhost:7860
```

**UI 사용 방법:**

1. 왼쪽 `gr.ImageEditor`에서 **브러시로 의류 영역을 빨간색으로 칠합니다**
2. "🔍 마스킹 미리보기" 버튼으로 결과 확인
3. 오른쪽 폼에서 라벨 입력 (타입 → 카테고리 → 색상 → 소재 등)
4. **한 이미지에 여러 의류 항목이 있는 경우:**
   - 첫 항목 칠하기 + 라벨 입력 → **`+ 항목 추가`** 클릭 (저장, 폼 초기화, 같은 이미지 유지)
   - 마지막 항목 → **`저장 후 다음 이미지 →`** 클릭
5. 완료된 라벨은 `output/manual_labels.jsonl` 에 JSONL로 저장됩니다

**라벨 파일 형식 (`output/manual_labels.jsonl`):**

```json
{
  "file_id": "1028690",
  "masked_path": "data/masked_images_archive/inbox_unlabeled/1028690_top.jpg",
  "타입": "상의",
  "카테고리": "블라우스",
  "색상": "화이트",
  "소재": ["쉬폰"],
  "핏": "루즈",
  "기장": "레귤러",
  "labeled_by": "gradio_manual"
}
```

---

### `retrain_classifier.py` — 분류기 자동 재학습

`ingest_labeled_flow` 가 10,000건 이상 누적 처리 시 자동으로 호출됩니다.  
직접 실행도 가능합니다.

---

## shared/ — 단계별 Prefect Task

| 파일 | Task | 역할 |
|---|---|---|
| `tasks_mask.py` | `mask_images_local` | 폴리곤 JSON → PIL 마스킹 |
| | `auto_mask_rembg` | rembg DNN 배경 제거 (Scenario B) |
| `tasks_caption.py` | `run_vlm_caption_batch` | Gemini Flash 비동기 캡셔닝 |
| `tasks_classify.py` | `classify_batch` | FashionClassifier v5 추론 |
| `tasks_embed.py` | `embed_images_siglip` | marqo-fashionSigLIP 768d 이미지 임베딩 |
| | `embed_texts_siglip` | 텍스트 임베딩 (검색용) |
| `tasks_r2.py` | `upload_masked_images` | R2 업로드 (manifest 기반 재개 가능) |
| `tasks_rdb.py` | `upsert_rdb` | Turso INSERT OR REPLACE |
| | `filter_already_indexed` | `indexed_ids.txt` 캐시로 중복 방지 |
| `tasks_qdrant.py` | `upsert_qdrant` | Qdrant upsert (768d 벡터) |

---

## 중복 처리 방지 메커니즘

DB 쿼리 없이 로컬 캐시만으로 중복 처리를 방지합니다:

```
output/indexed_ids.txt
  ← upsert_rdb() 가 처리 시 file_id 추가 (status 무관)
  ← filter_already_indexed() 가 이 파일을 읽어 이미 처리된 ID 제외
```

> **테스트 재실행 시**: `indexed_ids.txt` 에서 해당 file_id를 직접 삭제해야  
> 같은 이미지를 다시 처리할 수 있습니다.
