# K-Fashion 하이브리드 검색 엔진 — 자동화 파이프라인

> 최종 업데이트: 2026-06-06  
> 전체 파이프라인 테스트 통과 기준 (전 단계 [OK])

---

## 목적

K-Fashion 이미지가 들어오면 완전 자동으로 처리해 벡터 DB까지 적재한다.

| 시나리오 | 조건 | 처리 흐름 |
|---|---|---|
| **Scenario A** | 이미지 + 라벨 JSON 동시 도착 | 폴리곤 마스킹 → R2 → VLM 캡셔닝 → 분류기 → 임베딩 → Turso/Qdrant |
| **Scenario B** | 이미지만 도착 (라벨 없음) | rembg 자동 마스킹 → Gradio 수동 라벨링 대기 → (A와 동일) |
| **10,000건↑** | Scenario A 완료 후 자동 트리거 | 저신뢰도 샘플 추출 → Label Studio 큐 → 분류기 fine-tune |

오케스트레이션: **Prefect v3** (셀프호스트, 무료)

---

## 전체 아키텍처

```
[60초마다] inbox_watcher_flow
      │
      ├─ data/inbox/labeled/ 에 이미지+JSON 감지
      │       └─ ingest_labeled_flow
      │               Step 1  폴리곤 마스킹    (src/masking.py)
      │               Step 2  R2 업로드        (tasks_r2.py)
      │               Step 3  VLM 캡셔닝       (Gemini, tasks_caption.py)
      │               Step 4  인덱스 생성      dense_caption / flat_tags
      │               Step 5  분류기 추론      ┬ v5 모델: pp/ps/trim (전체 의류)
      │                                        └ bottom 모델: 기장/허리라인 (하의만)
      │               Step 6  이미지 임베딩    marqo-fashionSigLIP 768d
      │               Step 7  Turso RDB 적재   (tasks_rdb.py)
      │               Step 8  Qdrant 적재      (tasks_qdrant.py)
      │               Step 9  10,000건↑?   →  retrain_classifier_flow
      │                           Phase 1  저신뢰도 샘플 추출 → CSV
      │                           Phase 2  Label Studio 태스크 등록
      │                           Phase 3  LS 어노테이션으로 v5 fine-tune
      │
      └─ data/inbox/unlabeled/ 에 이미지만 감지
              └─ ingest_unlabeled_flow
                      Step 1  rembg 자동 마스킹
                      Step 2  R2 업로드 + Turso pending 등록
                      Step 3  Gradio UI 폴링 대기 (output/manual_labels.jsonl)
                                └ python flows/gradio_labeler.py → localhost:7860
                      Step 4  라벨링 완료 시 → ingest_labeled_flow와 동일 흐름 재개
```

---

## 파일 구조

```
.
├── src/
│   ├── caption.py                    # Gemini VLM 캡셔닝
│   ├── masking.py                    # 폴리곤 마스킹
│   ├── label_parser.py               # 라벨 JSON 파싱
│   ├── r2_client.py                  # R2 boto3 클라이언트
│   ├── ls_client.py                  # Label Studio API
│   ├── config.py                     # 환경변수 로더
│   ├── url_utils.py                  # STYLE_MAP, build_image_url
│   └── classifier/
│       └── inference.py              # FashionClassifier + BottomClassifier
│
├── flows/
│   ├── ingest_labeled.py             # Scenario A flow
│   ├── ingest_unlabeled.py           # Scenario B flow
│   ├── retrain_classifier.py         # 재학습 flow (Phase 1~3)
│   ├── gradio_labeler.py             # Gradio 수동 라벨링 UI
│   └── shared/
│       ├── tasks_caption.py          # VLM 캡셔닝 @task
│       ├── tasks_r2.py               # R2 업로드 @task
│       ├── tasks_mask.py             # 폴리곤 마스킹 + rembg @task
│       ├── tasks_rdb.py              # Turso 적재 @task (캐시 중복 방지)
│       ├── tasks_classify.py         # 분류기 추론 @task (v5 + bottom)
│       ├── tasks_embed.py            # marqo-fashionSigLIP 임베딩 @task
│       └── tasks_qdrant.py           # Qdrant upsert @task
│
├── deployments/
│   ├── inbox_watcher.py              # ★ 자동화 진입점 (60초 주기 폴더 감시)
│   └── deploy.py                     # Prefect work pool 등록 (서버 환경용)
│
├── models/
│   ├── fashion_classifier_v5.pt      # ★ 메인 분류기 (pp/ps/trim, 전체 의류)
│   └── fashion_classifier_bottom.pt  # ★ 하의 전용 분류기 (기장/허리라인)
│
├── scripts/
│   └── test_pipeline.py              # 10개 샘플로 전체 파이프라인 테스트
│
├── data/
│   ├── inbox/
│   │   ├── labeled/images/           # ← 라벨 있는 신규 이미지 투입
│   │   ├── labeled/labels/           # ← 대응 JSON 투입
│   │   └── unlabeled/                # ← 라벨 없는 신규 이미지 투입
│   └── masked_images_archive/        # 폴리곤 마스킹 완료 이미지
│
├── output/
│   ├── indexed_ids.txt               # Turso 중복 방지 캐시 (DB SELECT 없음)
│   ├── .inbox_seen.txt               # inbox watcher 처리 완료 기록
│   ├── manual_labels.jsonl           # Gradio 수동 라벨링 결과
│   └── low_confidence_new.csv        # 재학습 저신뢰도 샘플
│
├── .env
├── requirements.txt
└── HANDOFF.md
```

---

## 실행 방법

### 자동화 시작 (권장)

```bash
# 터미널 1 (선택): Prefect UI (http://localhost:4200)
prefect server start

# 터미널 2: 자동화 시작 — 60초마다 inbox 감시
python deployments/inbox_watcher.py
```

### 수동 실행

```bash
# Scenario A: 라벨 있는 데이터 즉시 처리
python -c "
import asyncio, sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from flows.ingest_labeled import ingest_labeled_flow
asyncio.run(ingest_labeled_flow())
"

# Gradio 라벨링 UI (Scenario B 전용)
python flows/gradio_labeler.py   # → http://localhost:7860

# 전체 파이프라인 테스트
python -X utf8 scripts/test_pipeline.py
```

---

## 분류기 상세

두 모델이 동시에 로드되고 카테고리에 따라 분기 추론한다.

### fashion_classifier_v5.pt — 전체 의류 공통

| 출력 | 타입 | 클래스 |
|---|---|---|
| `pattern_position` | 멀티라벨 (sigmoid) | none / allover / front / back / hem / upper / side / sleeve |
| `pattern_size` | 단일 (softmax) | none / tiny / medium / large |
| `trim` | 단일 (softmax) | plain / banded / mixed / rolled / unknown |

### fashion_classifier_bottom.pt — 하의 전용 (category == "bottom")

| 출력 | 타입 | 클래스 |
|---|---|---|
| `bottom_length` | 단일 (softmax) | 발목 / 미디 / 초숏 / 숏 / 판별불가 / 카프리 / 맥시 / 버뮤다 |
| `bottom_waist_rise` | 단일 (softmax) | 하이웨이스트 / normal / 판별불가 |

백본: **marqo-fashionSigLIP** (ViT-B-16-SigLIP, 768d) — 두 모델이 동일 백본 공유, clip 레이어 frozen.

---

## Turso 스키마 (fashion_items)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `file_id` | TEXT PK | 이미지 식별자 |
| `category` | TEXT | top / bottom / outerwear / dress |
| `image_url` | TEXT | R2 공개 URL |
| `label_color` / `label_sub_color` | TEXT | 원본 JSON 라벨: 색상 |
| `label_material` | TEXT (JSON) | 소재 리스트 |
| `label_fit` / `label_length` / `label_sleeve` / `label_neckline` | TEXT | 핏·기장·소매·넥라인 |
| `label_detail` / `label_print` | TEXT (JSON) | 디테일·프린트 리스트 |
| `caption_category` | TEXT | VLM: 세부 카테고리명 |
| `caption_micro_details` | TEXT (JSON) | VLM: 미세 디테일 |
| `mood_and_tpo` | TEXT (JSON) | VLM: 무드/TPO 태그 |
| `dense_caption` | TEXT | E5 임베딩용 명사구 나열 |
| `flat_tags` | TEXT | BM25 역색인용 공백 분리 토큰 |
| `pattern_position` | TEXT (JSON) | v5 분류기: 패턴 위치 |
| `pattern_size` | TEXT | v5 분류기: 패턴 크기 |
| `trim` | TEXT | v5 분류기: 밑단 마감 |
| `bottom_length` | TEXT | bottom 분류기: 하의 기장 (하의만) |
| `bottom_waist_rise` | TEXT | bottom 분류기: 허리라인 (하의만) |
| `status` | TEXT | complete / vlm_only / pending_label |
| `pipeline_run_id` | TEXT | 파이프라인 실행 ID |

### Qdrant — visual 컬렉션

- 벡터 차원: **768** (marqo-fashionSigLIP, COSINE)
- 포인트 ID: `int(file_id) * 10 + category_offset` (결정적)
- payload: Turso 컬럼 전체 포함 (`bottom_length` / `bottom_waist_rise` 포함)
- 인덱스: `flat_tags` 전문 검색 (whitespace tokenizer)

---

## Gradio 수동 라벨링 UI

`python flows/gradio_labeler.py` → http://localhost:7860

| 필드 | 노출 조건 |
|---|---|
| 의류 타입 / 세부 카테고리 / 색상 / 소재 등 | 항상 |
| **허리라인 (Waist Rise)** | **하의 선택 시에만 표시** |
| **하의 기장 (bottom_length)** | **하의 선택 시에만 표시** |

라벨 저장 → `output/manual_labels.jsonl` (JSONL append)

---

## Turso 과금 방지 원칙

- 중복 확인은 `output/indexed_ids.txt` 로컬 파일만 사용 — DB SELECT 없음
- 캐시 파일 없을 때만 `SELECT file_id FROM fashion_items` 1회
- 재학습 시 `SELECT ... LIMIT N` 1회만
- **루프 안에서 DB SELECT 절대 금지**

---

## 환경변수 (.env)

```env
# Cloudflare R2
R2_ENDPOINT_URL=https://5511965f46e4453fe7096afc51682c84.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=project2
R2_PUBLIC_URL=https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev
R2_BUCKET_PREFIX=image

# Gemini API
GEMINI_API_KEY=...

# Label Studio
LS_URL=https://exposure-ransack-cyclic.ngrok-free.dev
LS_API_TOKEN=...
LS_PROJECT_ID=12

# Turso DB
DB_URL=libsql://project2-jimin.aws-ap-northeast-1.turso.io
DB_ACCESS_TOKEN=...

# Qdrant
QDRANT_URL=https://bbca1aa5-1bcc-45f3-8d18-f6e2e4372389.eu-central-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=...
```

---

## 테스트 결과 (2026-06-06)

### Scenario A — `python -X utf8 scripts/test_pipeline.py` (K_fashion sample 10건)

| 단계 | 결과 |
|---|---|
| 폴리곤 마스킹 | [OK] 14개 (레트로/로맨틱/리조트 균등) |
| VLM 캡셔닝 (Gemini) | [OK] 10건 |
| dense_caption / flat_tags | [OK] |
| 이미지 임베딩 (768d) | [OK] CPU 정상 |
| 분류기 추론 (v5 + bottom) | [OK] 하의 4건: 기장·허리라인 정상 출력 |
| Turso RDB | [OK] 10건 저장 |
| Qdrant visual | [OK] 10건 업로드 |

### Scenario B — `python -X utf8 scripts/test_unlabeled.py` (라벨 없는 이미지 3건)

| 단계 | 결과 |
|---|---|
| rembg 자동 마스킹 (u2net, 최초 실행 시 176MB 다운로드) | [OK] 3건 |
| R2 업로드 | [OK] 3건 |
| Turso pending 등록 | [OK] |
| VLM 캡셔닝 (vlm_only 모드) | [OK] 3건, dense_caption 정상 생성 |
| 분류기 추론 (v5 + bottom 로드) | [OK] — category=unknown이므로 bottom 미발동 |
| 이미지 임베딩 (768d) | [OK] |
| Turso RDB (status=vlm_only, category=unknown) | [OK] 3건 |
| Qdrant visual | [OK] 3건 |

> **참고:** rembg는 카테고리 판별 불가 → 모든 아이템이 `category=unknown`으로 저장됨.  
> bottom 분류기는 `category=="bottom"` 인 경우에만 작동 → Gradio 수동 라벨링 이후 재색인 시 활성화.

---

## 구현 완료 vs 미구현

| 항목 | 상태 |
|---|---|
| Scenario A 전체 흐름 | ✅ |
| Scenario B 전체 흐름 | ✅ |
| inbox watcher (60초 폴링) | ✅ |
| Gradio 수동 라벨링 (하의 전용 필드 포함) | ✅ |
| 분류기 추론 (v5 전체 + bottom 하의) | ✅ |
| 재학습 flow Phase 1~3 (v5 대상) | ✅ |
| Turso RDB (bottom 컬럼 포함) | ✅ |
| Qdrant (bottom payload 포함) | ✅ |
| **검색 API (BM25 + kNN 하이브리드)** | ❌ 미구현 |
| **reindex_partial.py (vlm_only 재색인)** | ❌ 미구현 |
| **bottom 모델 재학습 자동화** | ❌ 미구현 (수동: `train_bottom_classifier.py`) |
| GPU 가속 | ❌ 현재 CPU 전용 |

---

## 주의사항

1. **VLM 캡셔닝** — Claude Code 내부 실행 금지, 외부 터미널에서만
2. **rembg 최초 실행** — DNN 모델 ~170MB 자동 다운로드
3. **marqo-fashionSigLIP 최초 실행** — HuggingFace 자동 다운로드 (~수 GB)
4. **Turso `stream not found`** — `tasks_rdb.py` auto-reconnect 처리됨
5. **HuggingFace symlink 경고 (Windows)** — 기능 영향 없음, 무시 가능
6. **bottom 모델 재학습** — `retrain_classifier_flow`는 v5만 대상. bottom 재학습은 `tldusdmlskr/fashion-search` `classifier-v4` 브랜치의 `train_bottom_classifier.py` 수동 실행

---

## 참조 레포지토리

| 레포 | 브랜치 | 용도 |
|---|---|---|
| `SeoJimin1234/DE_pj2_Storage` | main | R2 업로드, Turso 적재 패턴 |
| `tldusdmlskr/fashion-search` | `dev` | url_utils, sampler, task_builder |
| `tldusdmlskr/fashion-search` | `classifier-v4` | inference.py, train_classifier_v4.py, **train_bottom_classifier.py** |
| `tldusdmlskr/fashion-search` | `annotation/VLM_captioning` | mask_from_r2.py 패턴 |
