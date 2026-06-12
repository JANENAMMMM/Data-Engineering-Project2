# K-Fashion 자동화 파이프라인

패션 이미지를 자동으로 마스킹 → 캡셔닝 → 분류 → 임베딩 → 저장하는
K-Fashion 검색 시스템의 데이터 색인 파이프라인입니다.

---

## 폴더 구조

```
pipeline/
├── flows/                  # Prefect 오케스트레이션 flow (진입점)
│   ├── ingest_labeled.py   # Scenario A: 라벨 있는 이미지 자동 처리
│   ├── ingest_unlabeled.py # Scenario B: 라벨 없는 이미지 처리
│   ├── gradio_labeler.py   # 수동 라벨링 UI (Gradio, localhost:7860)
│   ├── retrain_classifier.py  # 분류기 자동 재학습
│   └── shared/             # 각 처리 단계별 Prefect task
│       ├── tasks_mask.py   # 폴리곤/rembg 마스킹
│       ├── tasks_caption.py# Gemini VLM 캡셔닝
│       ├── tasks_classify.py  # FashionClassifier 추론
│       ├── tasks_embed.py  # marqo-fashionSigLIP 임베딩
│       ├── tasks_r2.py     # Cloudflare R2 업로드
│       ├── tasks_rdb.py    # Turso(libSQL) 저장
│       └── tasks_qdrant.py # Qdrant 벡터 저장
│
├── src/                    # 핵심 라이브러리 (flow에서 호출)
│   ├── config.py           # 환경변수 로드 (R2, Gemini, Turso, Qdrant 등)
│   ├── caption.py          # Gemini 비동기 캡션 빌더
│   ├── masking.py          # 폴리곤 마스킹 유틸 (PIL 기반)
│   ├── classifier/         # FashionClassifier v5 + BottomClassifier
│   ├── label_parser.py     # Label Studio JSON 파서
│   ├── r2_client.py        # R2 boto3 클라이언트
│   └── task_builder.py     # Label Studio task 생성기
│
├── prompts/                # VLM 시스템 프롬프트
│   └── caption_system_prompt.txt
│
├── models/                 # ML 모델 파일 (.pt)
│   ├── fashion_classifier_v5.pt      # 상·아우터·원피스 분류기
│   └── fashion_classifier_bottom.pt  # 하의 전용 분류기
│
├── data/                   # 런타임 데이터
│   ├── inbox/              # 신규 이미지 투입 디렉토리
│   │   ├── labeled/        # Scenario A: 이미지 + JSON 라벨
│   │   └── unlabeled/      # Scenario B: 이미지만
│   ├── labels/             # 기존 라벨 JSON 아카이브
│   └── masked_images_archive/  # 처리 완료 마스킹 이미지
│
├── output/                 # 파이프라인 산출물
│   ├── manual_labels.jsonl # Gradio 수동 라벨 기록
│   ├── indexed_ids.txt     # 색인 완료 file_id 캐시
│   └── captions_*.jsonl    # VLM 캡션 JSONL (체크포인트)
│
├── scripts/                # 테스트 및 유틸리티 스크립트
│   ├── test_pipeline.py        # Scenario A 전체 파이프라인 테스트
│   ├── test_unlabeled.py       # Scenario B vlm_only 테스트
│   ├── test_unlabeled_full.py  # Scenario B full (Gradio 포함) 테스트
│   └── check_turso_schema.py   # Turso DB 스키마 · row 수 확인
│
└── deployments/            # 자동화 실행 설정
    ├── inbox_watcher.py    # 폴더 감시 → flow 자동 트리거 (권장)
    └── deploy.py           # Prefect Cloud 배포 등록 (선택)
```

---

## 파이프라인 아키텍처

```
신규 이미지 도착 (inbox/)
        │
        ▼
 [inbox_watcher.py]  ← 1분마다 폴더 감시
        │
        ├─ 라벨 JSON 있음 ─────────────────────────────────────────┐
        │                                                           │
        └─ 라벨 없음 ────────────────────────────┐                 │
                                                 │                 │
              Scenario B                         │   Scenario A    │
      ┌──────────────────────┐          ┌────────────────────────┐ │
      │ ingest_unlabeled_flow│          │  ingest_labeled_flow   │ │
      │                      │          │                        │ │
      │  rembg 배경제거       │          │  폴리곤 마스킹          │ │
      │  R2 업로드            │          │  R2 업로드             │ │
      │  Gradio 라벨링 대기   │          │                        │ │
      │    └─ 사용자 마스킹   │          │                        │ │
      │       + 라벨 입력    │          │                        │ │
      └──────────┬───────────┘          └──────────┬────────────┘ │
                 │                                 │              │
                 └──────────────┬──────────────────┘              │
                                ▼                                  │
                        Gemini VLM 캡셔닝                           │
                        FashionClassifier 추론                      │
                        fashionSigLIP 768d 임베딩                   │
                                │                                  │
                      ┌─────────┴──────────┐                      │
                      ▼                    ▼                      │
                Turso (libSQL)         Qdrant                     │
                패션 메타데이터        벡터 검색 인덱스             │
```

**자동화 진입점:** `python deployments/inbox_watcher.py` (1분마다 inbox 감시)

**실행 위치:** 모든 명령은 `pipeline/` 디렉토리 기준. 상세 실행 방법은 **[HOW_TO_RUN.md](HOW_TO_RUN.md)** 참고.

| 상황 | 방법 |
|---|---|
| 로컬 테스트 / 빠른 확인 | `asyncio.run(flow_function(...))` 직접 호출 |
| 로컬 자동화 데몬 운영 | `python deployments/inbox_watcher.py` |
| 팀 서버에서 스케줄/원격 실행 | `deploy.py` + Prefect Cloud |

---

## Shared Tasks (flows/shared/)

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

**ingest_unlabeled_flow status 값:**

| 값 | 의미 |
|---|---|
| `pending_label` | rembg 완료, Gradio 라벨링 대기 중 |
| `vlm_only` | VLM 출력만으로 색인됨 (mode=vlm_only) |
| `complete` | 수동 라벨 포함 완전 색인 |

---

## 외부 서비스 의존성

| 서비스 | 용도 | 환경변수 키 |
|---|---|---|
| Cloudflare R2 | 마스킹 이미지 저장 | `R2_*` |
| Google Gemini | VLM 캡셔닝 | `GEMINI_API_KEY` |
| Turso (libSQL) | 패션 메타데이터 DB | `DB_URL`, `DB_ACCESS_TOKEN` |
| Qdrant Cloud | 벡터 검색 | `QDRANT_URL`, `QDRANT_API_KEY` |
| Label Studio | (선택) 라벨링 플랫폼 | `LS_URL`, `LS_API_TOKEN` |

---

## Turso 스키마 (fashion_items)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `file_id` | TEXT PK | 이미지 식별자 |
| `category` | TEXT | top / bottom / outerwear / dress / unknown |
| `image_url` | TEXT | R2 공개 URL |
| `label_color` / `label_sub_color` | TEXT | 수동 라벨: 색상 |
| `label_material` | TEXT (JSON) | 소재 리스트 |
| `label_fit` / `label_length` / `label_sleeve` / `label_neckline` | TEXT | 핏·기장·소매·넥라인 |
| `label_detail` / `label_print` | TEXT (JSON) | 디테일·프린트 리스트 |
| `caption_category` | TEXT | VLM: 세부 카테고리명 |
| `caption_micro_details` | TEXT (JSON) | VLM: 미세 디테일 |
| `mood_and_tpo` | TEXT (JSON) | VLM: 무드/TPO 태그 |
| `dense_caption` | TEXT | 임베딩용 명사구 나열 |
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
- payload: Turso 컬럼 전체 포함
- 인덱스: `flat_tags` 전문 검색 (whitespace tokenizer)

---

## Turso 과금 방지 원칙

- 중복 확인은 `output/indexed_ids.txt` 로컬 파일만 사용 — **DB SELECT 없음**
- 캐시 파일 없을 때만 `SELECT file_id FROM fashion_items` 1회
- **루프 안에서 DB SELECT 절대 금지**

---

## 구현 완료 vs 미구현

| 항목 | 상태 |
|---|---|
| Scenario A 전체 흐름 | ✅ |
| Scenario B 전체 흐름 (full + vlm_only) | ✅ |
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

## 참조 레포지토리

| 레포 | 브랜치 | 용도 |
|---|---|---|
| `SeoJimin1234/DE_pj2_Storage` | main | R2 업로드, Turso 적재 패턴 |
| `tldusdmlskr/fashion-search` | `dev` | url_utils, sampler, task_builder |
| `tldusdmlskr/fashion-search` | `classifier-v4` | inference.py, train_classifier_v4.py, train_bottom_classifier.py |
| `tldusdmlskr/fashion-search` | `annotation/VLM_captioning` | mask_from_r2.py 패턴 |
