# K-Fashion 하이브리드 검색 엔진 — 자동화 파이프라인 인수인계

> 최종 업데이트: 2026-06-05  
> 구현 완료 기준. Cursor / Claude 인수인계용.

---

## 프로젝트 목적

새로운 K-Fashion 이미지 데이터가 들어올 때 완전 자동으로:
- **Scenario A** (라벨 있음): 마스킹 → VLM 캡셔닝 → 분류 → 임베딩 → R2/Turso/Qdrant 적재
- **Scenario B** (라벨 없음): rembg 자동 마스킹 → Gradio 수동 라벨링 → (A와 동일)
- **10,000건↑ 적재 시**: 저신뢰도 샘플 추출 → Label Studio 큐 → 분류기 재학습 자동 트리거

오케스트레이션: **Prefect v3** (셀프호스트, 무료)

---

## 참조 레포지토리

| 레포 | 브랜치 | 용도 |
|---|---|---|
| `SeoJimin1234/DE_pj2_Storage` | main | R2 업로드 패턴, Turso DB 적재 패턴 |
| `tldusdmlskr/fashion-search` | `dev` | url_utils, sampler, task_builder |
| `tldusdmlskr/fashion-search` | `classifier-v4` | inference.py, train_classifier_v4.py |
| `tldusdmlskr/fashion-search` | `annotation/VLM_captioning` | mask_from_r2.py 패턴 |

---

## 전체 파일 구조

```
.
├── src/                              # 기존 코드 (수정 없음)
│   ├── caption.py                    # VLM 캡셔닝 (batch_from_dir_async 등)
│   ├── masking.py                    # 폴리곤 마스킹
│   ├── label_parser.py               # 라벨 JSON 파싱
│   ├── r2_client.py                  # R2 boto3 클라이언트
│   ├── ls_client.py                  # Label Studio API
│   ├── config.py                     # 환경변수 (R2, Gemini, LS, DB, Qdrant)
│   ├── url_utils.py                  # STYLE_MAP, build_image_url
│   └── classifier/
│       ├── __init__.py
│       └── inference.py              # FashionClassifier (fashion-search 복사)
│
├── flows/
│   ├── ingest_labeled.py             # Scenario A flow (10K↑ 시 재학습 트리거)
│   ├── ingest_unlabeled.py           # Scenario B flow (rembg→Gradio→적재)
│   ├── retrain_classifier.py         # 재학습 flow (train_classifier_v4 로직 기반)
│   ├── gradio_labeler.py             # Gradio 수동 라벨링 UI
│   └── shared/
│       ├── tasks_caption.py          # VLM 캡셔닝 @task
│       ├── tasks_r2.py               # R2 업로드 @task
│       ├── tasks_mask.py             # 폴리곤 마스킹 + rembg @task
│       ├── tasks_rdb.py              # Turso RDB 적재 @task (캐시 기반 중복 방지)
│       ├── tasks_classify.py         # 분류기 추론 @task (기본: v5 모델)
│       ├── tasks_embed.py            # marqo-fashionSigLIP 이미지/텍스트 임베딩 @task
│       └── tasks_qdrant.py           # Qdrant upsert @task
│
├── deployments/
│   ├── inbox_watcher.py              # ★ 자동화 진입점 (60초 주기 폴더 감시)
│   └── deploy.py                     # Prefect work pool 등록 (서버 환경용)
│
├── models/
│   └── fashion_classifier_v5.pt      # ★ 현재 사용 중인 분류기 모델
│
├── scripts/
│   └── test_pipeline.py              # 10개 샘플로 전체 파이프라인 테스트
│
├── data/
│   ├── inbox/
│   │   ├── labeled/images/           # ← 라벨 있는 신규 이미지 투입 폴더
│   │   ├── labeled/labels/           # ← 대응 JSON 투입 폴더
│   │   └── unlabeled/                # ← 라벨 없는 신규 이미지 투입 폴더
│   ├── labels/                       # 기존 구조화 레이블
│   └── masked_images_archive/        # 폴리곤 마스킹 완료 이미지
│
├── K_fashion 이미지 sample/           # 2,200건 샘플 (레트로/로맨틱/리조트)
├── test_data/                         # 테스트용 8건 (이미 DB 적재됨)
├── output/
│   ├── indexed_ids.txt               # Turso 중복 방지 캐시 (DB 조회 없음)
│   ├── .inbox_seen.txt               # inbox watcher 처리 완료 기록
│   ├── manual_labels.jsonl           # Gradio 수동 라벨링 결과
│   └── low_confidence_new.csv        # 재학습 저신뢰도 샘플 CSV
│
├── .env                              # 환경변수 (아래 참조)
├── requirements.txt                  # 전체 패키지 목록
└── HANDOFF.md                        # 이 파일
```

---

## 자동화 실행 방법

### 방법 1 — 단순 실행 (권장, Prefect 서버 선택)

```bash
# 터미널 1 (선택): Prefect UI (http://localhost:4200)
prefect server start

# 터미널 2: 자동화 시작 (60초마다 inbox 감시)
python deployments/inbox_watcher.py
```

### 방법 2 — Prefect 서버 + 워커 분리

```bash
prefect server start
prefect worker start --pool "local-process" --type process
python deployments/deploy.py
```

### 수동 실행

```bash
# 라벨 있는 데이터 즉시 처리
python -c "
import asyncio, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from flows.ingest_labeled import ingest_labeled_flow
asyncio.run(ingest_labeled_flow())
"

# Gradio 라벨링 UI 시작 (Scenario B)
python flows/gradio_labeler.py
# → http://localhost:7860

# 파이프라인 테스트 (K_fashion sample 10건)
python -X utf8 scripts/test_pipeline.py
```

---

## 자동화 흐름 요약

```
[60초마다] inbox_watcher_flow
    │
    ├─ data/inbox/labeled/ 에 이미지+JSON 감지
    │      └─ ingest_labeled_flow
    │              ├─ 폴리곤 마스킹 (src/masking.py)
    │              ├─ R2 업로드 (tasks_r2.py)
    │              ├─ VLM 캡셔닝 (Gemini, tasks_caption.py)
    │              ├─ dense_caption + flat_tags 생성
    │              ├─ 분류기 추론 (models/fashion_classifier_v5.pt)
    │              ├─ marqo-fashionSigLIP 임베딩 (768d)
    │              ├─ Turso RDB 적재
    │              ├─ Qdrant visual 컬렉션 적재
    │              └─ 신규 >= 10,000건? → retrain_classifier_flow
    │                      ├─ Phase 1: 저신뢰도 샘플 추출 (train_classifier_v4 로직)
    │                      │           → output/low_confidence_new.csv
    │                      ├─ Phase 2: Label Studio 어노테이션 태스크 등록
    │                      └─ Phase 3: 기존 LS 어노테이션으로 fine-tune
    │                                  → models/fashion_classifier_v[N+1].pt
    │
    └─ data/inbox/unlabeled/ 에 이미지만 감지
           └─ ingest_unlabeled_flow
                   ├─ rembg 자동 배경 제거
                   ├─ R2 업로드 (pending 상태)
                   ├─ Turso RDB 등록 (status=pending_label)
                   ├─ Prefect 5분마다 output/manual_labels.jsonl 폴링
                   │   (Gradio UI: python flows/gradio_labeler.py → localhost:7860)
                   └─ 라벨링 완료 시 → ingest_labeled_flow와 동일 흐름 재개
```

---

## 환경변수 (.env) — 현재 값 설정됨

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

# Qdrant (클라우드 프리티어)
QDRANT_URL=https://bbca1aa5-1bcc-45f3-8d18-f6e2e4372389.eu-central-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=...
```

---

## 모델 버전 관리

| 버전 | 파일 | 상태 |
|---|---|---|
| v5 | `models/fashion_classifier_v5.pt` | **현재 사용 중** |
| v4 | (삭제 또는 백업) | 이전 버전 |

- 재학습 시 `retrain_classifier_flow(new_model_path="models/fashion_classifier_v6.pt")` 로 새 버전 생성
- tasks_classify.py / ingest_labeled.py / ingest_unlabeled.py 기본값: `v5`

---

## Turso 과금 방지 원칙

- **중복 확인은 `output/indexed_ids.txt` 로컬 파일만 사용** — DB SELECT 없음
- 캐시 파일 없을 때만 `SELECT file_id FROM fashion_items` 1회 실행
- 재학습 시 `SELECT ... LIMIT N` 1회만 사용
- 루프 안에서 DB SELECT 절대 금지

---

## 테스트 결과 (2026-06-05 기준)

K_fashion 이미지 sample 10건으로 전체 파이프라인 검증 완료:

| 단계 | 결과 |
|---|---|
| 폴리곤 마스킹 | [OK] 14개 (레트로/로맨틱/리조트 균등) |
| VLM 캡셔닝 (Gemini) | [OK] 10건, 0건 실패 |
| dense_caption / flat_tags | [OK] 정상 생성 |
| 이미지 임베딩 (marqo-fashionSigLIP 768d) | [OK] CPU 기준 정상 |
| 분류기 추론 | 현재 v5 모델로 [OK] |
| Turso RDB | [OK] 10건 저장 |
| Qdrant visual 컬렉션 | [OK] 10건 업로드 |

---

## 아직 안 된 것 (우선순위 순)

| 항목 | 설명 |
|---|---|
| `flows/reindex_partial.py` | `status='vlm_only'` 항목 완전 재색인 (선택사항) |
| 검색 API | Phase 5 하이브리드 검색 엔드포인트 (BM25 + kNN) |
| GPU 가속 | 현재 CPU 전용. 15만 건 임베딩 시 GPU 권장 |

---

## 코드 출처 원칙

모든 파일 상단에 주석:
- `# 출처: SeoJimin1234/DE_pj2_Storage/파일명` — DE_pj2_Storage 이식
- `# 출처: tldusdmlskr/fashion-search 브랜치/파일명` — fashion-search 이식
- `# 출처: 현재 프로젝트 src/파일명` — 기존 코드 래핑
- `# 출처: 신규 작성` — 처음부터 작성

---

## 주의사항

1. **VLM 캡셔닝 (batch_from_dir_async)** → Claude Code 내부 실행 금지, 외부 터미널에서만
2. **rembg 최초 실행** → DNN 모델 ~170MB 자동 다운로드
3. **marqo-fashionSigLIP 최초 실행** → HuggingFace에서 자동 다운로드 (~수 GB)
4. **Turso `stream not found`** → tasks_rdb.py auto-reconnect 처리됨
5. **HuggingFace symlink 경고 (Windows)** → 기능에 영향 없음, 무시 가능
