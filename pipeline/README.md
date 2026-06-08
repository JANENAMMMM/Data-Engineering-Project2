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
├── deployments/            # 자동화 실행 설정
│   ├── inbox_watcher.py    # 폴더 감시 → flow 자동 트리거 (권장)
│   └── deploy.py           # Prefect Cloud 배포 등록 (선택)
│
└── demo_scenario_b.py      # 수업 시연 스크립트
```

---

## 파이프라인 아키텍처

```
신규 이미지 도착 (inbox/)
        │
        ▼
 [inbox_watcher.py]  ← 1분마다 폴더 감시
        │
        ├─ 라벨 JSON 있음 ──────────────────────────────────────────────┐
        │                                                               │
        └─ 라벨 없음 ─────────────────────────────────┐               │
                                                      │               │
              Scenario B                              │   Scenario A  │
      ┌───────────────────────┐              ┌────────────────────────┐│
      │ ingest_unlabeled_flow │              │  ingest_labeled_flow   ││
      │                       │              │                        ││
      │  rembg 배경제거        │              │  폴리곤 마스킹          ││
      │  R2 업로드             │              │  R2 업로드             ││
      │  Gradio 라벨링 대기    │              │                        ││
      │    └─ 사용자 마스킹    │              │                        ││
      │       + 라벨 입력     │              │                        ││
      └──────────┬────────────┘              └──────────┬─────────────┘│
                 │                                      │              │
                 └──────────────────┬───────────────────┘              │
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

---

## 빠른 시작

환경 설정과 실행 방법은 **[HOW_TO_RUN.md](HOW_TO_RUN.md)** 를 참고하세요.

- 환경 변수 설정 → 의존성 설치 → 모델 확인 → 실행
- Scenario A / B 테스트 방법
- 수업 시연 데모 실행 방법

---

## 외부 서비스 의존성

| 서비스 | 용도 | 환경변수 키 |
|---|---|---|
| Cloudflare R2 | 마스킹 이미지 저장 | `R2_*` |
| Google Gemini | VLM 캡셔닝 | `GEMINI_API_KEY` |
| Turso (libSQL) | 패션 메타데이터 DB | `DB_URL`, `DB_ACCESS_TOKEN` |
| Qdrant Cloud | 벡터 검색 | `QDRANT_URL`, `QDRANT_API_KEY` |
| Label Studio | (선택) 라벨링 플랫폼 | `LS_URL`, `LS_API_TOKEN` |
