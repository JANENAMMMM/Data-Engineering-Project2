# K-Fashion 하이브리드 검색 파이프라인

K-Fashion 이미지 156,713건에 Gemini VLM 캡션을 생성하고 BM25 + E5 + FashionCLIP 하이브리드 검색 엔진을 구축하는 파이프라인.

전체 설계 및 검색 전략은 [project_pipeline.md](project_pipeline.md) 참고.  
개발 과정 시행착오는 [dev_log.md](dev_log.md) 참고.

---

## 환경 설정

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env` 파일 생성 후 아래 값 설정:

```
GEMINI_API_KEY=...
LABEL_STUDIO_URL=...
LABEL_STUDIO_API_KEY=...
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=...
```

---

## 폴더 구조

```
.
├── src/
│   ├── caption.py        # VLM 캡셔닝 핵심 모듈
│   │                     #   batch_from_dir()       — 배치 실행 진입점 (이미지 디렉토리 직접 처리)
│   │                     #   build_dense_caption()  — 인덱서에서 호출 (E5 임베딩용)
│   │                     #   build_flat_tags()      — 인덱서에서 호출 (BM25용)
│   │                     #   _parse_item_info()     — 기존 레이블 텍스트 파싱
│   ├── masking.py        # 폴리곤 마스킹 유틸리티
│   ├── config.py         # 환경변수 로드 (GEMINI_MODEL 포함)
│   ├── label_parser.py   # 원본 JSON 레이블 파서
│   ├── ls_client.py      # Label Studio REST API 클라이언트
│   ├── r2_client.py      # Cloudflare R2 클라이언트
│   ├── sampler.py        # 다양성 기반 샘플링
│   └── task_builder.py   # Label Studio task 생성
│
├── scripts/
│   ├── test_caption.py         # VLM 출력 품질 검증 (test_data/ 10건)
│   ├── generate_caption.py     # 단건 캡션 생성 테스트
│   ├── export_metadata.py      # R2 → data/labels/ JSON 다운로드
│   ├── create_tasks.py         # Label Studio task 생성
│   ├── create_tasks_from_csv.py
│   ├── update_ls_tasks.py      # Label Studio task 업데이트
│   ├── fix_urls.py             # task JSON URL 일괄 수정
│   └── count.py                # 라벨/task 수 집계
│
├── prompts/
│   └── caption_system_prompt.txt   # Gemini 시스템 프롬프트
│
├── test_data/            # 품질 검증용 샘플 (이미지 10건 + JSON)
│
├── data/                 # 입력 데이터 (gitignore)
│   ├── labels/           # 원본 구조화 레이블 JSON ({스타일}/{file_id}.json)
│   └── masked_images_archive/
│       └── masking_data (2)/   # 폴리곤 마스킹 이미지 156,713장
│                               # 파일명: {file_id}_{category}.jpg
│
└── output/               # 생성 결과물 (gitignore)
    └── captions_full_lite.jsonl  # VLM 배치 출력 — 인덱서 입력
```

---

## VLM 캡션 배치 생성

task JSON 없이 이미지 디렉토리를 직접 처리한다.

```python
from src.caption import batch_from_dir

batch_from_dir(
    images_dir=r"data/masked_images_archive/masking_data (2)",
    out_path="output/captions_full_lite.jsonl",
    concurrency=50,
)
```

- 체크포인트 지원: 중단 후 재실행하면 완료된 항목 자동 스킵
- **외부 터미널에서 실행** (Claude Code 내 실행 시 timeout으로 강제 종료됨)
- 출력 포맷: `{ file_id, category, caption_category, caption_micro_details, mood_and_tpo }`

---

## 품질 검증

`test_data/` 샘플 10건으로 VLM 출력 품질 확인:

```powershell
python scripts/test_caption.py
```

결과는 `test_data/caption_test_results.jsonl`에 저장. TPO 태그풀 위반, 데일리룩+캐주얼 중복 자동 체크.

---

## 인덱서 연동 (검색팀 참고)

`captions_full_lite.jsonl`과 `data/labels/` JSON을 `file_id`로 조인해 검색 인덱스 필드 생성:

```python
from src.caption import _parse_item_info, build_dense_caption, build_flat_tags

vlm = {
    "category":      row["caption_category"],
    "micro_details": row["caption_micro_details"],
    "mood_and_tpo":  row["mood_and_tpo"],
}
existing = _parse_item_info(item_info_text)   # 기존 레이블 텍스트 파싱

dense_caption = build_dense_caption(vlm, existing)  # E5 임베딩 입력
flat_tags      = build_flat_tags(vlm, existing)      # BM25 역색인 입력
```

Elasticsearch 스키마, RRF 검색 전략 등 상세 내용은 [project_pipeline.md](project_pipeline.md) Phase 4–5 참고.
