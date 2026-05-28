# Label Studio Project — 패션 이미지 라벨링 파이프라인

패션 이미지 데이터셋의 메타데이터 추출, 샘플링, Label Studio task 생성 및 Gemini 캡션 생성 파이프라인입니다.

## 환경 설정

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env.example`을 복사해 `.env`를 만들고 값을 채우세요:

```bash
cp .env.example .env
```

## 폴더 구조

```
├── src/                     # 재사용 가능한 모듈 (라이브러리)
│   ├── config.py            # 환경변수 및 도메인 상수
│   ├── url_utils.py         # 이미지 URL 생성 / 수정 (STYLE_MAP)
│   ├── r2_client.py         # Cloudflare R2 (S3) 클라이언트
│   ├── ls_client.py         # Label Studio REST API 클라이언트
│   ├── label_parser.py      # labels JSON 파싱, 인덱스 캐시
│   ├── task_builder.py      # Label Studio task dict 생성
│   ├── sampler.py           # 다양성 기반 샘플링
│   └── caption.py           # Gemini dense caption 생성
│
├── scripts/                 # 실행 진입점 (얇은 래퍼)
│   ├── export_metadata.py   # R2 → data/labels/ JSON 다운로드
│   ├── create_tasks.py      # 전체 데이터 샘플링 후 task 생성
│   ├── create_tasks_from_csv.py  # CSV 기반 task 생성
│   ├── generate_caption.py  # Gemini 캡션 추가
│   ├── fix_urls.py          # output/ tasks JSON URL 일괄 수정
│   ├── update_ls_tasks.py   # Label Studio 기존 task 업데이트
│   ├── count.py             # labels 파일 수 / task 수 카운트
│   └── query_ls_tasks.py    # Label Studio task ID 조회
│
├── data/                    # 입력 데이터 (gitignore)
│   ├── labels/              # R2에서 받은 원본 라벨 JSON
│   ├── json_index_cache.json  # 파싱 캐시 (자동 생성)
│   └── sample_1500(2).csv   # 샘플링 CSV
│
├── output/                  # 생성 결과물 (gitignore)
│   ├── tasks_final.json     # Label Studio import용 task 파일
│   └── tasks_with_caption.json  # 캡션 추가된 task 파일
│
├── .env                     # 인증 정보 (gitignore)
├── .env.example             # 인증 정보 템플릿
└── requirements.txt
```

## 실행 순서

```bash
# 1. R2에서 라벨 JSON 다운로드 → data/labels/
python scripts/export_metadata.py

# 2a. 전체 데이터 샘플링 후 task 생성 → output/tasks.json
python scripts/create_tasks.py

# 2b. (또는) CSV 기반 task 생성 → output/tasks_final.json
python scripts/create_tasks_from_csv.py data/sample_1500(2).csv output/tasks_final.json

# 3. (선택) 이미지 URL 수정 (한글 → 영문)
python scripts/fix_urls.py

# 4. (선택) Gemini 캡션 추가
python scripts/generate_caption.py output/tasks_final.json output/tasks_with_caption.json

# 5. (선택) Label Studio task 업데이트
python scripts/update_ls_tasks.py output/tasks_final.json
```

## 데이터

- `data/labels/` — R2에서 다운로드한 원본 라벨 JSON (gitignore)
- `data/json_index_cache.json` — 파싱 캐시 (자동 생성; 파싱 로직 변경 시 삭제)
- `output/tasks_final.json` — Label Studio import용 task 파일 (gitignore)
