# Label Studio Project — 패션 이미지 라벨링 파이프라인

패션 이미지 데이터셋의 메타데이터 추출, 샘플링, Label Studio task 생성 및 Gemini 캡션 생성 파이프라인입니다.

## 환경 설정

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install boto3 pandas python-dotenv google-genai requests
```

`.env.example`을 복사해 `.env`를 만들고 값을 채우세요:

```bash
cp .env.example .env
```

## 파일 구조

| 파일 | 설명 |
|------|------|
| `Export_Metadata.py` | R2에서 라벨 JSON 다운로드 |
| `count_labels.py` | labels 폴더 파일 수 카운트 |
| `count_objects.py` | tasks_final.json task 수 카운트 |
| `sample_diverse.py` | 다양성 기반 샘플링 |
| `create_tasks.py` | 전체 데이터 샘플링 후 task 생성 |
| `create_tasks_from_csv.py` | CSV 기반 task 생성 |
| `create_tasks_test.py` | 10개 테스트용 task 생성 |
| `generate_caption.py` | Gemini로 dense caption 생성 |
| `fix_urls.py` | tasks JSON의 이미지 URL 일괄 수정 |
| `check_r2.py` | R2 버킷 구조 확인 |
| `update_ls_tasks.py` | Label Studio 기존 task 데이터 업데이트 |
| `query_ls_tasks.py` | Label Studio task ID 조회 |

## 실행 순서

```bash
# 1. R2에서 라벨 JSON 다운로드
python Export_Metadata.py

# 2. CSV 기반 task 생성
python create_tasks_from_csv.py

# 3. (선택) Gemini 캡션 추가
python generate_caption.py
```

## 데이터

- `labels/` — R2에서 다운로드한 원본 라벨 JSON (gitignore)
- `sample_1500.csv` — 샘플링된 이미지 목록 (gitignore)
- `tasks_final.json` — Label Studio import용 task 파일 (gitignore)
