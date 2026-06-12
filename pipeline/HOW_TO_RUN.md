# HOW TO RUN — 파이프라인 실행 가이드

K-Fashion 자동화 파이프라인을 처음 세팅하고 실행하는 방법을 설명합니다.  
팀원 온보딩 기준으로 작성되었으며, 환경 세팅부터 시나리오별 실행까지 순서대로 따라하면 됩니다.

---

## 목차

1. [환경 세팅](#1-환경-세팅)
2. [Scenario A 테스트 — 라벨 있는 이미지 자동 색인](#2-scenario-a-테스트)
3. [Scenario B 테스트 (vlm_only) — 라벨 없는 이미지 즉시 색인](#3-scenario-b-테스트-vlm_only)
4. [Scenario B 테스트 (full) — Gradio 수동 라벨링 포함](#4-scenario-b-테스트-full)
5. [수업 시연 데모](#5-수업-시연-데모)
6. [자동화 데몬 실행](#6-자동화-데몬-실행)
7. [트러블슈팅](#7-트러블슈팅)

---

## 1. 환경 세팅

### 1-1. Python 가상환경 및 패키지 설치

프로젝트 루트(`Label Studio Project/`)에서 실행합니다.

```bash
# 가상환경 생성 (Python 3.10 이상 권장)
python -m venv .venv

# 활성화
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# Scenario B rembg 배경제거에 필요 (onnxruntime)
pip install onnxruntime
```

> `onnxruntime` 은 `requirements.txt` 에 포함되지 않아 별도 설치가 필요합니다.  
> rembg 첫 실행 시 u2net 모델(~176MB)이 자동 다운로드됩니다.

### 1-2. `.env` 파일 설정

프로젝트 루트(`Label Studio Project/.env`)에 아래 내용을 작성합니다.

```dotenv
# Cloudflare R2 (이미지 저장)
R2_ENDPOINT_URL=https://{account_id}.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET=your_bucket_name
R2_PUBLIC_URL=https://your_public_domain.com

# Google Gemini (VLM 캡셔닝)
GEMINI_API_KEY=your_gemini_api_key

# Turso (메타데이터 DB)
DB_URL=libsql://your-db.turso.io
DB_ACCESS_TOKEN=your_turso_token

# Qdrant (벡터 검색)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key

# Label Studio (선택 — Scenario A의 LS 연동 시에만 필요)
LS_URL=http://localhost:8080
LS_API_TOKEN=your_ls_token
LS_PROJECT_ID=1
```

### 1-3. 모델 파일 확인

```bash
# pipeline/ 내에 두 모델 파일이 있어야 합니다
ls pipeline/models/
# fashion_classifier_v5.pt
# fashion_classifier_bottom.pt
```

모델 파일이 없는 경우 팀 공유 스토리지에서 다운로드합니다.

### 1-4. fashionSigLIP 백본 다운로드

분류기 추론 및 임베딩에 사용하는 marqo-fashionSigLIP은 첫 실행 시 자동 다운로드됩니다.

```bash
# 미리 다운로드하려면 (약 900MB)
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')"
```

### 1-5. 작업 디렉토리 설정

**모든 실행 명령은 `pipeline/` 디렉토리를 기준으로 합니다.**

```bash
cd "Label Studio Project/pipeline"
```

---

## 2. Scenario A 테스트

**라벨(JSON) + 원본 이미지가 있을 때** 전체 파이프라인을 검증합니다.

### 2-1. 테스트 데이터 준비

```bash
# inbox/labeled/ 디렉토리 생성
mkdir -p data/inbox/labeled/images
mkdir -p data/inbox/labeled/labels

# 샘플 데이터 복사 (lab/ 폴더의 라벨 샘플 활용)
# images: 원본 JPG
cp "../lab/K_fashion 이미지 sample/원천데이터/원천데이터_1/레트로/1016530.jpg" \
   data/inbox/labeled/images/

# labels: 대응 JSON (파일명 = file_id, 확장자 .json)
# → 해당 JSON 파일을 data/inbox/labeled/labels/ 에 복사
```

> `lab/test_data/` 에도 소량의 테스트 이미지+JSON 세트가 있습니다.

### 2-2. 테스트 스크립트 실행

```bash
# pipeline/ 에서 실행
python -X utf8 scripts/test_pipeline.py
```

또는 Python 코드에서 직접 호출:

```python
import asyncio, sys, os
from pathlib import Path
from dotenv import load_dotenv

_p = Path.cwd()
PIPELINE = _p if (_p / "flows").exists() else _p / "pipeline"
sys.path.insert(0, str(PIPELINE)); os.chdir(PIPELINE)
load_dotenv(PIPELINE.parent / ".env")

from flows.ingest_labeled import ingest_labeled_flow

asyncio.run(ingest_labeled_flow(
    image_dir="data/inbox/labeled/images",
    label_dir="data/inbox/labeled/labels",
    concurrency=5,   # 테스트 시 낮게 설정
))
```

### 2-3. 결과 확인

```
[OK] 폴리곤 마스킹 완료: data/masked_images_archive/inbox/{file_id}_{cat}.jpg
[OK] R2 업로드 완료
[OK] VLM 캡셔닝: output/captions_*.jsonl
[OK] 분류기 추론
[OK] Turso 저장: status=complete
[OK] Qdrant 저장
```

DB 결과 조회:

```bash
# pipeline/ 에서
python scripts/check_turso_schema.py
```

---

## 3. Scenario B 테스트 (vlm_only)

**라벨 없이 이미지만 있을 때** VLM 출력만으로 즉시 색인하는 모드입니다.  
Gradio 대기 없이 완전 자동으로 처리됩니다.

### 3-1. indexed_ids.txt 캐시 정리 (재실행 시)

동일 이미지를 다시 처리하려면 캐시에서 해당 ID를 제거해야 합니다.

> **`test_unlabeled_full.py` (Scenario B full)** 는 실행 시작 시 자동으로 캐시와 `manual_labels.jsonl` 을 정리합니다. 별도 수작업 불필요.

vlm_only 테스트 또는 수동 정리가 필요한 경우:

```python
# Python으로 실행 (PowerShell의 Set-Content는 BOM 문제 발생)
import os
from pathlib import Path
_p = Path.cwd()
PIPELINE = _p if (_p / "flows").exists() else _p / "pipeline"
os.chdir(PIPELINE)

cache = Path("output/indexed_ids.txt")
ids = set(cache.read_text(encoding="utf-8").splitlines()) if cache.exists() else set()
ids -= {"1028690", "1029079", "101858"}   # 제거할 file_id
cache.write_text("\n".join(sorted(ids)), encoding="utf-8")
```

또는 프로젝트 루트의 `erase_cash.py` 를 실행하면 캐시 + `manual_labels.jsonl` 을 한 번에 정리합니다.

### 3-2. 테스트 실행

```bash
# pipeline/ 에서
python -X utf8 scripts/test_unlabeled.py
```

또는 직접 호출:

```python
import asyncio, sys, os
from pathlib import Path
from dotenv import load_dotenv

_p = Path.cwd()
PIPELINE = _p if (_p / "flows").exists() else _p / "pipeline"
sys.path.insert(0, str(PIPELINE)); os.chdir(PIPELINE)
load_dotenv(PIPELINE.parent / ".env")

from flows.ingest_unlabeled import ingest_unlabeled_flow

asyncio.run(ingest_unlabeled_flow(
    image_dir="data/inbox/unlabeled",
    mode="vlm_only",
    concurrency=5,
))
```

### 3-3. 예상 출력

```
신규 3건 처리 시작 (mode=vlm_only, run_id=xxxxxxxx)
rembg 자동 마스킹 완료: 3개
R2 업로드 완료: 3건
VLM 캡셔닝 진행 중...
분류기 추론 완료
임베딩 완료
VLM-Only 완료: 3건 (status=vlm_only)
```

---

## 4. Scenario B 테스트 (full)

**Gradio 수동 라벨링이 포함된** 완전한 Scenario B 테스트입니다.  
터미널 2개가 필요합니다.

### 4-1. 터미널 1 — Gradio UI 실행

```bash
# pipeline/ 에서
python flows/gradio_labeler.py
```

→ `http://localhost:7860` 접속 확인

### 4-2. 터미널 2 — 파이프라인 실행

```bash
# 다른 터미널, pipeline/ 에서
python -X utf8 scripts/test_unlabeled_full.py
```

또는 직접 호출:

```python
import asyncio, sys, os
from pathlib import Path
from dotenv import load_dotenv

_p = Path.cwd()
PIPELINE = _p if (_p / "flows").exists() else _p / "pipeline"
sys.path.insert(0, str(PIPELINE)); os.chdir(PIPELINE)
load_dotenv(PIPELINE.parent / ".env")

from flows.ingest_unlabeled import ingest_unlabeled_flow

asyncio.run(ingest_unlabeled_flow(
    image_dir="data/inbox/unlabeled",
    mode="full",
    concurrency=5,
    label_poll_interval=15,   # 테스트용: 15초마다 체크 (기본 300초)
))
```

### 4-3. Gradio에서 라벨링

파이프라인이 `pending_label` 상태로 등록한 뒤 Gradio 대기 상태가 됩니다.  
`http://localhost:7860` 에서 각 이미지를 라벨링합니다.

**라벨링 순서:**

```
① 이미지가 캔버스에 자동으로 로드됨
② 캔버스 위를 클릭해 의류 영역의 꼭짓점을 찍음 (최소 3개)
   - 첫 번째 점은 초록색, 이후는 빨간색
   - ↩ 되돌리기 / ✕ 초기화 버튼 사용 가능
③ "🔍 마스킹 미리보기" 클릭으로 폴리곤 마스킹 결과 확인
④ 오른쪽에서 라벨 입력:
   - 의류 타입 선택 (상의 / 하의 / 아우터 / 원피스)
   - 세부 카테고리 선택
   - 색상, 소재, 핏, 기장 등 입력
⑤ 하나의 이미지에 여러 의류가 있는 경우:
   → 첫 항목 폴리곤 + 라벨 → [+ 항목 추가] 클릭 (캔버스 초기화됨)
   → 다음 항목 폴리곤 + 라벨 → [+ 항목 추가] 반복
   → 마지막 항목 → [저장 후 다음 이미지 →] 클릭
⑥ 모든 이미지 완료 → 파이프라인 자동 재개
```

### 4-4. 파이프라인 자동 재개 확인

Gradio 라벨링 완료 후 터미널 2에서:

```
라벨링 완료: 이미지 3건, 항목 3개 — 파이프라인 재개
VLM 캡셔닝 진행 중...
분류기 추론 완료
임베딩 완료
Scenario B 완료: 3건 (status=complete)
```

### 4-5. 결과 확인

```python
import sys, os
from pathlib import Path
from dotenv import load_dotenv

_p = Path.cwd()
PIPELINE = _p if (_p / "flows").exists() else _p / "pipeline"
sys.path.insert(0, str(PIPELINE)); os.chdir(PIPELINE)
load_dotenv(PIPELINE.parent / ".env")

from libsql import connect
from src.config import DB_URL, DB_ACCESS_TOKEN

conn = connect(DB_URL, auth_token=DB_ACCESS_TOKEN, _uri=True)
rows = conn.execute(
    "SELECT file_id, status, label_color, dense_caption "
    "FROM fashion_items WHERE file_id IN ('1028690','1029079','101858')"
).fetchall()
for r in rows:
    print(r)
conn.close()
```

기대값: `status = "complete"`, `label_color` 채워짐

---

## 5. 수업 시연 데모

수업에서 자동화 파이프라인을 시연할 때 사용하는 통합 스크립트입니다.  
단계별 진행 상황을 출력하고 최종 결과를 테이블로 보여줍니다.

### 5-1. 준비

```bash
# 터미널 1: Gradio UI
cd "Label Studio Project/pipeline"
python flows/gradio_labeler.py
```

### 5-2. 데모 실행

```bash
# 터미널 2
cd "Label Studio Project/pipeline"
python demo_scenario_b.py
```

### 5-3. 데모 진행 순서

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  K-Fashion 자동화 파이프라인 — Scenario B 라이브 데모
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [1] 이미지 Inbox 투입
      ✓  투입: 1028690.jpg
      ✓  투입: 1029079.jpg
      ✓  투입: 101858.jpg

  [2] 파이프라인 시작 (mode='full', Gradio 대기 포함)
  ★★★ 지금 브라우저에서 http://localhost:7860 를 여세요! ★★★

      (AI 자동 배경제거 → R2 업로드 → DB pending 등록)
      (Gradio 라벨링 완료 대기 중...)

      (라벨링 완료 감지 → 파이프라인 재개)
      (VLM 캡셔닝 → 분류 → 임베딩 → 저장)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  파이프라인 처리 결과
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  file_id      category     status         color      pattern_size
  ------------ ------------ -------------- ---------- ------------
  ✓ 1028690    top          complete       화이트     없음
  ✓ 1029079    top          complete       블랙       스트라이프
  ✓ 101858     dress        complete       플로럴     대형
```

---

## 6. 자동화 데몬 실행

개발/운영 환경에서 inbox를 자동으로 감시하는 방법입니다.

```bash
cd "Label Studio Project/pipeline"

# 자동화 데몬 시작 (1분마다 inbox 스캔)
python deployments/inbox_watcher.py
```

**동작 확인:**

```
inbox watcher 시작 (1분 간격)
labeled inbox: data/inbox/labeled/images
unlabeled inbox: data/inbox/unlabeled
```

이제 `data/inbox/unlabeled/` 에 이미지를 복사하면 1분 내에 자동으로 처리됩니다.  
`data/inbox/labeled/` 에 이미지+JSON을 넣으면 Scenario A가 자동 실행됩니다.

> **Note**: Scenario B의 `mode="full"` 에서 Gradio 대기가 발생하므로,  
> 데몬 실행 전 `python flows/gradio_labeler.py` 를 별도 터미널에서 켜두세요.

---

## 7. 트러블슈팅

### 같은 이미지가 "이미 처리됨"으로 스킵될 때

```python
import os
from pathlib import Path
_p = Path.cwd()
PIPELINE = _p if (_p / "flows").exists() else _p / "pipeline"
os.chdir(PIPELINE)

cache = Path("output/indexed_ids.txt")
ids = set(cache.read_text(encoding="utf-8").splitlines()) if cache.exists() else set()
ids -= {"제거할_file_id_1", "제거할_file_id_2"}
cache.write_text("\n".join(sorted(ids)), encoding="utf-8")
print("캐시 정리 완료:", len(ids), "건 남음")
```

### `ModuleNotFoundError: No module named 'onnxruntime'`

```bash
pip install onnxruntime
```

### Gradio가 "완료! 전체 0개 처리됨" 표시

`data/inbox/unlabeled/` 가 비어있거나, `output/manual_labels.jsonl` 에 이미 해당 이미지가 기록된 경우입니다.

```bash
# inbox 확인
ls data/inbox/unlabeled/

# 수동 라벨 기록 확인 (마지막 10줄)
tail -10 output/manual_labels.jsonl
```

### Gradio 캔버스에 이미지가 로드되지 않을 때

`python flows/gradio_labeler.py` 를 `pipeline/` 디렉토리에서 실행했는지 확인합니다.  
`data/inbox/unlabeled/` 에 처리할 `.jpg` 파일이 있어야 캔버스에 이미지가 뜹니다.

```bash
# pipeline/ 에서
ls data/inbox/unlabeled/
```

### VLM 캡셔닝 rate limit 오류

`src/config.py` 에서 `GEMINI_MODEL` 을 낮은 요금제 모델로 변경합니다.

```python
GEMINI_MODEL = "gemini-2.5-flash-lite"   # 저렴한 모델
```

또는 `ingest_unlabeled_flow(concurrency=10)` 으로 동시 요청 수를 줄입니다.

### Turso 연결 오류

```bash
# .env 파일 위치 확인 (프로젝트 루트에 있어야 함)
ls ../.env

# DB 연결 테스트 (pipeline/ 에서)
python scripts/check_turso_schema.py
```

---

## 참고

| 문서 | 위치 |
|---|---|
| 파이프라인 전체 구조 · 스키마 · 구현 현황 | `pipeline/README.md` |
| 소스 라이브러리 상세 설명 | `pipeline/src/README.md` |
