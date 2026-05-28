# K-Fashion 하이브리드 검색 엔진 파이프라인

## 프로젝트 개요

**목표:** "화이트 스티치 생지 데님", "퍼프 소매 트위드 블레이저" 같은 롱테일 마이크로 디테일 쿼리를 정확히 처리하는 패션 특화 하이브리드 검색 엔진 구축.

**데이터 규모:** 총 156,713건 (masked image 기준)

---

## Phase 1: 폴리곤 마스킹 전처리

원본 이미지에서 의류 실루엣만 남기고 배경을 순백색(RGB 255, 255, 255)으로 처리.

**저장 위치:** `data/masked_images_archive/masking_data (2)/`

**파일명 규칙:** `{file_id}_{category}.jpg`
- `category` 값: `top` / `bottom` / `outerwear` / `dress`
- 예: `1000029_outerwear.jpg`, `46538_bottom.jpg`

VLM에 마스킹 이미지를 입력하면 모델·조명·겹옷 등 시각적 노이즈가 없어 환각이 줄어든다.

---

## Phase 2: 기존 구조화 레이블

**저장 위치:** `data/labels/{스타일명}/{file_id}.json`

원본 JSON에 이미 아래 정보가 포함되어 있으므로 **VLM이 재추출하지 않는다.**

| 필드 | 값 예시 | 종류 수 |
|---|---|---|
| 카테고리 | 팬츠, 티셔츠, 원피스, 재킷 | 16종 |
| 색상 / 서브색상 | 블랙, 화이트, 인디고, 베이지 | 20종 |
| 소재 | 우븐, 데님, 니트, 린넨, 저지 | 24종 |
| 핏 | 노멀, 루즈, 오버핏, 스키니 | 6종 |
| 기장 | 크롭, 레귤러, 롱, 미디, 맥시 | 5종 |
| 소매기장 | 민소매, 반팔, 긴소매, 7부소매 | 5종 |
| 넥라인 | 브이넥, 라운드넥, 오프숄더 | 20종+ |
| 디테일 | 단추, 지퍼, 포켓, 플리츠 | 40종+ |
| 프린트 | 무지, 스트라이프, 체크, 플로럴 | 18종 |

**파싱 함수:** `src/caption.py` → `_parse_item_info(item_info: str) -> dict`
(item_info 텍스트 포맷을 구조화 dict로 변환. 인덱서에서 사용.)

---

## Phase 3: VLM 캡셔닝 (Gemini 2.5 Flash)

### VLM의 역할

기존 구조화 레이블이 커버하지 못하는 **3가지 정보만** 추출한다.

```python
class CaptionOutput(TypedDict):
    category: str             # 세밀한 서브카테고리 ("팬츠" → "스트레이트 데님팬츠")
    micro_details: list[str]  # taxonomy 미커버 디테일
    mood_and_tpo: list[str]   # 3~5개 검색용 TPO 태그
```

### micro_details 추출 범위

| 항목 | 예시 |
|---|---|
| 워싱·가공 | 생지, 캣브러쉬, 스톤워싱, 데미지 |
| 소재 텍스처 | 골지, 자카드, 번아웃, 코듀로이, 케이블 니트 |
| 섬세한 부자재 | 금장 버튼, 코퍼 리벳, 투웨이 지퍼, 플랩 포켓 |
| 구조적 디테일 | 핀턱, 셔링, 컷오프 헴라인, 드롭숄더, 퍼프 소매 |
| 로고·그래픽 | 좌측 가슴 스몰 자수, 백프린팅 |

일반 색상·소재·핏·넥라인·프린트는 포함하지 않는다.

### mood_and_tpo 허용 태그 풀 (20개)

```
하객룩, 오피스룩, 데이트룩, 피크닉룩, 바캉스룩, 페스티벌룩,
등산룩, 골프룩, 리조트룩, 웨딩게스트, 미니멀룩, 고프코어,
Y2K, 뉴트로, 페미닌, 클래식, 스트리트, 이지웨어, 데일리룩, 캐주얼
```

- `데일리룩`과 `캐주얼`은 동의어로 취급. 동시 출현 금지 (BM25 IDF 희소성 유지).

### 배치 실행

```python
from src.caption import batch_from_dir

batch_from_dir(
    images_dir="data/masked_images_archive/masking_data (2)",
    out_path="output/captions_full.jsonl",
    concurrency=50,
)
```

체크포인트 지원: 중단 후 재실행하면 완료된 항목을 자동 스킵.

---

## Phase 4: 캡션 출력 포맷 및 인덱스 문서 생성

### 4-1. 캡션 출력 (output/captions_full.jsonl)

배치 완료 후 JSONL 한 줄 = 이미지 1건.

```json
{
  "file_id": "1000029",
  "category": "outerwear",
  "caption_category": "니트 가디건",
  "caption_micro_details": ["플랩 포켓", "케이블 니트"],
  "mood_and_tpo": ["이지웨어", "데일리룩", "미니멀룩"]
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `file_id` | str | 이미지 식별자. 기존 레이블 JSON과 조인 키. |
| `category` | str | `top` / `bottom` / `outerwear` / `dress` |
| `caption_category` | str | VLM이 생성한 세밀한 서브카테고리 |
| `caption_micro_details` | list[str] | 마이크로 디테일 목록 (없으면 `[]`) |
| `mood_and_tpo` | list[str] | TPO 태그 3~5개 |

### 4-2. 인덱스 문서 생성 (인덱서 담당)

VLM 캡션과 기존 구조화 레이블은 **분리 저장**되어 있다.  
인덱서가 `file_id`를 키로 조인한 뒤 아래 두 필드를 생성해야 한다.

**조인 방법:**

```python
from src.caption import _parse_item_info, build_dense_caption, build_flat_tags
import json
from pathlib import Path

# captions_full.jsonl에서 VLM 출력 로드
vlm_row = json.loads(line)  # {"file_id": "1000029", "category": "outerwear", ...}

# 기존 레이블 로드 (data/labels/ 하위 file_id.json)
label_json = json.load(open(f"data/labels/.../{vlm_row['file_id']}.json"))
item_info_text = ...  # label_json에서 item_info 텍스트 추출
existing = _parse_item_info(item_info_text)  # dict로 파싱

# VLM CaptionOutput 형태로 변환
vlm = {
    "category":      vlm_row["caption_category"],
    "micro_details": vlm_row["caption_micro_details"],
    "mood_and_tpo":  vlm_row["mood_and_tpo"],
}

# 인덱스 필드 생성
dense_caption = build_dense_caption(vlm, existing)
flat_tags      = build_flat_tags(vlm, existing)
```

**dense_caption 예시 (E5 임베딩용, 콤마 구분):**
```
스트레이트 데님팬츠, 레귤러핏, 인디고, 데님, 롱기장, 생지, 화이트 스티치, 컷오프 헴라인, 코퍼 리벳, 고프코어, 스트리트, 데일리룩
```

**flat_tags 예시 (BM25 역색인용, 공백 구분):**
```
스트레이트데님팬츠 레귤러핏 인디고 데님 롱기장 생지 화이트스티치 컷오프헴라인 코퍼리벳 고프코어 스트리트 데일리룩
```

토큰 규칙:
- 공백 포함 토큰은 붙여쓰기로 정규화 ("화이트 스티치" → "화이트스티치")
- "무지" 제외 (검색 의미 없음)
- 중복 토큰 자동 제거 (기존 디테일 "플리츠" + VLM micro_details "플리츠" → 1개)
- 순서: VLM 서브카테고리 → 핏 → 색상/서브색상 → 소재 → 기장 → 소매기장 → 넥라인 → 프린트 → 기존 디테일 → VLM micro_details → TPO 태그

### 4-3. Elasticsearch 인덱스 스키마

```json
{
  "mappings": {
    "properties": {
      "file_id":        { "type": "keyword" },
      "category":       { "type": "keyword" },
      "flat_tags":      { "type": "text", "analyzer": "whitespace" },
      "dense_caption":  { "type": "text" },
      "dense_vector":   { "type": "dense_vector", "dims": 1024 },
      "clip_vector":    { "type": "dense_vector", "dims": 512 },
      "caption_category":      { "type": "keyword" },
      "caption_micro_details": { "type": "keyword" },
      "mood_and_tpo":          { "type": "keyword" }
    }
  }
}
```

- `flat_tags`는 `whitespace` analyzer 사용 (붙여쓰기 정규화된 토큰을 공백 기준으로만 분리)
- `dense_vector` dims=1024: E5-mistral-7b-instruct 또는 multilingual-e5-large 기준
- `clip_vector` dims=512: FashionCLIP 기준

---

## Phase 5: 2-Stage 하이브리드 검색

### Stage 1: 멀티 인덱스 검색 + RRF

3개 인덱스를 동시 검색 후 RRF(Reciprocal Rank Fusion)으로 합산.

```
score_rrf(d) = Σ 1 / (k + rank_i(d))    (k=60 권장)
```

| 인덱스 | 필드 | 모델/엔진 | 커버하는 쿼리 |
|---|---|---|---|
| Keyword | `flat_tags` | Elasticsearch BM25 | "화이트 스티치 생지 데님", "하객룩 원피스" |
| Semantic | `dense_vector` | E5 multilingual | "결혼식에 입을 드레스", "여름 여행 편한 옷" |
| Visual | `clip_vector` | FashionCLIP | 이미지 업로드 유사도 검색 |

**쿼리 타입별 기대 동작:**

| 쿼리 | 주도 인덱스 | 이유 |
|---|---|---|
| "화이트 스티치 생지 데님" | BM25 | micro_details에 exact 토큰 존재 |
| "하객룩 원피스" | BM25 | mood_and_tpo 태그 직접 매칭 |
| "결혼식에 입을 드레스" | E5 | "하객룩" 의미 근접 임베딩 |
| "리조트룩 린넨 원피스" | BM25 | 3개 토큰 모두 flat_tags에 존재 |
| "Y2K 무드 청바지" | E5 + BM25 | TPO 태그 + 데님 의미 |

**쿼리 처리:**
- 텍스트 쿼리 → E5로 임베딩 → `dense_vector` kNN 검색
- 텍스트 쿼리 → 그대로 → `flat_tags` BM25 검색
- Top-100 후보를 RRF로 합산

### Stage 2: Cross-Encoder 재랭킹

RRF Top-100 → Cross-Encoder → Top-10

- 모델: `cross-encoder/ms-marco-MiniLM-L-6-v2` 또는 한국어 특화 모델
- 입력: (쿼리 텍스트, dense_caption 문자열) 쌍
- 출력: 관련성 스코어 → 재정렬

---

## Phase 6: HITL 파인튜닝 (Label Studio)

Label Studio에서 수동 라벨링한 데이터로 FashionCLIP 엣지 클래스 파인튜닝.

- 취약 클래스 (오답률 높은 카테고리) 집중 라벨링
- Replay Buffer: 기존 데이터 20% 혼합 → Catastrophic Forgetting 방어
- 파인튜닝 후 `clip_vector` 재생성 필요

---

## 디렉토리 구조

```
.
├── src/
│   ├── caption.py        # VLM 캡셔닝 + 인덱스 필드 빌더
│   │                     #   batch_from_dir()       — 배치 실행 진입점
│   │                     #   build_dense_caption()  — 인덱서에서 호출
│   │                     #   build_flat_tags()      — 인덱서에서 호출
│   │                     #   _parse_item_info()     — 기존 레이블 파싱
│   ├── masking.py        # 폴리곤 마스킹 유틸리티
│   ├── config.py         # 환경변수 (GEMINI_API_KEY 등)
│   └── label_parser.py   # 원본 JSON 레이블 파서
├── scripts/
│   ├── test_caption.py   # 품질 검증 (test_data/ 샘플 10건)
│   └── generate_caption.py
├── prompts/
│   └── caption_system_prompt.txt   # Gemini 시스템 프롬프트
├── test_data/            # 검증용 샘플 이미지 + JSON (10건)
├── data/
│   ├── labels/           # 원본 구조화 레이블 JSON ({스타일}/{file_id}.json)
│   └── masked_images_archive/
│       └── masking_data (2)/   # 폴리곤 마스킹 이미지 156,713장
│                               # 파일명: {file_id}_{category}.jpg
└── output/
    └── captions_full.jsonl     # VLM 배치 출력 (인덱서 입력)
```

---

## 데이터 흐름 요약

```
data/masked_images_archive/masking_data (2)/*.jpg
    │
    ▼  [Phase 3] batch_from_dir() — Gemini 2.5 Flash, concurrency=50
    │
output/captions_full.jsonl
    { file_id, category, caption_category, caption_micro_details, mood_and_tpo }
    │
    ├── JOIN ──▶ data/labels/.../{file_id}.json  (기존 구조화 레이블)
    │              _parse_item_info() → existing dict
    │
    ▼  [Phase 4] 인덱서
    build_dense_caption(vlm, existing) → dense_caption 문자열
    build_flat_tags(vlm, existing)     → flat_tags 문자열
    E5 임베딩(dense_caption)           → dense_vector (1024d)
    FashionCLIP(masked image)          → clip_vector (512d)
    │
    ▼  Elasticsearch index
    │
    ▼  [Phase 5] 검색
    BM25(flat_tags) + kNN(dense_vector) + kNN(clip_vector)
    → RRF Top-100
    → Cross-Encoder Top-10
    │
    ▼  검색 결과
```
