# 개발 로그 — VLM 캡션 파이프라인

K-Fashion 이미지 데이터셋에 Gemini VLM 캡션 생성 파이프라인을 설계·구현하는 과정의 시행착오 기록.

---

## 1. VLM 출력 구조 설계 (8필드 → 3필드)

### 초기 설계 (8필드)
처음에는 VLM이 기존 구조화 레이블(색상, 소재, 핏 등)을 포함한 8개 필드를 전부 출력하도록 설계했다.

```python
class CaptionOutput(TypedDict):
    category: str
    sub_category: str
    color: str
    material: str
    fit: str
    length: str
    details: list[str]
    mood: list[str]
```

**문제**: 기존 JSON 레이블에 이미 색상·소재·핏 등이 구조화되어 있는데 VLM이 동일 정보를 중복 추출 → 일관성 저하, 토큰 낭비.

### 개선 (3필드)
기존 레이블에 없는 정보만 VLM이 담당하도록 역할 분리.

```python
class CaptionOutput(TypedDict):
    category: str             # 세밀한 서브카테고리 ("스트레이트 데님팬츠")
    micro_details: list[str]  # taxonomy 미커버: 워싱·텍스처·부자재·구조적 디테일
    mood_and_tpo: list[str]   # 검색용 TPO 태그 3~5개
```

**핵심 원칙**: VLM은 기존 데이터가 커버 못하는 영역만 채운다.

---

## 2. 시스템 프롬프트 반복 개선 (4라운드)

### Round 1 문제점
- `category`에 색상·핏이 prefix로 붙음: "루즈핏 셔츠원피스", "화이트 블라우스"
- `mood_and_tpo`에 형용사 등장: "편안한", "여유로운"
- `데일리룩` + `캐주얼` 동시 출력 (동의어 중복)

### Round 2 개선
- category 금지 prefix 명시: "색상(화이트·베이지), 핏(루즈핏·오버핏), 넥라인(브이넥·라운드넥) 앞에 붙이지 말 것"
- mood_and_tpo에 허용 태그 풀 20개 고정

**결과**: 색상·핏 prefix 제거됨. 데일리룩+캐주얼 중복은 5건 → 3건으로 감소.

### Round 3 개선
- 좋은 예 / 나쁜 예 few-shot 쌍 추가
- "데일리룩과 캐주얼은 완전히 동의어입니다. 절대 함께 쓰지 마십시오" 명문화

**결과**: 중복 2건 → 1건. 프롬프트만으론 100% 억제 불가 확인.

### Round 4 개선
- 코드 레벨에서 `_postprocess()` 추가: VLM 출력 후 "데일리룩" + "캐주얼" 동시 존재 시 "캐주얼" 자동 제거
- few-shot 예시 4개로 확장 (빈 micro_details 케이스 포함)

**결과**: 10/10 품질 이슈 0건. 프롬프트 + 코드 두 레이어로 방어.

---

## 3. 인덱스 필드 설계

### dense_caption (E5 임베딩용)
```
셔츠원피스, 루즈, 베이지, 린넨, 미디기장, 7부소매, 단추, 드롭 숄더, 데일리룩, 이지웨어, 미니멀룩
```
- 기존 레이블 + VLM 출력을 콤마로 결합
- 자연어 문장 대신 명사구 나열 → E5-multilingual 임베딩 최적화

### flat_tags (BM25 역색인용)
```
셔츠원피스 루즈 베이지 린넨 미디기장 7부소매 단추 드롭숄더 데일리룩 이지웨어 미니멀룩
```
- 공백으로 분리된 토큰 → BM25 IDF 계산용
- "화이트 스티치" → "화이트스티치" (공백 제거): 2-token으로 분리되면 IDF가 개별 단어로 희석됨

### 토큰 중복 제거 문제
기존 레이블의 `디테일: 플리츠`와 VLM `micro_details: 플리츠`가 동시에 들어오면 중복.
→ `_collect_tokens()`에 `seen` set 도입, 먼저 등장한 토큰이 우선.

---

## 4. 파이프라인 아키텍처 결정

### 초기: 기존 레이블 + VLM 출력 배치에서 결합
`_process_task`에서 `item_info` 파싱 → `build_dense_caption(vlm, existing)` 호출.

**문제**: `output/tasks_final_2.json`의 task는 `item_info` 텍스트 필드만 있고 구조화 키가 없어서 `existing`이 항상 `None`이 됨. dense_caption이 VLM 출력만으로 만들어지는 상황.

### 개선: 레이블과 캡션 분리 저장
> "기존 레이블이랑 vlm 생성 dense_caption이랑 분리해서 저장해도 되잖아"

배치 파이프라인은 VLM 3필드만 저장. 인덱스 결합은 Elasticsearch 업로드 단계에서 처리.

```python
# 배치 출력: VLM 3필드만
task["data"]["caption_category"]      = vlm["category"]
task["data"]["caption_micro_details"] = vlm["micro_details"]
task["data"]["mood_and_tpo"]          = vlm["mood_and_tpo"]

# build_dense_caption / build_flat_tags는 인덱서에서 호출
```

---

## 5. 이미지 소스 변경: R2 URL → 로컬 masked image

### 초기: R2 URL에서 이미지 fetch
```python
img_bytes, mime_type = await _fetch_image_bytes_async(url)
```

### 개선: 로컬 폴리곤 마스킹 이미지 사용
`data/masked_images_archive/masking_data (2)/` — 156,713장
- 파일명 패턴: `{file_id}_{category}.jpg`
- 폴리곤 마스킹으로 배경 제거 → VLM이 의류만 분석, 노이즈 감소

매핑 로직:
```python
_TYPE_TO_SUFFIX = {"상의": "top", "하의": "bottom", "아우터": "outerwear", "원피스": "dress"}
# item_info에서 타입 파싱 → {file_id}_{suffix}.jpg 경로 조립
# 로컬 파일 없으면 R2 URL 폴백
```

---

## 6. task JSON 제거

### 문제 인식
> "vlm에 이미지 주고 캡션만 생성한다고 했으면서 왜 task json이 필요한거야?"

task JSON이 필요했던 이유:
1. 이미지 URL (R2) → 로컬 파일로 대체
2. 기존 레이블 (item_info) → 분리 저장으로 배치에서 불필요

### 개선: `batch_from_dir()` 신규 함수
```python
batch_from_dir(
    images_dir="data/masked_images_archive/masking_data (2)",
    out_path="output/captions_full.jsonl",
    concurrency=50,
)
```
- 이미지 디렉토리 직접 순회
- 파일명에서 `file_id`, `category` 추출
- task JSON 완전히 불필요

---

## 7. 엔지니어링 최종 검토에서 발견한 버그 3개

### Bug 1: `asyncio.gather`에 156k 코루틴 동시 생성
```python
# 문제: 156k 코루틴 객체를 메모리에 전부 올린 뒤 gather
await asyncio.gather(*[_process_image(p, ...) for p in remaining])
```
→ 메모리 압박 + 이벤트 루프 초기화 지연

```python
# 수정: Queue + concurrency개 워커
queue = asyncio.Queue()
for p in remaining:
    queue.put_nowait(p)

async def worker():
    while True:
        try:
            img_path = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        await _process_image(img_path, sem, ...)

await asyncio.gather(*[worker() for _ in range(concurrency)])
```

### Bug 2: API 호출 타임아웃 없음
단일 요청이 hang하면 세마포어 슬롯 영구 점유 → 전체 파이프라인 정지.
```python
# 수정: 120초 타임아웃
response = await asyncio.wait_for(
    client.aio.models.generate_content(...),
    timeout=120,
)
```

### Bug 3: rate-limit 마지막 재시도에서 `None` 반환
```python
# 문제: 마지막 attempt에서 429 → sleep만 하고 raise 없이 함수 종료 → None 반환
if "429" in err_str:
    wait = _parse_retry_delay(err_str)
    await asyncio.sleep(wait)
elif attempt < retries - 1:
    await asyncio.sleep(2 ** attempt)
else:
    raise  # rate limit일 때는 여기 안 탐
```

```python
# 수정: rate-limit 여부와 무관하게 마지막 attempt는 항상 raise
is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
if attempt < retries - 1:
    wait = _parse_retry_delay(err_str) if is_rate_limit else 2 ** attempt
    await asyncio.sleep(wait)
else:
    raise
```

---

## 8. 최종 성능

| 단계 | 동시성 | 속도 | 에러율 |
|---|---|---|---|
| 초기 테스트 (concurrency=15) | 15 | ~3.5 req/s | 0% |
| 최종 (concurrency=50, Queue 패턴) | 50 | ~14 req/s | 0% |

- 156,713건 예상 소요: **약 185분 (3.1시간)**
- 출력 예상 용량: **약 29MB** (JSONL)
- 체크포인트: 중단 후 재실행 시 완료분 자동 스킵

---

## 최종 파이프라인 흐름

```
data/masked_images_archive/masking_data (2)/*.jpg
    ↓ batch_from_dir() — asyncio Queue, concurrency=50
    ↓ Gemini 2.5 Flash (response_schema=CaptionOutput, thinking_budget=0)
    ↓ _postprocess() — 데일리룩+캐주얼 중복 제거
output/captions_full.jsonl
    { file_id, category, caption_category, caption_micro_details, mood_and_tpo }
    ↓ (인덱서)
    ↓ build_dense_caption(vlm, existing) — E5 임베딩
    ↓ build_flat_tags(vlm, existing)     — BM25 역색인
Elasticsearch index
    ↓ BM25 + E5(dense) + FashionCLIP → RRF → Cross-encoder
검색 결과
```
