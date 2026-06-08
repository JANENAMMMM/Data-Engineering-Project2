# deployments/ — 자동화 실행 설정

새로운 이미지가 `data/inbox/` 에 들어오면 파이프라인을 자동 트리거하는 컴포넌트입니다.

---

## `inbox_watcher.py` — 폴더 감시 실행 (권장)

Prefect 서버 없이 로컬에서 간단히 실행할 수 있는 **자동화 데몬**입니다.  
1분마다 inbox 폴더를 스캔해 신규 파일을 발견하면 해당 flow를 즉시 실행합니다.

```bash
# pipeline/ 에서 실행
python deployments/inbox_watcher.py
```

**감시 폴더 구조:**

```
data/inbox/
├── labeled/
│   ├── images/   ← 여기에 .jpg 추가 → ingest_labeled_flow 트리거
│   └── labels/   ← 대응하는 .json 라벨 (파일명 = file_id)
└── unlabeled/    ← 여기에 .jpg 추가 → ingest_unlabeled_flow 트리거
```

**동작 방식:**

- `output/.inbox_seen.txt` 에 처리 완료 file_id 기록 (DB 조회 없음)
- 처리 이력은 프로세스 재시작 후에도 유지됨
- Prefect UI(`http://localhost:4200`)에서 실행 이력 확인 가능 (Prefect 서버 실행 시)

---

## `deploy.py` — Prefect Cloud 배포 등록 (선택)

팀 운영 환경에서 Prefect Cloud를 통해 flow를 스케줄 실행하거나  
원격에서 트리거할 때 사용합니다. **로컬 테스트 시에는 불필요합니다.**

```bash
# 1. Prefect 서버 시작 (별도 터미널)
prefect server start

# 2. 워커 시작 (별도 터미널)
prefect worker start --pool "local-process" --type process

# 3. Flow 등록
python deployments/deploy.py

# 4. CLI 또는 Prefect UI에서 실행
prefect deployment run "ingest-labeled/ingest-labeled-prod"
```

---

## 언제 어느 것을 쓰나?

| 상황 | 방법 |
|---|---|
| 로컬 테스트 / 빠른 확인 | `asyncio.run(flow_function(...))` 직접 호출 |
| 로컬 자동화 데몬 운영 | `python deployments/inbox_watcher.py` |
| 팀 서버에서 스케줄/원격 실행 | `deploy.py` + Prefect Cloud |
