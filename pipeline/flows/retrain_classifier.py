# 출처: tldusdmlskr/fashion-search classifier-v4/classifier/train_classifier_v4.py 기반 재작성
# 원본 대비 변경점:
#   - 하드코딩된 경로 → 파라미터화
#   - standalone main() → Prefect @flow로 래핑
#   - CSV 출력 + Label Studio 태스크 자동 등록 추가
#   - 실제 fine-tuning 루프 추가 (원본엔 없던 부분)
#
# 전체 재학습 사이클:
#   Phase 1 (sampling) : 현재 모델로 신규 적재 항목 추론 → 저신뢰도 샘플 CSV 추출
#                        원본 train_classifier_v4.py의 run_inference + sample_by_task 사용
#   Phase 2 (queue)    : 저신뢰도 샘플을 Label Studio에 어노테이션 태스크로 등록
#   Phase 3 (retrain)  : Label Studio 완료 어노테이션으로 분류기 헤드 fine-tune
#                        → models/fashion_classifier_v[N+1].pt 저장

from __future__ import annotations

import csv
import io
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
import torch.nn as nn
from prefect import flow, get_run_logger

# ── 상수 (train_classifier_v4.py 동일) ──────────────────────────────────────

PP_CLASSES   = ["none", "allover", "front", "back", "hem", "upper", "side", "sleeve"]
PS_CLASSES   = ["none", "tiny", "medium", "large"]
TRIM_CLASSES = ["plain", "banded", "mixed", "rolled", "unknown"]

EMBED_DIM    = 768
BATCH_SIZE   = 32
WORKERS      = 8
PER_TASK     = 100      # 태스크별 샘플 수
FOLDER_LIMIT = 15       # 폴더당 최대 샘플 수
INFER_LIMIT  = 10_000   # 추론 상한

FINE_TUNE_LR     = 1e-3
FINE_TUNE_EPOCHS = 10
MIN_ANNOTATIONS  = 50   # 재학습에 필요한 최소 어노테이션 수

# Label Studio 어노테이션 필드명
LS_PP_TAG   = "pattern_position"
LS_PS_TAG   = "pattern_size"
LS_TRIM_TAG = "trim"


# ── 모델 정의 (inference.py와 동일 구조) ────────────────────────────────────

class FashionAttributeClassifier(nn.Module):
    """출처: tldusdmlskr/fashion-search classifier-v4/classifier/inference.py 동일 구조"""
    def __init__(self, clip_model, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        self.classifier_pp   = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(PP_CLASSES)))
        self.classifier_ps   = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, len(PS_CLASSES)))
        self.classifier_trim = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(TRIM_CLASSES)))

    def forward(self, images):
        with torch.no_grad():
            emb = self.clip.encode_image(images).float()
        return {"pp": self.classifier_pp(emb), "ps": self.classifier_ps(emb), "trim": self.classifier_trim(emb)}


# ── Phase 1: 저신뢰도 샘플 추출 ─────────────────────────────────────────────
# 출처: train_classifier_v4.py — fetch_image, run_inference, sample_by_task

def _fetch_image(url: str):
    """출처: train_classifier_v4.py — fetch_image()"""
    try:
        resp = requests.get(url.strip(), timeout=10)
        resp.raise_for_status()
        from PIL import Image
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _run_inference(model, preprocess, device, records: list[dict]) -> list[dict]:
    """
    출처: train_classifier_v4.py — run_inference() 그대로 이식
    변경: MODEL_PATH 의존 제거, model 객체 직접 수신
    """
    all_results = []
    processed   = 0

    for start in range(0, min(len(records), INFER_LIMIT), BATCH_SIZE):
        batch  = records[start:start + BATCH_SIZE]
        images, valid = [], []

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            future_map = {executor.submit(_fetch_image, r["image_url"]): r for r in batch}
            for future in as_completed(future_map):
                rec = future_map[future]
                img = future.result()
                if img is not None:
                    images.append(preprocess(img))
                    valid.append(rec)

        if not images:
            continue

        tensor = torch.stack(images).to(device)
        with torch.no_grad():
            outputs = model(tensor)

        pp_sig   = torch.sigmoid(outputs["pp"]).cpu()
        ps_soft  = torch.softmax(outputs["ps"],   dim=1).cpu()
        trim_soft= torch.softmax(outputs["trim"], dim=1).cpu()

        for i, rec in enumerate(valid):
            pp_pred  = [PP_CLASSES[j] for j, v in enumerate(pp_sig[i]) if v > 0.5] or ["none"]
            ps_idx   = int(ps_soft[i].argmax())
            trim_idx = int(trim_soft[i].argmax())
            all_results.append({
                "image_id":  rec["image_id"],
                "folder":    rec.get("folder", ""),
                "image_url": rec["image_url"],
                "pp_pred":   ",".join(pp_pred),
                "ps_pred":   PS_CLASSES[ps_idx],
                "trim_pred": TRIM_CLASSES[trim_idx],
                "pp_conf":   round(float(pp_sig[i].max()),   4),
                "ps_conf":   round(float(ps_soft[i].max()),  4),
                "trim_conf": round(float(trim_soft[i].max()), 4),
            })

        processed += len(valid)
        print(f"  추론: {processed}/{min(len(records), INFER_LIMIT)}", end="\r")

    print()
    return all_results


def _sample_by_task(all_results: list[dict], task: str, n: int) -> list[dict]:
    """출처: train_classifier_v4.py — sample_by_task() 그대로 이식"""
    conf_key = f"{task}_conf"
    sorted_r = sorted(all_results, key=lambda x: x[conf_key])

    folder_total = Counter(r["folder"] for r in all_results)
    total        = sum(folder_total.values())

    folders = sorted(folder_total.keys())
    folder_quota = {}
    for folder in folders:
        ratio = folder_total.get(folder, 0) / max(total, 1)
        folder_quota[folder] = max(1, min(round(ratio * n), FOLDER_LIMIT))

    folder_groups = defaultdict(list)
    for r in sorted_r:
        folder_groups[r["folder"]].append(r)

    sampled = []
    for folder, quota in folder_quota.items():
        sampled.extend(folder_groups.get(folder, [])[:quota])

    if len(sampled) < n:
        seen = {r["image_id"] for r in sampled}
        rest = [r for r in sorted_r if r["image_id"] not in seen]
        sampled.extend(rest[:n - len(sampled)])

    return sampled[:n]


def _extract_low_confidence_samples(
    model_path: str,
    records: list[dict],
    output_csv: str,
) -> list[dict]:
    """
    현재 모델로 records 추론 → 저신뢰도 샘플 CSV 저장.
    출처: train_classifier_v4.py — main() 흐름 그대로, 파라미터화만 적용.
    """
    import open_clip
    from pathlib import Path as P

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    clip_model = clip_model.to(device)

    clf = FashionAttributeClassifier(clip_model).to(device)
    state = torch.load(model_path, map_location=device)
    # 헤드 키만 로드 (clip 제외)
    head_state = {k: v for k, v in state.items() if k.startswith("classifier_")}
    if head_state:
        clf.load_state_dict(head_state, strict=False)
    clf.eval()

    random.shuffle(records)
    all_results = _run_inference(clf, preprocess, device, records)

    pp_samples   = _sample_by_task(all_results, "pp",   PER_TASK)
    ps_samples   = _sample_by_task(all_results, "ps",   PER_TASK)
    trim_samples = _sample_by_task(all_results, "trim", PER_TASK)

    final, seen = [], set()
    for task, samples in [("pp", pp_samples), ("ps", ps_samples), ("trim", trim_samples)]:
        for r in samples:
            if r["image_id"] not in seen:
                seen.add(r["image_id"])
                final.append({**r, "sampled_by": task})

    random.shuffle(final)

    CSV_FIELDS = ["image_id", "folder", "image_url",
                  "pp_pred", "ps_pred", "trim_pred",
                  "pp_conf", "ps_conf", "trim_conf", "sampled_by"]
    P(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(final)

    print(f"  저신뢰도 샘플 저장: {output_csv} ({len(final)}건)")
    return final


# ── Phase 2: Label Studio 태스크 등록 ────────────────────────────────────────

def _push_to_label_studio(samples: list[dict]) -> int:
    """저신뢰도 샘플을 Label Studio에 어노테이션 태스크로 등록."""
    from src.config import LS_URL, LS_API_TOKEN, LS_PROJECT_ID

    headers = {
        "Authorization": f"Bearer {LS_API_TOKEN}",
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    tasks = []
    for s in samples:
        tasks.append({
            "data": {
                "image":    s["image_url"],
                "file_id":  s["image_id"],
                "pp_pred":  s["pp_pred"],
                "ps_pred":  s["ps_pred"],
                "trim_pred":s["trim_pred"],
                "pp_conf":  s["pp_conf"],
                "ps_conf":  s["ps_conf"],
                "trim_conf":s["trim_conf"],
                "retrain_candidate": True,
            }
        })

    # 100건씩 배치 등록
    registered = 0
    for i in range(0, len(tasks), 100):
        batch = tasks[i:i+100]
        resp  = requests.post(
            f"{LS_URL}/api/projects/{LS_PROJECT_ID}/import",
            headers=headers,
            json=batch,
        )
        if resp.status_code in (200, 201):
            registered += len(batch)
        else:
            print(f"  LS 등록 실패 ({resp.status_code}): {resp.text[:200]}")

    return registered


# ── Phase 3: Fine-tuning ─────────────────────────────────────────────────────

def _fetch_ls_annotations() -> list[dict]:
    """Label Studio에서 완료된 어노테이션 수집."""
    from src.config import LS_URL, LS_API_TOKEN, LS_PROJECT_ID

    headers = {"Authorization": f"Bearer {LS_API_TOKEN}", "ngrok-skip-browser-warning": "true"}
    samples, page = [], 1
    while True:
        resp = requests.get(f"{LS_URL}/api/tasks", headers=headers,
                            params={"project": LS_PROJECT_ID, "page": page, "page_size": 100})
        if resp.status_code != 200:
            break
        data, task_list = resp.json(), resp.json().get("tasks", [])
        if not task_list:
            break
        for task in task_list:
            for ann in task.get("annotations", []):
                pp, ps, trim = [], "none", "plain"
                for item in ann.get("result", []):
                    name, value = item.get("from_name", ""), item.get("value", {})
                    choices = value.get("choices", [])
                    if name == LS_PP_TAG:
                        pp = [c.lower() for c in choices]
                    elif name == LS_PS_TAG and choices:
                        ps = choices[0].lower()
                    elif name == LS_TRIM_TAG and choices:
                        trim = choices[0].lower()
                if pp or ps != "none" or trim != "plain":
                    samples.append({"image_url": task["data"].get("image", ""),
                                    "pp_labels": pp, "ps_label": ps, "trim_label": trim})
        if not data.get("next"):
            break
        page += 1
    return samples


def _fine_tune(annotations: list[dict], model_path: str, new_model_path: str) -> None:
    """
    어노테이션 기반 분류기 헤드 fine-tune.
    clip 인코더는 frozen, 3개 head만 학습.
    """
    import open_clip
    from torch.utils.data import DataLoader, Dataset

    class AnnotationDataset(Dataset):
        def __init__(self, samples, preprocess, device):
            self.samples   = samples
            self.preprocess= preprocess
            self.device    = device

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            img = _fetch_image(s["image_url"])
            if img is None:
                img = __import__("PIL").Image.new("RGB", (224, 224), (255, 255, 255))
            t   = self.preprocess(img)
            pp  = torch.zeros(len(PP_CLASSES))
            for lbl in s.get("pp_labels", []):
                if lbl in PP_CLASSES: pp[PP_CLASSES.index(lbl)] = 1.0
            ps   = torch.tensor(PS_CLASSES.index(s.get("ps_label", "none")), dtype=torch.long)
            trim = torch.tensor(TRIM_CLASSES.index(s.get("trim_label", "plain")), dtype=torch.long)
            return t, pp, ps, trim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    clip_model = clip_model.to(device)

    clf   = FashionAttributeClassifier(clip_model).to(device)
    state = torch.load(model_path, map_location=device)
    head_state = {k: v for k, v in state.items() if k.startswith("classifier_")}
    if head_state:
        clf.load_state_dict(head_state, strict=False)
        print(f"  기존 가중치 로드: {model_path}")

    dataset   = AnnotationDataset(annotations, preprocess, device)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(
        list(clf.classifier_pp.parameters()) +
        list(clf.classifier_ps.parameters()) +
        list(clf.classifier_trim.parameters()),
        lr=FINE_TUNE_LR,
    )
    loss_bce = nn.BCEWithLogitsLoss()
    loss_ce  = nn.CrossEntropyLoss()

    clf.train()
    for epoch in range(FINE_TUNE_EPOCHS):
        total = 0.0
        for imgs, pp_t, ps_t, trim_t in loader:
            imgs, pp_t = imgs.to(device), pp_t.to(device)
            ps_t, trim_t = ps_t.to(device), trim_t.to(device)
            out  = clf(imgs)
            loss = loss_bce(out["pp"], pp_t) + loss_ce(out["ps"], ps_t) + loss_ce(out["trim"], trim_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  Epoch {epoch+1}/{FINE_TUNE_EPOCHS} loss={total/len(loader):.4f}")

    Path(new_model_path).parent.mkdir(parents=True, exist_ok=True)
    # 헤드만 저장 (inference.py 호환: clip + heads 포함 state_dict)
    torch.save({**dict(clip_model.state_dict()), **dict(clf.state_dict())}, new_model_path)
    print(f"  새 모델 저장: {new_model_path}")


# ── Prefect Flow ─────────────────────────────────────────────────────────────

@flow(name="retrain-classifier", log_prints=True)
def retrain_classifier_flow(
    model_path: str = "models/fashion_classifier_v5.pt",
    new_model_path: str | None = None,    # None이면 model_path 덮어씀
    records: list[dict] | None = None,    # ingest_labeled_flow에서 전달받은 신규 항목 목록
    low_conf_csv: str = "output/low_confidence_new.csv",
    skip_finetune: bool = False,
    update_qdrant: bool = True,
    qdrant_update_limit: int = 5000,
) -> bool:
    """
    출처: tldusdmlskr/fashion-search classifier-v4/classifier/train_classifier_v4.py 기반
    ingest_labeled_flow에서 신규 >= 10,000건 시 자동 트리거.

    Phase 1: 신규 항목 추론 → 저신뢰도 샘플 CSV 저장
    Phase 2: Label Studio에 어노테이션 태스크 등록
    Phase 3: 기존 LS 어노테이션으로 분류기 헤드 fine-tune (skip_finetune=False 시)

    반환: True(재학습 완료) / False(건너뜀)
    """
    logger        = get_run_logger()
    t0            = time.time()
    out_model     = new_model_path or model_path

    if not Path(model_path).exists():
        logger.warning(f"모델 파일 없음: {model_path}. 재학습 건너뜀.")
        return False

    # ── Phase 1: 저신뢰도 샘플 추출 ────────────────────────────────────────
    logger.info("Phase 1: 현재 모델로 신규 데이터 추론 중...")

    if not records:
        # records 없으면 RDB에서 최근 INFER_LIMIT건 조회 (SELECT 1회)
        from libsql import connect
        from src.config import DB_URL, DB_ACCESS_TOKEN
        conn = connect(DB_URL, auth_token=DB_ACCESS_TOKEN, _uri=True)
        rows = conn.execute(
            f"SELECT file_id, image_url FROM fashion_items "
            f"ORDER BY indexed_at DESC LIMIT {INFER_LIMIT}"
        ).fetchall()
        conn.close()
        records = [{"image_id": r[0], "image_url": r[1], "folder": ""} for r in rows if r[1]]

    if not records:
        logger.warning("추론할 레코드 없음. 재학습 건너뜀.")
        return False

    logger.info(f"추론 대상: {len(records)}건")
    low_conf_samples = _extract_low_confidence_samples(model_path, records, low_conf_csv)
    logger.info(f"저신뢰도 샘플: {len(low_conf_samples)}건 → {low_conf_csv}")

    # ── Phase 2: Label Studio 큐 등록 ────────────────────────────────────
    logger.info("Phase 2: Label Studio 어노테이션 태스크 등록 중...")
    try:
        registered = _push_to_label_studio(low_conf_samples)
        logger.info(f"LS 태스크 등록: {registered}건 (PP/PS/TRIM 어노테이션 필요)")
    except Exception as e:
        logger.warning(f"LS 등록 실패: {e}. CSV만 저장됨 — 수동 업로드 가능.")

    # ── Phase 3: Fine-tuning (기존 LS 어노테이션 활용) ────────────────────
    if skip_finetune:
        logger.info("Phase 3: skip_finetune=True → 재학습 건너뜀.")
        logger.info(f"완료 ({time.time()-t0:.1f}초). 저신뢰도 CSV: {low_conf_csv}")
        return True

    logger.info("Phase 3: Label Studio 기존 어노테이션 수집 중...")
    annotations = _fetch_ls_annotations()
    logger.info(f"어노테이션 수집: {len(annotations)}건")

    if len(annotations) < MIN_ANNOTATIONS:
        logger.warning(
            f"어노테이션 부족 ({len(annotations)}/{MIN_ANNOTATIONS}). "
            "fine-tuning 건너뜀. Label Studio 어노테이션 완료 후 재실행."
        )
        return False

    logger.info(f"fine-tuning 시작: {len(annotations)}건, {FINE_TUNE_EPOCHS} epochs")
    _fine_tune(annotations, model_path, out_model)
    logger.info(f"새 모델 저장: {out_model}")

    # ── Qdrant 패턴 필드 업데이트 ────────────────────────────────────────
    if update_qdrant:
        logger.info(f"Qdrant 패턴 필드 재추론 (최근 {qdrant_update_limit}건)...")
        _rerun_qdrant(out_model, qdrant_update_limit)

    logger.info(f"재학습 완료 ({time.time()-t0:.1f}초)")
    return True


def _rerun_qdrant(model_path: str, limit: int) -> None:
    """재학습된 모델로 Qdrant 패턴 필드 업데이트."""
    import os
    from libsql import connect
    from src.config import DB_URL, DB_ACCESS_TOKEN
    from src.classifier.inference import FashionClassifier
    from qdrant_client import QdrantClient

    conn = connect(DB_URL, auth_token=DB_ACCESS_TOKEN, _uri=True)
    rows = conn.execute(
        f"SELECT file_id, category, image_url FROM fashion_items "
        f"ORDER BY indexed_at DESC LIMIT {limit}"
    ).fetchall()
    conn.close()
    if not rows: return

    clf    = FashionClassifier(model_path)
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    _CAT   = {"top": 0, "bottom": 1, "outerwear": 2, "dress": 3, "unknown": 4}

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        for file_id, cat, url in batch:
            if not url: continue
            try:
                res = clf.predict(url)
                try: base = int(file_id)
                except ValueError: base = abs(hash(file_id)) % 10_000_000
                pid = base * 10 + _CAT.get(cat, 5)
                client.set_payload(
                    collection_name="visual",
                    payload={"pattern_position": res["pattern_position"],
                             "pattern_size": res["pattern_size"],
                             "trim": res["trim"]},
                    points=[pid],
                )
            except Exception as e:
                print(f"  Qdrant 업데이트 실패 {file_id}: {e}")

    print(f"  Qdrant 패턴 필드 업데이트 완료: {len(rows)}건")
