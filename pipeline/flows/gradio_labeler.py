# 출처: 신규 작성 (Gradio 공식 docs 기반)
# 라벨 옵션값: 현재 프로젝트 project_pipeline.md 기반
# 실행: python flows/gradio_labeler.py
#        → http://localhost:7860 접속

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image

# ── 라벨 옵션 ─────────────────────────────────────────────────────────────────
CLOTHING_TYPES = ["상의", "하의", "아우터", "원피스"]

CATEGORIES: dict[str, list[str]] = {
    "상의":   ["티셔츠", "셔츠", "블라우스", "니트", "후드티", "맨투맨", "탑", "캐미솔"],
    "하의":   ["팬츠", "청바지", "스트레이트팬츠", "와이드팬츠", "슬랙스",
               "미니스커트", "미디스커트", "롱스커트", "숏츠"],
    "아우터": ["재킷", "코트", "가디건", "패딩", "블레이저", "점퍼", "베스트"],
    "원피스": ["미니원피스", "미디원피스", "맥시원피스", "점프수트"],
}

COLORS = [
    "블랙", "화이트", "그레이", "네이비", "블루", "인디고",
    "레드", "핑크", "베이지", "브라운", "카키", "그린",
    "옐로우", "오렌지", "퍼플", "민트", "코럴", "아이보리",
    "버건디", "올리브",
]
MATERIALS = [
    "우븐", "데님", "니트", "린넨", "저지", "쉬폰",
    "실크", "코튼", "폴리에스터", "레이온", "벨벳",
    "코듀로이", "트위드", "가죽", "스웨이드", "플리스",
    "테리", "메시",
]
FITS      = ["노멀", "루즈", "오버핏", "슬림", "스키니", "레귤러"]
LENGTHS   = ["크롭", "레귤러", "롱", "미디", "맥시"]
SLEEVES   = ["민소매", "반팔", "7부소매", "긴소매"]
NECKLINES = [
    "라운드넥", "브이넥", "터틀넥", "오프숄더",
    "스퀘어넥", "홀터넥", "보트넥", "U넥", "칼라", "후드",
]
DETAILS = [
    "단추", "지퍼", "포켓", "플리츠", "셔링", "핀턱",
    "레이스", "러플", "리본", "프릴", "슬릿", "벨트",
    "드롭숄더", "퍼프소매", "컷오프헴", "로고자수",
]
PRINTS = [
    "무지", "스트라이프", "체크", "플로럴", "도트",
    "레터링", "페이즐리", "타이다이", "카무플라주", "동물프린트",
]
BOTTOM_WAIST_RISE = ["하이웨이스트", "normal", "판별불가"]
BOTTOM_LENGTH     = ["발목", "미디", "초숏", "숏", "판별불가", "카프리", "맥시", "버뮤다"]

QUEUE_DIR          = Path("data/inbox/unlabeled")
OUTPUT_PATH        = Path("output/manual_labels.jsonl")
MASKED_ARCHIVE_DIR = Path("data/masked_images_archive/inbox_unlabeled")

CLOTHING_TYPE_TO_CATEGORY = {
    "상의": "top", "하의": "bottom", "아우터": "outerwear", "원피스": "dress",
}


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _get_done_ids() -> set[str]:
    if not OUTPUT_PATH.exists():
        return set()
    done: set[str] = set()
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["file_id"])
            except Exception:
                pass
    return done


def _load_queue() -> list[Path]:
    done = _get_done_ids()
    return [p for p in sorted(QUEUE_DIR.glob("*.jpg")) if p.stem not in done]


def _save_label(row: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_mask(editor_data, file_id: str, clothing_type: str | None) -> str:
    """
    gr.ImageEditor 출력에서 사용자가 칠한 영역을 마스크로 추출해 파일 저장.
    칠한 영역 없으면 원본 이미지 전체를 저장.
    반환: 저장된 마스킹 이미지 경로
    """
    MASKED_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    category = CLOTHING_TYPE_TO_CATEGORY.get(clothing_type or "", "unknown")
    out_path = MASKED_ARCHIVE_DIR / f"{file_id}_{category}.jpg"

    if not editor_data or not isinstance(editor_data, dict):
        return str(out_path)

    background = editor_data.get("background")
    layers     = editor_data.get("layers") or []

    if background is None:
        return str(out_path)

    bg_img = background if isinstance(background, Image.Image) else Image.fromarray(np.array(background))
    bg_img = bg_img.convert("RGB")

    if not layers:
        bg_img.save(out_path, quality=95)
        return str(out_path)

    # 첫 번째 레이어에서 칠한 영역(alpha > 0) 추출
    layer = layers[0]
    layer_np = np.array(layer) if isinstance(layer, Image.Image) else np.array(layer)

    if layer_np.ndim == 3 and layer_np.shape[2] == 4:
        mask = layer_np[:, :, 3] > 32
    elif layer_np.ndim == 3:
        mask = layer_np.sum(axis=2) > 0
    else:
        bg_img.save(out_path, quality=95)
        return str(out_path)

    if not mask.any():
        bg_img.save(out_path, quality=95)
        return str(out_path)

    bg_np = np.array(bg_img)
    # 해상도 불일치 시 마스크 리사이즈
    if mask.shape != bg_np.shape[:2]:
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (bg_np.shape[1], bg_np.shape[0]), Image.NEAREST
        )
        mask = np.array(mask_pil) > 128

    result = np.full_like(bg_np, 255)
    result[mask] = bg_np[mask]
    Image.fromarray(result).save(out_path, quality=95)
    return str(out_path)


def _compute_preview(editor_data) -> Image.Image | None:
    """마스킹 결과 미리보기 (파일 저장 없음)."""
    if not editor_data or not isinstance(editor_data, dict):
        return None
    background = editor_data.get("background")
    layers     = editor_data.get("layers") or []
    if background is None:
        return None

    bg_img = background if isinstance(background, Image.Image) else Image.fromarray(np.array(background))
    bg_img = bg_img.convert("RGB")

    if not layers:
        return bg_img

    layer    = layers[0]
    layer_np = np.array(layer) if isinstance(layer, Image.Image) else np.array(layer)

    if layer_np.ndim == 3 and layer_np.shape[2] == 4:
        mask = layer_np[:, :, 3] > 32
    elif layer_np.ndim == 3:
        mask = layer_np.sum(axis=2) > 0
    else:
        return bg_img

    if not mask.any():
        return bg_img

    bg_np = np.array(bg_img)
    if mask.shape != bg_np.shape[:2]:
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (bg_np.shape[1], bg_np.shape[0]), Image.NEAREST
        )
        mask = np.array(mask_pil) > 128

    result = np.full_like(bg_np, 255)
    result[mask] = bg_np[mask]
    return Image.fromarray(result)


# ── UI ───────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:

    def _refresh() -> dict:
        q = _load_queue()
        return {"idx": 0, "queue": q, "total_done": len(_get_done_ids())}

    def _load_current(state: dict):
        idx, q = state["idx"], state["queue"]
        total  = state["total_done"] + len(q)
        if idx >= len(q):
            return (
                gr.update(value=None),  # img_editor
                None,                   # img_preview
                f"완료! 전체 {total}개 처리됨",
                state,
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        path = q[idx]
        orig = Image.open(path).convert("RGB")
        prog = f"{state['total_done'] + idx + 1} / {total}  |  {path.stem}"
        # ImageEditor에 원본 이미지 로드 (레이어 초기화)
        return orig, None, prog, state, gr.update(interactive=True), gr.update(interactive=True)

    def _build_row(path, ctype, cat, color, scolor, mat, fit, length,
                   sleeve, neck, detail, print_, mask_q, bottom_waist, bottom_len,
                   masked_path: str) -> dict:
        row: dict = {
            "file_id":      path.stem,
            "masked_path":  masked_path,
            "타입":         ctype,
            "카테고리":     cat,
            "색상":         color,
            "서브색상":     scolor if scolor and scolor != "없음" else None,
            "소재":         mat,
            "핏":           fit,
            "기장":         length,
            "소매기장":     sleeve,
            "넥라인":       neck,
            "디테일":       detail,
            "프린트":       print_,
            "mask_quality": mask_q,
            "labeled_by":   "gradio_manual",
        }
        if ctype == "하의":
            row["bottom_waist_rise"] = bottom_waist
            row["bottom_length"]     = bottom_len
        return row

    def _reset_form():
        return (
            gr.update(value=None),              # ctype
            gr.update(choices=[], value=None),  # cat
            gr.update(value=None),              # color
            gr.update(value="없음"),            # scolor
            gr.update(value=[]),                # mat
            gr.update(value=None),              # fit
            gr.update(value=None),              # length
            gr.update(value=None),              # sleeve
            gr.update(value=None),              # neck
            gr.update(value=[]),                # detail
            gr.update(value=[]),                # print_
            gr.update(value="좋음"),            # mask_q
            gr.update(value=None),              # bottom_waist
            gr.update(value=None),              # bottom_len
            gr.update(visible=False),           # bottom_group
        )

    def _submit(
        state, editor_data, ctype, cat, color, scolor,
        mat, fit, length, sleeve, neck, detail, print_, mask_q,
        bottom_waist, bottom_len,
    ):
        idx, q = state["idx"], state["queue"]
        if idx >= len(q):
            return _load_current(state)
        path        = q[idx]
        masked_path = _extract_mask(editor_data, path.stem, ctype)
        row         = _build_row(path, ctype, cat, color, scolor, mat, fit, length,
                                 sleeve, neck, detail, print_, mask_q,
                                 bottom_waist, bottom_len, masked_path)
        _save_label(row)
        state["idx"] += 1
        return _load_current(state)

    def _add_item(
        state, editor_data, ctype, cat, color, scolor,
        mat, fit, length, sleeve, neck, detail, print_, mask_q,
        bottom_waist, bottom_len,
    ):
        """현재 항목 저장 후 같은 이미지 유지 + 폼 초기화 (다음 항목 라벨링)"""
        idx, q = state["idx"], state["queue"]
        if idx >= len(q):
            return (*_load_current(state), *_reset_form())
        path        = q[idx]
        masked_path = _extract_mask(editor_data, path.stem, ctype)
        row         = _build_row(path, ctype, cat, color, scolor, mat, fit, length,
                                 sleeve, neck, detail, print_, mask_q,
                                 bottom_waist, bottom_len, masked_path)
        _save_label(row)
        # idx 유지 — 같은 이미지, 레이어 초기화
        return (*_load_current(state), *_reset_form())

    def _skip(state: dict):
        state["idx"] += 1
        return _load_current(state)

    with gr.Blocks(title="K-Fashion 수동 라벨링", theme=gr.themes.Soft()) as app:
        gr.Markdown("# K-Fashion 수동 라벨링")
        gr.Markdown(
            "**사용법:** 왼쪽 이미지에서 라벨링할 의류 영역을 브러시로 칠하세요. "
            "여러 항목이 있으면 첫 항목 칠하기 → 라벨 입력 → **+ 항목 추가**, "
            "마지막 항목은 **저장 후 다음 이미지 →**."
        )
        state    = gr.State(_refresh())
        progress = gr.Textbox(label="진행", interactive=False)

        with gr.Row():
            # ── 이미지 패널 ──────────────────────────────────────────────
            with gr.Column(scale=1):
                img_editor = gr.ImageEditor(
                    label="의류 영역 마스킹 (브러시로 칠하기)",
                    brush=gr.Brush(
                        colors=["#FF4444"],
                        color_mode="fixed",
                        default_size=20,
                    ),
                    eraser=gr.Eraser(default_size=20),
                )
                btn_preview = gr.Button("🔍 마스킹 미리보기", variant="secondary", size="sm")
                img_preview = gr.Image(label="마스킹 결과 미리보기", type="pil", visible=True)
                mask_q      = gr.Radio(["좋음", "보통", "나쁨"], label="마스킹 품질", value="좋음")

            # ── 라벨 패널 ────────────────────────────────────────────────
            with gr.Column(scale=1):
                ctype  = gr.Radio(CLOTHING_TYPES, label="의류 타입 *")
                cat    = gr.Dropdown([], label="세부 카테고리 *")

                with gr.Row():
                    color  = gr.Dropdown(COLORS, label="색상 *")
                    scolor = gr.Dropdown(["없음"] + COLORS, label="서브색상", value="없음")

                mat    = gr.CheckboxGroup(MATERIALS, label="소재 * (복수선택)")
                fit    = gr.Radio(FITS,     label="핏")
                length = gr.Radio(LENGTHS,  label="기장")
                sleeve = gr.Radio(SLEEVES,  label="소매기장")
                neck   = gr.Dropdown(NECKLINES, label="넥라인")
                detail = gr.CheckboxGroup(DETAILS, label="디테일 (복수선택)")
                print_ = gr.CheckboxGroup(PRINTS,  label="프린트 (복수선택)")

                with gr.Group(visible=False) as bottom_group:
                    gr.Markdown("#### 하의 전용 속성")
                    bottom_waist = gr.Radio(BOTTOM_WAIST_RISE, label="허리라인 (Waist Rise)")
                    bottom_len   = gr.Radio(BOTTOM_LENGTH,     label="하의 기장")

                def _on_type_change_full(ctype_val: str):
                    return (
                        gr.update(choices=CATEGORIES.get(ctype_val, []), value=None),
                        gr.update(visible=(ctype_val == "하의")),
                    )

                ctype.change(_on_type_change_full, inputs=ctype, outputs=[cat, bottom_group])

                with gr.Row():
                    btn_skip     = gr.Button("건너뛰기",           variant="secondary")
                    btn_add_item = gr.Button("+ 항목 추가",         variant="secondary")
                    btn_submit   = gr.Button("저장 후 다음 이미지 →", variant="primary")

        # ── 이벤트 배선 ───────────────────────────────────────────────────────
        shared_outputs   = [img_editor, img_preview, progress, state, btn_submit, btn_skip]
        form_fields      = [ctype, cat, color, scolor, mat, fit, length, sleeve,
                            neck, detail, print_, mask_q, bottom_waist, bottom_len, bottom_group]
        submit_inputs    = [state, img_editor, ctype, cat, color, scolor,
                            mat, fit, length, sleeve, neck, detail, print_, mask_q,
                            bottom_waist, bottom_len]
        add_item_outputs = shared_outputs + form_fields  # 6 + 15 = 21

        def _on_load(_state):
            fresh = _refresh()
            return _load_current(fresh)

        app.load(_on_load,    inputs=state,          outputs=shared_outputs)
        btn_preview.click(_compute_preview,  inputs=[img_editor],    outputs=[img_preview])
        btn_submit.click(_submit,   inputs=submit_inputs, outputs=shared_outputs)
        btn_add_item.click(_add_item, inputs=submit_inputs, outputs=add_item_outputs)
        btn_skip.click(_skip,     inputs=state,          outputs=shared_outputs)

    return app


if __name__ == "__main__":
    build_app().launch(share=False)
