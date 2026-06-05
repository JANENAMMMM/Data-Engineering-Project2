# 출처: 신규 작성 (Gradio 공식 docs 기반)
# 라벨 옵션값: 현재 프로젝트 project_pipeline.md 기반
# 실행: python flows/gradio_labeler.py
#        → http://localhost:7860 접속

from __future__ import annotations

import io
import json
from pathlib import Path

import gradio as gr
from PIL import Image

# ── 라벨 옵션 (현재 프로젝트 project_pipeline.md 기반) ────────────────────
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

FITS     = ["노멀", "루즈", "오버핏", "슬림", "스키니", "레귤러"]
LENGTHS  = ["크롭", "레귤러", "롱", "미디", "맥시"]
SLEEVES  = ["민소매", "반팔", "7부소매", "긴소매"]
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

QUEUE_DIR   = Path("data/inbox/unlabeled")
OUTPUT_PATH = Path("output/manual_labels.jsonl")


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


def _auto_mask(img_path: Path) -> Image.Image:
    """출처: 신규 작성 (rembg 라이브러리) — 폴리곤 없이 DNN 배경 제거"""
    try:
        from rembg import remove
        removed = remove(img_path.read_bytes())
        fg      = Image.open(io.BytesIO(removed)).convert("RGBA")
        bg      = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        bg.paste(fg, mask=fg.split()[3])
        return bg.convert("RGB")
    except ImportError:
        # rembg 미설치 시 원본 반환
        return Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"rembg 실패 ({img_path.name}): {e}")
        return Image.open(img_path).convert("RGB")


def _save_label(row: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
                None, None,
                f"완료! 전체 {total}개 처리됨",
                state,
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        path   = q[idx]
        orig   = Image.open(path)
        masked = _auto_mask(path)
        prog   = f"{state['total_done'] + idx + 1} / {total}  |  {path.stem}"
        return orig, masked, prog, state, gr.update(interactive=True), gr.update(interactive=True)

    def _on_type_change(ctype: str):
        return gr.update(choices=CATEGORIES.get(ctype, []), value=None)

    def _submit(
        state, ctype, cat, color, scolor,
        mat, fit, length, sleeve, neck, detail, print_, mask_q,
    ):
        idx, q = state["idx"], state["queue"]
        if idx >= len(q):
            return _load_current(state)
        path = q[idx]
        _save_label({
            "file_id":    path.stem,
            "타입":       ctype,
            "카테고리":   cat,
            "색상":       color,
            "서브색상":   scolor if scolor and scolor != "없음" else None,
            "소재":       mat,
            "핏":         fit,
            "기장":       length,
            "소매기장":   sleeve,
            "넥라인":     neck,
            "디테일":     detail,
            "프린트":     print_,
            "mask_quality": mask_q,
            "labeled_by": "gradio_manual",
        })
        state["idx"] += 1
        return _load_current(state)

    def _skip(state: dict):
        state["idx"] += 1
        return _load_current(state)

    outputs_def = [
        "img_orig", "img_masked", "progress",
        "state", "btn_submit", "btn_skip",
    ]

    with gr.Blocks(title="K-Fashion 수동 라벨링", theme=gr.themes.Soft()) as app:
        gr.Markdown("# K-Fashion 수동 라벨링")
        state    = gr.State(_refresh())
        progress = gr.Textbox(label="진행", interactive=False)

        with gr.Row():
            # ── 이미지 패널 ──────────────────────────────────────────────
            with gr.Column(scale=1):
                img_masked = gr.Image(label="마스킹 이미지 (rembg 자동)", type="pil")
                img_orig   = gr.Image(label="원본 이미지", type="pil", visible=False)
                show_orig  = gr.Checkbox(label="원본 보기", value=False)
                mask_q     = gr.Radio(
                    ["좋음", "보통", "나쁨"], label="마스킹 품질", value="좋음"
                )
                show_orig.change(
                    lambda v: [gr.update(visible=not v), gr.update(visible=v)],
                    inputs=show_orig,
                    outputs=[img_masked, img_orig],
                )

            # ── 라벨 패널 ────────────────────────────────────────────────
            with gr.Column(scale=1):
                ctype  = gr.Radio(CLOTHING_TYPES, label="의류 타입 *")
                cat    = gr.Dropdown([], label="세부 카테고리 *")
                ctype.change(_on_type_change, inputs=ctype, outputs=cat)

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

                with gr.Row():
                    btn_skip   = gr.Button("건너뛰기",    variant="secondary")
                    btn_submit = gr.Button("저장 후 다음 →", variant="primary")

        shared_outputs = [img_orig, img_masked, progress, state, btn_submit, btn_skip]
        submit_inputs  = [
            state, ctype, cat, color, scolor,
            mat, fit, length, sleeve, neck, detail, print_, mask_q,
        ]

        app.load(_load_current, inputs=state, outputs=shared_outputs)
        btn_submit.click(_submit, inputs=submit_inputs, outputs=shared_outputs)
        btn_skip.click(_skip,    inputs=state,          outputs=shared_outputs)

    return app


if __name__ == "__main__":
    build_app().launch(share=False)  # share=True 하면 외부 공개 URL 생성
