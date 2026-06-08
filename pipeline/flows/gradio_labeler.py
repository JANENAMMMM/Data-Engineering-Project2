# 출처: 신규 작성 (Gradio 공식 docs 기반)
# 라벨 옵션값: 현재 프로젝트 project_pipeline.md 기반
# 실행: python flows/gradio_labeler.py  (pipeline/ 디렉토리에서)
#        → http://localhost:7860 접속

from __future__ import annotations

import sys
import json
from pathlib import Path

# pipeline/ 디렉토리를 sys.path에 추가 (flows/ 서브디렉토리에서 실행 시 필요)
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from PIL import Image

from src.masking import apply_polygon_mask

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

# ── 폴리곤 캔버스 HTML/JS ─────────────────────────────────────────────────────
# 전략: gr.Image(elem_id="poly-source-img")를 CSS로 숨기고,
#       JS가 150ms 폴링으로 <img> src를 감지해 canvas에 로드.
#       base64 WebSocket 전달 방식(불안정)을 완전히 배제.
CANVAS_HTML = """
<style>
  /* gr.Image 소스는 DOM에 존재하되 화면에서 숨김 */
  #poly-source-img { display: none !important; }

  #poly-canvas {
    cursor: crosshair;
    border: 2px solid #e5e7eb;
    border-radius: 8px;
    max-width: 100%;
    display: block;
    background: #f9fafb;
  }
  #poly-controls {
    margin-top: 8px;
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
  }
  .poly-btn {
    padding: 5px 14px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background: white;
    cursor: pointer;
    font-size: 13px;
    font-family: sans-serif;
  }
  .poly-btn:hover { background: #f3f4f6; }
  #poly-status { color: #6b7280; font-size: 13px; font-family: sans-serif; }
</style>

<canvas id="poly-canvas" width="550" height="400"></canvas>
<div id="poly-controls">
  <button class="poly-btn" onclick="window.polyUndo()">↩ 되돌리기</button>
  <button class="poly-btn" onclick="window.polyClear()">✕ 초기화</button>
  <span id="poly-status">이미지 로딩 대기 중...</span>
</div>

<script>
(function () {
  var canvas  = document.getElementById('poly-canvas');
  var ctx     = canvas.getContext('2d');
  var pts     = [];
  var imgEl   = new Image();
  var lastSrc = '';

  function getCoordTb() {
    return document.querySelector('#poly-coords textarea');
  }

  /* ── 캔버스 재렌더 ───────────────────────────────── */
  function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imgEl.complete && imgEl.naturalWidth > 0) {
      ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);
    }
    if (!pts.length) return;

    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    if (pts.length >= 3) {
      ctx.closePath();
      ctx.fillStyle = 'rgba(239,68,68,0.15)';
      ctx.fill();
    }
    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth   = 2;
    ctx.stroke();

    pts.forEach(function(p, i) {
      ctx.beginPath();
      ctx.arc(p[0], p[1], 5, 0, 2 * Math.PI);
      ctx.fillStyle   = (i === 0) ? '#16a34a' : '#EF4444';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth   = 1.5;
      ctx.stroke();
      ctx.fillStyle       = 'white';
      ctx.font            = 'bold 9px sans-serif';
      ctx.textAlign       = 'center';
      ctx.textBaseline    = 'middle';
      ctx.fillText(String(i + 1), p[0], p[1]);
    });
  }

  /* ── 좌표 → Textbox 동기화 ──────────────────────── */
  function pushCoords() {
    var tb = getCoordTb();
    if (!tb) return;
    var fracs = pts.map(function(p) {
      return [p[0] / canvas.width, p[1] / canvas.height];
    });
    var val = JSON.stringify(fracs);
    if (tb.value === val) return;
    tb.value = val;
    tb.dispatchEvent(new Event('input',  { bubbles: true }));
    tb.dispatchEvent(new Event('change', { bubbles: true }));
  }

  /* ── 이미지 로드 ─────────────────────────────────── */
  function loadFromSrc(src) {
    if (!src || src === lastSrc) return;
    lastSrc = src;
    var img = new Image();
    img.onload = function() {
      var maxW  = Math.min(550, img.naturalWidth);
      var ratio = maxW / img.naturalWidth;
      canvas.width  = Math.round(img.naturalWidth  * ratio);
      canvas.height = Math.round(img.naturalHeight * ratio);
      imgEl = img;
      pts   = [];
      pushCoords();
      redraw();
      document.getElementById('poly-status').textContent =
        '클릭으로 꼭짓점 추가 | ↩ 되돌리기 | ✕ 초기화';
    };
    img.onerror = function() {
      document.getElementById('poly-status').textContent = '이미지 로드 실패';
    };
    img.src = src;
  }

  /* ── 클릭 ────────────────────────────────────────── */
  canvas.addEventListener('click', function(e) {
    if (e.detail > 1) return;
    var r = canvas.getBoundingClientRect();
    var x = Math.round((e.clientX - r.left) * canvas.width  / r.width);
    var y = Math.round((e.clientY - r.top)  * canvas.height / r.height);
    pts.push([x, y]);
    redraw();
    pushCoords();
    var n = pts.length;
    document.getElementById('poly-status').textContent =
      '점 ' + n + '개' + (n >= 3 ? ' ✓ 폴리곤 완성' : ' (최소 3개 필요)');
  });

  window.polyUndo = function() {
    if (pts.length) { pts.pop(); redraw(); pushCoords(); }
    document.getElementById('poly-status').textContent = '점 ' + pts.length + '개';
  };
  window.polyClear = function() {
    pts = []; redraw(); pushCoords();
    document.getElementById('poly-status').textContent = '초기화됨. 다시 클릭하세요.';
  };

  /* ── gr.Image src 폴링 (150ms) ───────────────────── */
  // gr.Image(elem_id="poly-source-img")의 <img> src를 감지해 canvas 로드.
  // visible=True + CSS display:none → DOM에 존재, 화면에만 숨김.
  setInterval(function() {
    var el = document.querySelector('#poly-source-img img');
    if (!el || !el.src) return;
    loadFromSrc(el.src);
  }, 150);
})();
</script>
"""


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


def _open_for_display(path: Path) -> Image.Image:
    """원본 이미지를 열어 canvas 표시용으로 반환. 최대 800px로 축소(비율 유지)."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > 800:
        img = img.copy()
        img.thumbnail((800, 800), Image.LANCZOS)
    return img


def _extract_mask_polygon(coords_json: str, file_id: str, img_path: Path,
                          clothing_type: str | None) -> str:
    """폴리곤 비율 좌표 → 원본 이미지에 마스킹 적용 후 저장."""
    MASKED_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    category = CLOTHING_TYPE_TO_CATEGORY.get(clothing_type or "", "unknown")
    out_path  = MASKED_ARCHIVE_DIR / f"{file_id}_{category}.jpg"

    try:
        fracs = json.loads(coords_json or "[]")
    except Exception:
        fracs = []

    img  = Image.open(img_path).convert("RGB")
    w, h = img.size  # 항상 원본 해상도 기준

    if len(fracs) >= 3:
        points = [(round(fx * w), round(fy * h)) for fx, fy in fracs]
        masked = apply_polygon_mask(img, points, crop=True)
    else:
        masked = img

    masked.save(out_path, "JPEG", quality=95)
    return str(out_path)


def _compute_preview_polygon(coords_json: str, img_path: Path | None) -> Image.Image | None:
    """폴리곤 비율 좌표 → 미리보기 PIL Image (파일 저장 없음)."""
    if img_path is None or not img_path.exists():
        return None
    try:
        fracs = json.loads(coords_json or "[]")
    except Exception:
        fracs = []

    img  = Image.open(img_path).convert("RGB")
    w, h = img.size

    if len(fracs) < 3:
        return img

    points = [(round(fx * w), round(fy * h)) for fx, fy in fracs]
    return apply_polygon_mask(img, points, crop=True)


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
                None,   # source_img: None → 이미지 없음
                None,   # img_preview
                f"완료! 전체 {total}개 처리됨",
                state,
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        path    = q[idx]
        display = _open_for_display(path)
        prog    = f"{state['total_done'] + idx + 1} / {total}  |  {path.stem}"
        return display, None, prog, state, gr.update(interactive=True), gr.update(interactive=True)

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

    def _submit(state, coords_json, ctype, cat, color, scolor,
                mat, fit, length, sleeve, neck, detail, print_, mask_q,
                bottom_waist, bottom_len):
        idx, q = state["idx"], state["queue"]
        if idx >= len(q):
            return _load_current(state)
        path        = q[idx]
        masked_path = _extract_mask_polygon(coords_json, path.stem, path, ctype)
        row         = _build_row(path, ctype, cat, color, scolor, mat, fit, length,
                                 sleeve, neck, detail, print_, mask_q,
                                 bottom_waist, bottom_len, masked_path)
        _save_label(row)
        state["idx"] += 1
        return _load_current(state)

    def _add_item(state, coords_json, ctype, cat, color, scolor,
                  mat, fit, length, sleeve, neck, detail, print_, mask_q,
                  bottom_waist, bottom_len):
        """현재 항목 저장 후 같은 이미지 유지 + 폼 초기화.
        gr.Image에 새 PIL 객체를 반환하면 Gradio가 새 temp URL 생성 → JS가 src 변화
        감지 → canvas pts 자동 초기화."""
        idx, q = state["idx"], state["queue"]
        if idx >= len(q):
            return (*_load_current(state), *_reset_form())
        path        = q[idx]
        masked_path = _extract_mask_polygon(coords_json, path.stem, path, ctype)
        row         = _build_row(path, ctype, cat, color, scolor, mat, fit, length,
                                 sleeve, neck, detail, print_, mask_q,
                                 bottom_waist, bottom_len, masked_path)
        _save_label(row)
        # idx 유지(같은 이미지). _load_current가 새 PIL 객체 반환 → 새 temp URL → canvas 초기화
        return (*_load_current(state), *_reset_form())

    def _skip(state: dict):
        state["idx"] += 1
        return _load_current(state)

    def _preview(coords_json: str, state: dict):
        idx, q = state["idx"], state["queue"]
        if idx >= len(q):
            return None
        return _compute_preview_polygon(coords_json, q[idx])

    with gr.Blocks(title="K-Fashion 수동 라벨링", theme=gr.themes.Soft()) as app:
        gr.Markdown("# K-Fashion 수동 라벨링")
        gr.Markdown(
            "**사용법:** 이미지 위 클릭으로 의류 영역 꼭짓점을 찍으세요 (3개 이상 → 폴리곤 완성). "
            "첫 점은 초록색. ↩ 되돌리기, ✕ 초기화. "
            "여러 항목은 폴리곤 + 라벨 → **+ 항목 추가**, 마지막은 **저장 후 다음 이미지 →**."
        )
        state    = gr.State(_refresh())
        progress = gr.Textbox(label="진행", interactive=False)

        # Gradio가 이미지를 temp URL로 서빙. CSS로 숨기고 JS가 src를 폴링.
        source_img = gr.Image(
            elem_id="poly-source-img",
            type="pil",
            interactive=False,
            show_label=False,
        )

        with gr.Row():
            # ── 이미지 패널 ──────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.HTML(CANVAS_HTML)

                poly_coords = gr.Textbox(
                    label="📍 폴리곤 좌표 (자동)",
                    elem_id="poly-coords",
                    interactive=True,
                    lines=1,
                    max_lines=2,
                    placeholder="이미지를 클릭하면 자동으로 채워집니다",
                )
                btn_preview = gr.Button("🔍 마스킹 미리보기", variant="secondary", size="sm")
                img_preview = gr.Image(label="마스킹 결과 미리보기", type="pil")
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

                def _on_type_change(ctype_val: str):
                    return (
                        gr.update(choices=CATEGORIES.get(ctype_val, []), value=None),
                        gr.update(visible=(ctype_val == "하의")),
                    )

                ctype.change(_on_type_change, inputs=ctype, outputs=[cat, bottom_group])

                with gr.Row():
                    btn_skip     = gr.Button("건너뛰기",            variant="secondary")
                    btn_add_item = gr.Button("+ 항목 추가",          variant="secondary")
                    btn_submit   = gr.Button("저장 후 다음 이미지 →", variant="primary")

        # ── 이벤트 배선 ───────────────────────────────────────────────────────
        shared_outputs   = [source_img, img_preview, progress, state, btn_submit, btn_skip]
        form_fields      = [ctype, cat, color, scolor, mat, fit, length, sleeve,
                            neck, detail, print_, mask_q, bottom_waist, bottom_len, bottom_group]
        submit_inputs    = [state, poly_coords, ctype, cat, color, scolor,
                            mat, fit, length, sleeve, neck, detail, print_, mask_q,
                            bottom_waist, bottom_len]
        add_item_outputs = shared_outputs + form_fields  # 6 + 15 = 21

        def _on_load(_state):
            return _load_current(_refresh())

        app.load(_on_load,        inputs=state,                  outputs=shared_outputs)
        btn_preview.click(_preview,    inputs=[poly_coords, state],  outputs=[img_preview])
        btn_submit.click(_submit,      inputs=submit_inputs,         outputs=shared_outputs)
        btn_add_item.click(_add_item,  inputs=submit_inputs,         outputs=add_item_outputs)
        btn_skip.click(_skip,          inputs=state,                 outputs=shared_outputs)

    return app


if __name__ == "__main__":
    build_app().launch(share=False)
