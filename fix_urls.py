import json
from pathlib import Path
from urllib.parse import unquote, quote

STYLE_MAP = {
    "아방가르드":       "avant_garde",
    "클래식":          "classic",
    "컨트리":          "country",
    "기타":            "etc",
    "페미닌":          "feminine",
    "젠더리스":        "genderless",
    "힙합":            "hiphop",
    "히피":            "hippie",
    "키치":            "kitsch",
    "매니시":          "mannish",
    "밀리터리":        "military",
    "모던":            "modern",
    "오리엔탈":        "oriental",
    "프레피":          "preppy",
    "펑크":            "punk",
    "리조트":          "resort",
    "레트로":          "retro",
    "로맨틱":          "romantic",
    "섹시":            "sexy",
    "소피스트케이티드": "sophisticated",
    "스포티":          "sporty",
    "스트리트":        "street",
    "톰보이":          "tomboy",
    "웨스턴":          "western",
}

def fix_url(url: str) -> str:
    decoded = unquote(url)
    for ko, en in STYLE_MAP.items():
        if f"/image/{ko}/" in decoded:
            return decoded.replace(f"/image/{ko}/", f"/image/{en}/")
    return decoded  # 이미 영문이거나 매핑 없으면 그대로

def fix_tasks_file(src: Path, dst: Path = None):
    if dst is None:
        dst = src
    with open(src, encoding="utf-8") as f:
        tasks = json.load(f)

    changed = 0
    for task in tasks:
        old_url = task["data"]["image"]
        new_url = fix_url(old_url)
        if old_url != new_url:
            task["data"]["image"] = new_url
            changed += 1

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"{src.name}: {changed}/{len(tasks)}개 URL 수정 → {dst.name}")

if __name__ == "__main__":
    project_dir = Path(__file__).parent
    targets = list(project_dir.glob("tasks*.json"))
    if not targets:
        print("tasks*.json 파일이 없습니다.")
    for p in targets:
        fix_tasks_file(p)
