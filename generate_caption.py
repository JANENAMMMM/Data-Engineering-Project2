import os
import json
import time
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.environ["GEMINI_API_KEY"]
MODEL   = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = """Role: 
당신은 한국 이커머스 패션 플랫폼의 전문 카탈로그 데이터 구축 모델입니다.

Task: 
입력된 패션 이미지를 분석하여 객관적인 물리적 속성과 주관적인 감성/TPO 태그를 철저히 분리하여 JSON 형식으로 추출하십시오.

Constraints:
1. dense_caption (고밀도 물리 캡션 및 특이점 포착):
- 이미지 속 의류의 시각적 특징만 건조한 명사구 형태로 압축하여 나열하십시오. 절대 완성된 문장으로 쓰지 마십시오. (금지어: "아름다운", "여유로운", "입고 있는", "보이는", "배경")
- [기본 속성]: 전체적인 카테고리, 핏, 주 색상, 소재를 먼저 서술하십시오.
- [마이크로 디테일 (필수 추론)]: 일반적인 옷과 구별되는 '특이점'을 반드시 찾아내어 패션 전문 용어로 묘사하십시오. 아래 목록을 기준으로 이미지를 스캔하십시오.
  * 마감 및 봉제: 스티치 배색(예: 화이트 스티치, 굵은 스티치), 컷오프(올 풀림), 비대칭 컷팅, 핀턱/플리츠 위치.
  * 부자재 및 장식: 리벳/스터드 장식, 버튼의 형태와 색상(예: 금장 버튼, 뿔테 단추), 지퍼 디테일(예: 투웨이 지퍼, 사선 지퍼), 스트링/드로우코드, D링.
  * 패턴 및 가공: 워싱 기법(예: 캣브러쉬, 스톤워싱, 샌드워싱), 데미지/디스트로이드(파열) 위치, 타이다이, 그라데이션.
  * 구조적 변형: 컷아웃(파임), 드롭 숄더, 크롭 기장, 레이어드(겹쳐짐) 디테일, 카고 포켓 등 입체 포켓의 위치.
  * 로고 및 그래픽: 자수, 프린팅, 아플리케, 패치워크 등의 기법과 위치(예: 좌측 가슴 스몰 로고 자수).

2. mood_and_tpo (감성 및 상황 태그):
- 사용자가 해당 옷을 찾기 위해 입력할 만한 주관적, 감성적 무드 및 TPO 키워드를 3~5개의 단어로 추출하십시오. (예: "바캉스", "미니멀", "고프코어", "Y2K", "하객룩")

Output Format:
반드시 아래의 JSON 스키마 구조만 반환하십시오. 마크다운 백틱(```)이나 추가적인 텍스트는 절대 포함하지 마십시오.

{
  "dense_caption": "스트링 텍스트",
  "mood_and_tpo": ["태그1", "태그2", "태그3"]
}"""

client = genai.Client(api_key=API_KEY)


def _fetch_image_bytes(url: str) -> tuple[bytes, str]:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    mime = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
    return resp.content, mime


def generate_caption(image_url: str, retries: int = 3) -> dict:
    """
    이미지 URL을 받아 Gemini로 dense_caption과 mood_and_tpo를 생성합니다.
    반환값: {"dense_caption": str, "mood_and_tpo": list[str]}
    """
    img_bytes, mime_type = _fetch_image_bytes(image_url)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                    SYSTEM_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text)

        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트에서 추출 시도
            text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            return json.loads(text)

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
            else:
                raise e


def batch_generate(tasks_path: str, out_path: str, limit: int = None):
    """
    tasks_final.json을 읽어 각 task에 caption을 추가하고 저장합니다.
    limit: 테스트용 처리 개수 제한 (None이면 전체)
    """
    with open(tasks_path, encoding="utf-8") as f:
        tasks = json.load(f)

    if limit:
        tasks = tasks[:limit]

    results = []
    for i, task in enumerate(tasks):
        url = task["data"]["image"]
        try:
            caption = generate_caption(url)
            task["data"]["dense_caption"] = caption.get("dense_caption", "")
            task["data"]["mood_and_tpo"]  = caption.get("mood_and_tpo", [])
            print(f"[{i+1}/{len(tasks)}] OK  {url.split('/')[-1]}")
        except Exception as e:
            task["data"]["dense_caption"] = ""
            task["data"]["mood_and_tpo"]  = []
            print(f"[{i+1}/{len(tasks)}] ERR {url.split('/')[-1]} — {e}")

        results.append(task)
        time.sleep(0.5)  # rate limit 여유

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n→ {out_path} 저장 완료 ({len(results)}개)")


if __name__ == "__main__":
    # 단일 테스트
    test_url = "https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev/image/modern/1106011.jpg"
    result = generate_caption(test_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
