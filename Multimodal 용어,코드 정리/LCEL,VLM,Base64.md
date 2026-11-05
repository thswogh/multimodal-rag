# LCEL,VLM,Base64

> 실습 ② 요약 생성 — 텍스트·표·이미지를 **검색용 요약문(색인 카드)** 으로 바꾸기

```
텍스트 ──┐
  표   ──┼──▶  요약문 생성 (LLM / VLM)  ──▶  검색용 색인
이미지 ──┘
```

---

## 📌 오늘의 키워드

| 키워드 | 한 줄 정리 |
| --- | --- |
| **python-dotenv** | `.env`에서 API 키 로드, 출력은 앞 7자 + 뒤 4자만 |
| **pickle** | 파이썬 객체 냉동(`dump`) / 해동(`load`) → 비싼 작업 결과 재사용 |
| **LCEL 파이프 `\|`** | 컨베이어 벨트: 왼쪽 결과가 오른쪽 입력 |
| **ChatPromptTemplate** | `{element}` 자리표시자가 있는 프롬프트 템플릿 |
| **StrOutputParser** | `AIMessage` → 순수 문자열 |
| **`.invoke` vs `.batch`** | 1건 vs 여러 건 동시 처리 |
| **max_concurrency** | 동시 요청 수 제한, 너무 높으면 `RateLimitError` |
| **temperature=0** | 일관된 출력 (요약은 창의성보다 일관성) |
| **압축률** | 요약 길이 ÷ 원본 길이 |
| **VLM** | Vision Language Model, 이미지를 읽는 모델 |
| **Base64** | 이미지(바이너리) → 문자열 포장 (API는 문자열만 주고받음) |
| **HumanMessage (멀티모달)** | `content` 리스트에 text 조각 + image_url 조각 |
| **육안 검증** | 요약이 틀리면 검색 자체가 틀어짐 → 원본과 대조 |
| **tiktoken** | 토큰 수 로컬 계산, 비용 = 토큰 수 × 단가 |

---

## 1. 준비 — API 키 & 지난 결과 복원

- 키는 코드에 쓰지 않고 `.env` → `load_dotenv` → `os.getenv`
- `OpenAI().models.list()` 호출로 키 유효성 확인 (`AuthenticationError`)
- PDF 추출(`hi_res`)은 오래 걸리니 지난 시간 저장한 `extracted.pkl`을 불러옴

```python
with open(PROJECT_ROOT / "data/processed/extracted.pkl", "rb") as f:
    data = pickle.load(f)
texts_2k_token, tables = data["texts_2k_token"], data["tables"]
```

## 2. 텍스트·표 요약 (LLM)

**LCEL 체인 4칸**

| 칸 | 받는 것 | 내보내는 것 |
| --- | --- | --- |
| `{"element": lambda x: x}` | 문자열 | `{"element": 문자열}` |
| `prompt` | 딕셔너리 | 완성된 프롬프트 |
| `model` | 프롬프트 | `AIMessage` |
| `StrOutputParser()` | `AIMessage` | 문자열 |

```python
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 2})
```

- 결과: 텍스트 27건 / 표 35건 요약 (약 60초)
- `table_summaries`가 비어 있다면 → ① 표가 그림으로 그려짐(이미지로 추출됨) ② `strategy="hi_res"` 누락

## 3. 이미지 요약 (VLM)

- 흐름: `open("rb")` → `base64.b64encode()` (bytes) → `.decode("utf-8")` (str)
- 메시지 구조: 하나의 `HumanMessage`에 **텍스트 + 이미지**를 같이 넣음
- URL 형식: `data:image/jpeg;base64,<문자열>`
- `gpt-4-vision-preview`는 지원 종료(404) → `gpt-4o-mini` 사용
- 프롬프트 팁: **지시는 영어, 결과는 한국어**로 ("written in Korean (Hangul)")
- 이미지마다 **요약문(검색용) + Base64 원본(답변 생성용)** 두 개를 저장, **순서가 같아야** 짝지을 수 있음

## 4. 요약 품질 검증 ⭐

| 수준 | 예시 | 검색 가능성 |
| --- | --- | --- |
| 나쁨 | "차트 이미지입니다" | 거의 불가능 |
| 보통 | "매출 관련 그래프" | 낮음 |
| 좋음 | "2021~2023년 분기별 매출 추이 선그래프, 하반기 하락세" | 높음 |

- 내 결과: **보통** — 수치·축 이름은 잘 뽑았지만 "추가 질문 있으신가요?" 같은 챗봇식 문장이 섞임
- 개선 프롬프트: "이미지 속 용어 그대로, 추측 금지", "인사말·질문 없이 요약문만", "축 이름·수치 범위·추세 반드시 포함"

## 5. 비용 측정 & 저장

- `tiktoken.encoding_for_model("gpt-4o-mini")` → 입력 15,155 / 출력 4,916 토큰
- 이미지 토큰은 해상도에 따라 달라 계산에서 제외
- 결과는 `summaries.pkl`로 저장 → 다음 시간(벡터 저장소) 재사용

## 6. 생각해 볼 질문

- **이미지를 왜 텍스트로 요약?** → 사용자는 텍스트로 질문 → 텍스트 임베딩으로 검색하려면 텍스트가 필요
- **요약만 남기고 원본을 버리면?** → 요약은 *검색용*, LLM에 보내는 건 *원본*. 세부 수치·정보 소실
- **max_concurrency=100?** → 분당 요청 한도 초과 → `RateLimitError`

## 7. 자주 나는 에러

| 증상 | 해결 |
| --- | --- |
| `AuthenticationError` | `.env` 키 오타·공백 확인, `load_dotenv()` 호출 |
| `404 model not found` | `gpt-4o-mini` 사용 |
| `RateLimitError` | `max_concurrency` 낮추기 |
| `FileNotFoundError: extracted.pkl` | 지난 노트북 저장 셀 실행 |
| 요약이 영어로 나옴 | 프롬프트에 한국어 지시 추가 |

> ⏭️ 다음: 요약문을 임베딩해 **Chroma**에 넣고, 요약으로 검색 → **원본**을 돌려주는 **멀티 벡터 검색기**

---

## 🧭 전체 흐름 한눈에 보기

```python
# ============================================================
# 0. 준비: 라이브러리 + API 키
# ============================================================
import os, time, pickle, base64
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path.cwd()
load_dotenv(PROJECT_ROOT / ".env")                 # .env 의 OPENAI_API_KEY 를 환경변수로 등록
key = os.getenv("OPENAI_API_KEY")
if not key:
    raise ValueError("OPENAI_API_KEY 없음")
print(f"API Key: {key[:7]}...{key[-4:]}")          # 키 전체 출력 금지 (유출 방지)

# ============================================================
# 1. 지난 시간 추출 결과 복원 (pickle = 해동)
# ============================================================
with open(PROJECT_ROOT / "data/processed/extracted.pkl", "rb") as f:
    data = pickle.load(f)
texts_2k_token = data["texts_2k_token"]            # 약 2k 토큰 단위로 묶은 텍스트 청크
tables         = data["tables"]                    # 표 (HTML/텍스트)
figures_dir    = PROJECT_ROOT / "data/figures"     # 추출해 둔 이미지 폴더

# ============================================================
# 2. 텍스트·표 요약 체인 (LCEL)
# ============================================================
prompt = ChatPromptTemplate.from_template(
    "표와 텍스트를 검색용으로 요약하는 도우미입니다. 요약은 임베딩되어 원본 검색에 쓰입니다. "
    "검색에 최적화된 간결한 요약을 작성하세요. 요약할 표 또는 텍스트: {element}"
)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)   # temperature=0 → 일관된 요약

# 컨베이어 벨트: 문자열 → {"element": 문자열} → 프롬프트 → AIMessage → 문자열
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# .batch() 로 여러 건 동시 처리, max_concurrency 로 동시 요청 수 제한 (RateLimitError 방지)
text_summaries  = summarize_chain.batch(texts_2k_token, {"max_concurrency": 2})
table_summaries = summarize_chain.batch(tables,         {"max_concurrency": 2})

# 압축률 확인: 너무 높으면 요약 안 됨, 너무 낮으면 정보 손실
print(f"압축률: {len(text_summaries[0]) / len(texts_2k_token[0]) * 100:.1f}%")

# ============================================================
# 3. 이미지 요약 (VLM)
# ============================================================
def encode_image(path):
    # 이미지(바이너리) → Base64 bytes → utf-8 문자열 (API는 문자열만 전송 가능)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def image_summarize(img_b64, prompt_text):
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)   # gpt-4-vision-preview 는 지원 종료
    msg = chat.invoke([
        HumanMessage(content=[                                  # 멀티모달: 텍스트 + 이미지 한 메시지에
            {"type": "text", "text": prompt_text},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ])
    ])
    return msg.content

img_prompt = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
Use the exact terms in the image, do not guess. Output only the summary, no greetings or questions.
The summary should be written in Korean (Hangul)."""     # 지시는 영어, 결과는 한국어

img_base64_list, image_summaries = [], []
for img_file in sorted(figures_dir.iterdir()):
    if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
        b64 = encode_image(img_file)
        img_base64_list.append(b64)                            # 원본 → 답변 생성 때 VLM 에 다시 전달
        image_summaries.append(image_summarize(b64, img_prompt))  # 요약 → 검색용 (두 리스트 순서 동일!)

# ============================================================
# 4. 품질 검증 (육안) — 이미지와 요약을 나란히
# ============================================================
from IPython.display import Image, display
for img_file, summary in zip(sorted(figures_dir.glob("*.jpg"))[:3], image_summaries[:3]):
    display(Image(filename=str(img_file), width=450))
    print(summary)                                             # 축 이름·수치·추세가 들어있는지 확인

# ============================================================
# 5. 토큰(비용) 측정
# ============================================================
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o-mini")
in_tok  = sum(len(enc.encode(t)) for t in texts_2k_token)
out_tok = sum(len(enc.encode(s)) for s in text_summaries)
print(f"입력 {in_tok:,} / 출력 {out_tok:,} 토큰")              # 비용 = 토큰 수 × 단가 (이미지 토큰 제외)

# ============================================================
# 6. 저장 (pickle = 냉동) → 다음 시간 벡터 저장소에서 사용
# ============================================================
out_dir = PROJECT_ROOT / "data/processed"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "summaries.pkl", "wb") as f:
    pickle.dump({
        "text_summaries":  text_summaries,
        "table_summaries": table_summaries,
        "image_summaries": image_summaries,
        "img_base64_list": img_base64_list,
    }, f)
```