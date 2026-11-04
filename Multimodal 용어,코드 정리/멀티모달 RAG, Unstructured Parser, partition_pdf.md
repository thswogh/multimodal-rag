# 멀티모달 RAG, Unstructured Parser, partition_pdf

## 📌 오늘의 키워드

| 키워드 | 한 줄 정리 |
| --- | --- |
| **멀티모달 RAG** | 텍스트, 표, 이미지를 모두 검색하고 답변 근거로 쓰는 RAG |
| **Unstructured Parser** | 비정형 문서를 **표준화된 Element로 바꾸는 전처리기**. RAG의 맨 첫 단계 |
| **Element** | 파싱 결과 단위 객체 (category / text / metadata) |
| **`partition_pdf()`** | PDF를 Element 리스트로 분리하고 이미지는 jpg로 저장 |
| **`strategy="hi_res"`** | 레이아웃 모델(YOLOX)이 페이지를 보고 Title, Table, Image 영역을 탐지 |
| **`chunking_strategy="by_title"`** | Title 요소가 나올 때마다 새 청크를 시작 |
| **VLM** | 이미지를 텍스트 요약으로 바꿔 검색 가능하게 만드는 모델 |
| **Multi-Vector Retriever** | **요약으로 검색**하고 **원본으로 답변**하는 검색기 |
| **CharacterTextSplitter** | 토큰 기준 재분할 (chunk_size=2000, overlap=200) |

---

## 1. 멀티모달 RAG 개념

| 구분 | 일반 RAG | 멀티모달 RAG |
| --- | --- | --- |
| 검색 대상 | 텍스트 | 텍스트 + 표 + 이미지 |
| 필요 모델 | LLM | LLM + VLM |
| 답변 근거 | 텍스트 청크 | 원본 텍스트, 원본 표, 원본 이미지 |

### 아키텍처 3단계

1. **파싱**: Unstructured Parser로 텍스트, 표, 이미지를 분리한 뒤 각각 요약
2. **저장/검색**: 요약은 벡터DB, 원본은 docstore에 저장. 요약으로 찾고 원본을 반환
3. **생성**: 멀티모달 LLM이 원본을 보고 답변한 뒤 한국어로 번역

> 💡 **검색에는 요약, 답변에는 원본**
> - 이미지(jpg)는 그대로는 임베딩이 안 되므로 VLM으로 요약해 텍스트로 만든다.
> - 요약만 넘기면 디테일이 사라지므로 답변에는 원본을 넘긴다.

---

## 2. Unstructured Parser

### 2-1. 역할과 위치

- PDF, HTML, Word 같은 **비정형 문서를 표준화된 요소(Element)로 바꾸는 전처리기**입니다.
- RAG 파이프라인의 **맨 처음**에 씁니다. 순서가 중요해요.

```
PDF ──▶ [Unstructured Parser] ──▶ 요약 ──▶ 임베딩/저장 ──▶ 검색 ──▶ 답변
         ↑ 여기가 첫 단계
```

### 2-2. 반환값: Element 리스트

```python
raw_pdf_elements = extract_pdf_elements(fpath, fname)
# → [CompositeElement, Table, CompositeElement, ...]  (문자열이 아닌 객체 리스트)
```

Element 하나는 택배 박스 같은 객체입니다.

```
Element (예: Title 객체)
├── category  → "Title"            # 이 요소가 무엇인지 (송장의 '취급 주의' 스티커)
├── text      → "1. 사업 개요"      # 실제 내용물 (박스 안 물건)
└── metadata  → {page_number: 3,   # 부가 정보 (송장의 주소, 발송일 등)
                 filename: ...,
                 coordinates: ...}
```

```python
el.category                  # "CompositeElement" / "Table"
el.text                      # 텍스트 (str(el)과 같음)
el.metadata.page_number      # 페이지 번호
el.metadata.text_as_html     # Table만 있음. 행과 열 구조 보존 (infer_table_structure=True일 때)
el.metadata.orig_elements    # 청킹 전 원본 요소들
```

- 이미지는 리스트에 들어 있지 않습니다. `figures/` 폴더에 jpg로 저장돼요.

### 2-3. 요소 타입

| 청킹 전 (원본) | by_title 청킹 후 |
| --- | --- |
| Title, NarrativeText, ListItem, Table, Image ... | **CompositeElement** (텍스트 묶음), **Table**, TableChunk |

### 2-4. by_title은 제목을 어떻게 아나?

1. **제목 판별**: `hi_res` 레이아웃 모델이 글자 크기, 굵기, 위치 같은 시각적 특징으로 `Title` 라벨을 붙입니다. (`fast`는 텍스트 규칙으로 판별)
2. **청킹**: `Title`이 나올 때마다 새 청크를 시작합니다. `Table`은 항상 독립된 청크가 됩니다.
3. **크기 조정**: `max_characters`, `new_after_n_chars`, `combine_text_under_n_chars`로 청크 크기를 조절합니다.

> ⚠️ 제목을 잘못 잡을 수 있습니다. 예를 들어 `texts[0]`가 `"목 차"` 한 줄뿐인 청크로 나왔어요. 청크가 이상하면 청킹 없이 파싱해서 `category == "Title"` 목록부터 확인하세요.

---

## 3. 환경 구성 요약

```bash
python -m venv .venv                  # Python 3.11 또는 3.12
.venv\Scripts\activate                # mac: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt       # langchain-classic 필수 (MultiVectorRetriever)
pip freeze > requirements.lock.txt    # 버전 고정
python -m ipykernel install --user --name multimodal-rag
```

- Poppler(PDF를 이미지로 변환)와 Tesseract(OCR)는 OS에 설치합니다. Path 등록 후 **재시작**하세요.
- 경로에 **한글이나 공백이 있으면 안 됩니다** (`C:\workspaces\multimodal-rag`).
- `.env`는 Git에서 제외하고 `.env.example`만 커밋합니다.

```
multimodal-rag/
├── data/
│   ├── invest.pdf
│   ├── figures/          ← 추출 이미지
│   └── processed/        ← extracted.pkl
└── notebooks/            ← 노트북 실행 위치
```

---

## 4. 데이터 추출 및 분할

### 4-1. `partition_pdf()` 핵심 옵션

| 옵션 | 하는 일 | 빼면? |
| --- | --- | --- |
| `strategy="hi_res"` | 레이아웃 모델로 정밀 추출 | 표와 이미지를 못 찾음 |
| `extract_images_in_pdf=True` | 그림을 jpg로 저장 | 이미지가 사라짐 |
| `infer_table_structure=True` | 표의 행과 열 구조를 추론 | 표가 줄글로 뭉개짐 |
| `chunking_strategy="by_title"` | 제목 기준으로 블록화 | 잘게 쪼개진 요소가 그대로 나옴 |
| `extract_image_block_output_dir` | 이미지 저장 폴더 | |

### 4-2. 분류, 분할, 저장

- `Table`은 `tables`, `CompositeElement`는 `texts`에 넣습니다.
- 텍스트를 합친 뒤 토큰 2000 단위로 다시 자르고, 200 토큰을 겹칩니다.
- 파싱이 느리므로 결과를 `extracted.pkl`로 저장해 재사용합니다.

### 4-3. 실습 결과 (invest.pdf)

| 항목 | 결과 |
| --- | --- |
| 파싱 시간 | 약 169초 |
| 전체 요소 | 74개 (CompositeElement 39, Table 35) |
| 추출 이미지 | 6개 (`figure-페이지-번호.jpg`) |
| 분할 후 청크 | 39 → 27개 |

### 💬 생각해 볼 질문

| 질문 | 답 |
| --- | --- |
| hi_res가 없으면? | fast는 글자만 뽑기 때문에 표와 그림의 **시각적 영역**을 못 찾음 |
| 왜 overlap을 두나? | 경계에서 잘린 문장이 한쪽 청크에는 온전히 들어가도록 해서 정보 손실을 막음 |
| 이미지는 왜 아직 검색이 안 되나? | jpg는 임베딩이 안 됨. VLM으로 텍스트 요약을 만든 뒤 임베딩해야 함 |


## 🔁 전체 흐름 한눈에 보기

```python
# =========================================================
# 0. 임포트 + 환경 점검
# =========================================================
import os, sys, time, shutil, pickle
from pathlib import Path
from collections import Counter

from unstructured.partition.pdf import partition_pdf          # Unstructured Parser (PDF용)
from langchain_text_splitters import CharacterTextSplitter    # 텍스트 재분할

print("venv 사용 중?", ".venv" in sys.prefix)                  # 가상환경 커널인지 확인
print("Poppler  :", "OK" if shutil.which("pdftoppm") else "✗")  # PDF → 이미지 변환
print("Tesseract:", "OK" if shutil.which("tesseract") else "✗") # OCR

# =========================================================
# 1. 경로 설정 (노트북은 notebooks/ 안에서 실행)
# =========================================================
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
fname = "invest.pdf"
fpath = str(PROJECT_ROOT / "data") + os.sep                   # path + fname 형태로 쓰기 위해 끝에 구분자
print("PDF 존재 여부:", os.path.exists(fpath + fname))         # True여야 진행

# =========================================================
# 2. 파싱 함수: PDF → Element 리스트 (+ 이미지는 jpg로 저장)
#    RAG의 맨 첫 단계. 이후 모든 단계가 이 결과를 사용
# =========================================================
def extract_pdf_elements(path, fname):
    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",                 # 레이아웃 모델이 Title / Table / Image 영역 탐지
        extract_images_in_pdf=True,        # 그림을 jpg로 저장
        infer_table_structure=True,        # 표 구조 추론 → metadata.text_as_html 생성
        chunking_strategy="by_title",      # Title이 나올 때마다 새 청크 시작
        max_characters=4000,               # 청크 최대 4000자 (넘으면 강제 분할)
        new_after_n_chars=3800,            # 3800자를 넘으면 다음 요소부터 새 청크
        combine_text_under_n_chars=2000,   # 짧은 섹션은 다음 섹션과 합치기
        extract_image_block_output_dir=os.path.join(path, "figures"),  # data/figures에 저장
    )

# =========================================================
# 3. 분류 함수: Element의 타입으로 표 / 텍스트 구분
#    Element는 객체이므로 str()로 텍스트를 꺼냄 (= el.text)
# =========================================================
def categorize_elements(raw_pdf_elements):
    tables, texts = [], []
    for el in raw_pdf_elements:
        t = str(type(el))
        if "unstructured.documents.elements.Table" in t:
            tables.append(str(el))                   # 표 → 나중에 표 요약
        elif "unstructured.documents.elements.CompositeElement" in t:
            texts.append(str(el))                    # 텍스트 묶음 → 분할 후 요약
    return texts, tables

# =========================================================
# 4. 실행: 파싱(2~3분) → Element 구조 확인 → 분류 → 집계
# =========================================================
start = time.time()
raw_pdf_elements = extract_pdf_elements(fpath, fname)          # Element 객체 리스트
print(f"추출 완료 {time.time() - start:.1f}초, 요소 {len(raw_pdf_elements)}개")

el = raw_pdf_elements[0]                                        # Element 하나 들여다보기
print(el.category, "| p.", el.metadata.page_number, "|", el.text[:40])

texts, tables = categorize_elements(raw_pdf_elements)
print(Counter(e.category for e in raw_pdf_elements))           # 예: CompositeElement 39, Table 35

# =========================================================
# 5. 이미지 확인: Element 리스트가 아니라 figures 폴더에 있음
# =========================================================
figures_dir = Path(fpath) / "figures"
image_files = sorted(f for f in figures_dir.iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
print("추출 이미지:", len(image_files))                         # figure-페이지-번호.jpg

# =========================================================
# 6. 텍스트 재분할: 토큰 2000 단위, 200 겹침
# =========================================================
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000,       # 청크 최대 크기 (토큰)
    chunk_overlap=200,     # 경계에서 의미가 끊기지 않게 겹침
)
texts_2k_token = text_splitter.split_text(" ".join(texts))     # 전체를 합친 뒤 다시 자름
print("원본:", len(texts), "→ 분할 후:", len(texts_2k_token))

# =========================================================
# 7. 저장: 파싱이 느리므로 pickle로 보관해 다음 단계에서 재사용
# =========================================================
output_dir = PROJECT_ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "extracted.pkl", "wb") as f:
    pickle.dump({"texts": texts, "tables": tables,
                 "texts_2k_token": texts_2k_token}, f)

# =========================================================
# 다음 단계
#   texts_2k_token, tables → LLM 요약
#   figures/*.jpg          → VLM 이미지 요약
#   요약은 벡터DB, 원본은 docstore → Multi-Vector Retriever
#   질문 → 원본 검색 → 멀티모달 LLM 답변 → 한국어 번역
# =========================================================
```