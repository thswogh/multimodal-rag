# 멀티모달 RAG 시스템

PDF의 **텍스트·표·이미지**를 따로 추출해서 각각 요약하고, **요약으로 검색한 뒤 원본으로 답변을 생성**하는 멀티모달 RAG 프로젝트입니다.

> 일반 RAG가 글자만 읽는 사서라면, 멀티모달 RAG는 그림과 표도 볼 줄 아는 사서입니다.

## 문제 정의

이 프로젝트에서 다룬 문서는 KOSPI200 지수를 기초자산으로 하는 투자설명서(`invest.pdf`, 39쪽)입니다. 투자자가 꼭 알아야 할 정보 상당수가 **본문 글이 아니라 표와 그림 안에** 들어 있습니다.


텍스트만 읽는 일반 RAG는 이런 정보를 검색 대상에서 놓칩니다. <br/> 실제로 "상환 차수별 횟수와 빈도율은?"이라고 물었을 때, 텍스트 전용 RAG는 **자료에 정보가 없다**고 답했습니다. <br/> 같은 질문에 멀티모달 RAG는 그림을 찾아 **1차 5,009회**라고 정확히 답했습니다. 아래 [Before/After 비교](#before텍스트rag--after멀티모달rag-비교)에 결과를 정리했습니다.

**해결하려는 문제:** 표와 그림에만 있는 정보까지 검색하고, 그 원본을 근거로 답하는 질의응답 시스템을 만드는 것입니다.

## 일반 RAG vs 멀티모달 RAG

| 구분 | 일반 RAG | 멀티모달 RAG |
|---|---|---|
| 검색 대상 | 텍스트만 | 텍스트 + 이미지 + 표 |
| 버려지는 데이터 | 그래프, 사진, 표 구조 | 없음 |
| 답변 근거 | 텍스트 청크 | 텍스트 청크 + 원본 이미지 + 원본 표 |
| 필요한 모델 | LLM | LLM + VLM(비전 모델) |
| 이 문서에서 결과 | 그림 속 상환 빈도를 물으면 "정보 없음" | 그림을 찾아 97.21%로 정확히 답변 |

### 멀티모달 RAG를 쓰는 이유

1. **정보 손실이 없다.** 투자설명서처럼 핵심 수치가 표와 그래프에 있는 문서는 텍스트만 추출하면 답할 근거 자체가 사라집니다.
2. **이미지도 검색할 수 있다.** 이미지는 그대로는 임베딩해서 검색하기 어렵습니다. VLM으로 텍스트 요약을 만들면 질문과 비교해 찾을 수 있습니다.
3. **답변은 원본으로 정확하게 만든다.** 검색은 요약으로 하지만, 답변을 만들 때는 원본 표와 이미지를 LLM에 넘겨서 요약 과정에서 빠진 세부 수치까지 활용합니다.
4. **근거를 확인할 수 있다.** 답변마다 어떤 종류의 원본(이미지/표/텍스트)을 근거로 했는지 확인할 수 있습니다. 아래 [질의응답 5건](#질의응답-5건)에서 질문별로 정리했습니다.

## 아키텍처

```
PDF ─▶ ① 파싱 (partition_pdf)
         ├─ 텍스트 → 분할 → 요약 ─┐
         ├─ 표     → 요약 ────────┼─▶ ② 요약은 벡터DB에 임베딩, 원본은 문서 저장소에 저장
         └─ 이미지 → VLM 요약 ────┘          │
                                             ▼
                     질문 ─▶ 요약으로 검색 ─▶ 원본(텍스트·표·이미지) 반환
                                             │
                                             ▼
                              ③ 멀티모달 LLM이 답변 생성 ─▶ 한국어 번역
```

1. **파싱**: Unstructured의 `partition_pdf()`로 PDF를 텍스트·표·이미지 요소로 나눕니다.
2. **저장과 검색**: 요약은 Chroma에 임베딩하고, 원본은 문서 저장소에 같은 ID로 저장합니다. 검색은 요약으로 하고, 결과로는 원본을 돌려줍니다. 도서관의 색인 카드와 같은 방식입니다.
3. **답변 생성**: 검색으로 찾은 원본 텍스트·표·이미지를 멀티모달 LLM에 넘겨 답변을 만듭니다.

**왜 이미지를 요약하나요?** 이미지는 그대로는 검색할 수 없습니다. 텍스트 요약으로 바꿔야 임베딩해서 검색할 수 있습니다.

**요약만 LLM에 주면 안 되나요?** 요약하면 디테일이 사라집니다. 그래서 검색에는 요약을 쓰고, 답변 생성에는 원본을 씁니다.

## 프로젝트 구조

```
multimodal-rag/
├── .env.example            # 필요한 환경변수 템플릿 (.env는 Git 제외)
├── requirements.txt        # 의존성 목록 (사람이 관리)
├── requirements.lock.txt   # 버전 고정 목록 (기계가 재현)
├── data/
│   ├── invest.pdf          # 원본 PDF (Git 제외)
│   └── figures/            # 추출된 이미지 (Git 제외)
├── notebooks/
│   └── Multi_modal_RAG.ipynb
└── src/
    ├── extract.py          # PDF 파싱
    ├── summarize.py        # 요약 체인
    └── retriever.py        # 다중 벡터 검색기
```

## Before(텍스트RAG) / After(멀티모달RAG) 비교
![alt text](image.png)
텍스트 RAG에는 질문 내용이 포함되지 않아 LLM이 모른다고 답변하였지만, 
멀티모달RAG는 표, 이미지에서 찾아 답변을 생성해 주었따.

## 질의응답 5건

| # | 질문 | 답변 | 근거가 된 원본 종류 |
|---|---|---|---|
| 1 | 과거 20년 시뮬레이션에서 1차 자동조기상환으로 상환된 비율은 몇 %인가요? | 자료에 1차 자동조기상환 비율 정보가 없어 알 수 없다고 답변 | 이미지, 표/텍스트 |
| 2 | 2차 자동조기상환평가일은 언제이고, 조건을 충족하면 액면금액의 몇 %를 받나요? | 2027년 11월 12일이며, 조건 충족 시 액면금액의 106.10%를 받습니다. | 표/텍스트 |
| 3 | 만기평가가격이 최초기준가격의 80% 미만이면 만기에 얼마를 돌려받나요? | 액면금액의 100.00%를 지급받습니다. | 표/텍스트 |
| 4 | 중도상환을 신청하면 공정가액 결정과 대금 지급은 각각 언제 이루어지나요? | 공정가액은 신청일 익거래소영업일에 결정되고, 대금은 신청일 이후 2영업일에 지급됩니다. | 표/텍스트 |
| 5 | 이 상품에 투자할 때 원금 손실이 생길 수 있는 요인은 무엇인가요? | 예금자 보호 대상이 아닌 금융투자상품이라는 점, 약정 조건 미충족, 기초자산 가격 변동 등 | 표/텍스트 |

- 2~5번은 표와 텍스트 원본을 근거로 정확한 날짜와 비율을 답했다.
- 1번은 검색 결과에 해당 이미지가 들어 있었지만, LLM이 그림 속 수치(1차 상환 97.21%)를 읽지 못하고 모른다고 답했다.
## 실행 방법

### 1. OS 프로그램 설치 (pip로 설치되지 않음)

| 도구 | 역할 | macOS | Windows |
|---|---|---|---|
| Poppler | PDF를 이미지로 변환 | `brew install poppler` | 압축 해제 후 `Library\bin`을 Path에 등록 |
| Tesseract | 이미지 속 글자 인식 (OCR) | `brew install tesseract` | 설치 후 설치 폴더를 Path에 등록 |

```bash
pdftoppm -v
tesseract -v
```

> Path를 등록한 뒤에는 터미널과 에디터를 완전히 껐다가 다시 켜야 합니다.
> Windows에서는 프로젝트 경로에 한글이나 공백이 있으면 파싱 단계에서 오류가 납니다.

### 2. 가상환경과 패키지 설치 (Python 3.11 또는 3.12)

```bash
python -m venv .venv
```

| 환경 | 활성화 명령 |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| Git Bash | `source .venv/Scripts/activate` |

```bash
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
```

- `requirements.txt`: 사람이 읽고 관리하는 목록 (버전 범위)
- `requirements.lock.txt`: 동작을 확인한 버전 그대로 고정한 목록 (`pip freeze`로 생성)

> PowerShell에서 스크립트 실행이 막히면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 실행하세요.
> 설치는 반드시 프롬프트 앞에 `(.venv)`가 보이는 상태에서 하세요. `unstructured[all-docs]`는 설치에 5~15분 걸립니다.

### 3. 환경변수 설정

```bash
cp .env.example .env   # 그다음 .env에 실제 OPENAI_API_KEY를 입력
```

### 4. 노트북 커널 등록과 실행

```bash
python -m ipykernel install --user --name multimodal-rag --display-name "Python (multimodal-rag)"
```

VS Code에서 `notebooks/Multi_modal_RAG.ipynb`를 열고, 커널로 **Python (multimodal-rag)**를 선택한 뒤 실행하세요.

## 주요 패키지

| 패키지 | 역할 |
|---|---|
| `unstructured` | PDF에서 텍스트·표·이미지 요소를 분리 추출 |
| `langchain-classic` | `MultiVectorRetriever` (LangChain 1.x에서 분리됨) |
| `langchain-chroma`, `chromadb` | 요약 임베딩을 저장하는 벡터 검색 엔진 |
| `langchain-openai` | LLM / VLM 및 임베딩 |
| `langchain-ollama` | 로컬 오픈소스 모델 (번역 체인 등) |
| `pillow` | 추출 이미지 전처리 |
| `tiktoken` | 토큰 단위 텍스트 분할 |
| `python-dotenv` | `.env`에서 API 키 로딩 |
