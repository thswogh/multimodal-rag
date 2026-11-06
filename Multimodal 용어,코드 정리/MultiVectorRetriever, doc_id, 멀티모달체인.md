# MultiVectorRetriever, doc_id, 멀티모달체인

## 오늘의 키워드

| 키워드 | 한 줄 정리 |
| --- | --- |
| **MultiVectorRetriever** | 요약으로 **검색**하고, 원본을 **반환**하는 검색기 |
| **vectorstore (Chroma)** | 요약문 임베딩 저장소 = 색인 카드 서랍 → **검색용** |
| **docstore (InMemoryStore)** | 원본(텍스트·표·Base64 이미지) 저장소 = 서가 → **전달용** |
| **id_key = "doc_id"** | 요약과 원본을 잇는 번호표(청구기호) |
| **uuid.uuid4()** | 겹치지 않는 고유 ID 생성 |
| **mset** | docstore에 `(id, 원본)` 쌍을 한 번에 저장 |
| **Base64 정규식** | `^[A-Za-z0-9+/]{100,}={0,2}$` → 이미지/텍스트 구분 |
| **HumanMessage (image_url + text)** | 멀티모달 LLM에 이미지와 텍스트를 함께 전달 |
| **LCEL `|`** | `{context, question} → prompt → model → parser` 체인 연결 |
| **RunnableLambda / RunnablePassthrough** | 일반 함수를 체인 부품으로 / 입력을 그대로 통과 |
| **Before / After** | 텍스트 전용 RAG vs 멀티모달 RAG 비교 |

> 핵심 한 줄: **요약과 원본에 같은 번호표(doc_id)를 붙인다.**

---

## 1. 결과 복원

앞에서 저장한 pickle을 다시 불러온다.

- `.env`가 있는 폴더를 프로젝트 루트로 자동 탐색
- `extracted.pkl` → `texts`, `tables`, `texts_2k_token`
- `summaries.pkl` → `text_summaries`, `table_summaries`, `image_summaries`, `img_base64_list`

```python
PROJECT_ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".env").exists()), Path.cwd()
)
with open(processed / "summaries.pkl", "rb") as f:
    d = pickle.load(f)
```

✅ 원본 수 = 요약 수 인지 확인

---

## 2. 다중 벡터 검색기 구조

| 구성 요소 | 비유 | 들어가는 것 | 역할 |
| --- | --- | --- | --- |
| `vectorstore` | 색인 카드 서랍 | 요약문 임베딩 | 검색 |
| `docstore` | 서가 | 원본 | 전달 |
| `id_key` | 청구기호 | `doc_id` | 연결 |

```python
vectorstore = Chroma(collection_name="multimodal_rag",
                     embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
store = InMemoryStore()
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
```

- langchain 1.x: `langchain_classic.retrievers.multi_vector`, `langchain_core.stores`
- `InMemoryStore`는 커널 재시작 시 사라짐 → 실서비스는 파일/Redis

---

## 3. 요약 ↔ 원본 짝짓기 (add_documents)

```python
def add_documents(retriever, doc_summaries, doc_contents):
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]})
                    for i, s in enumerate(doc_summaries)]
    retriever.vectorstore.add_documents(summary_docs)            # 요약 → 검색용
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))    # 원본 → 전달용
```

⚠️ 두 번 실행하면 중복 저장 → 검색기 생성부터 재실행

---

## 4. 검색 동작 순서 (`retriever.invoke(query)`)

1. 질문 → 임베딩
2. vectorstore에서 가장 비슷한 **요약문** 찾기
3. 요약의 `metadata["doc_id"]` 읽기
4. docstore에서 같은 id의 **원본** 꺼내 반환

💡 질문은 **그래프·표를 봐야만 답할 수 있는 것**으로 해야 Before/After 차이가 난다.

---

## 5. 이미지 / 텍스트 분리

```python
def split_image_text_types(docs):
    b64_images, texts_out = [], []
    for doc in docs:
        content = doc.page_content if isinstance(doc, Document) else doc
        if re.match(r"^[A-Za-z0-9+/]{100,}={0,2}$", content):
            b64_images.append(content)
        else:
            texts_out.append(content)
    return {"images": b64_images, "texts": texts_out}
```

---

## 6. 최종 멀티모달 체인

```
{검색 + 질문} → img_prompt_func → 멀티모달 LLM → StrOutputParser
 재료 모으기     메시지로 조립       답변 생성       문자열로 꺼내기
```

- 이미지는 `{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}`
- 이미지를 먼저 넣고, 그 다음 지시문 + 질문 + 참고 텍스트
- `ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)`

---

## 7. Before / After 비교

| | 텍스트 전용 RAG | 멀티모달 RAG |
| --- | --- | --- |
| 저장 | `texts_2k_token`만 `add_texts` | 텍스트+표+이미지 요약 |
| 검색기 | `as_retriever(search_kwargs={"k": 4})` | MultiVectorRetriever |
| 결과 | "자료에 정보가 없습니다" | 1차 5009회(97.21%) 등 표/그래프 수치로 답변 |

**차이 이유:** 표·이미지를 요약 → 임베딩해 검색 대상에 넣었기 때문에 그래프에만 있는 정보도 찾아낸다.

---

## 8. 생각해 볼 질문

- **doc_id를 안 맞추면?** 요약은 찾아도 원본을 못 꺼내 결과가 빈다.
- **검색은 요약, 답변은 원본인 이유?** 요약은 검색에 유리하지만 정보 손실이 있음 → 답변은 원본으로.
- **실서비스라면?** InMemoryStore·Chroma를 디스크/외부 DB로 영구 저장.

---

## 9. 자주 나는 에러

| 증상 | 해결 |
| --- | --- |
| `FileNotFoundError: summaries.pkl` | 앞 노트북 저장 셀 실행 |
| 검색 결과에 이미지 없음 | 이미지 요약 프롬프트 보강 |
| 같은 원본 중복 | `add_documents` 재실행 금지, 검색기부터 다시 |
| Before/After 차이 없음 | 그래프에만 있는 정보를 묻도록 질문 수정 |

---

## 오늘의 전체 흐름 (한 번에 보기)

```python
# ============================================================
# 멀티모달 RAG 최종: 요약으로 검색 → 원본 꺼내기 → 멀티모달 LLM 답변
# ============================================================
import os, re, time, uuid, pickle
from pathlib import Path
from dotenv import load_dotenv

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever  # 0.3: langchain.retrievers.multi_vector
from langchain_core.stores import InMemoryStore                            # 0.3: langchain.storage
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ── 1. 환경 설정 & 이전 결과 복원 ─────────────────────────
# 현재 폴더부터 상위로 올라가며 .env가 있는 곳을 프로젝트 루트로 사용
PROJECT_ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".env").exists()), Path.cwd()
)
load_dotenv(PROJECT_ROOT / ".env")                    # OPENAI_API_KEY 로드
processed = PROJECT_ROOT / "data" / "processed"

with open(processed / "extracted.pkl", "rb") as f:    # 원본(추출 결과)
    d = pickle.load(f)
    texts_2k_token, tables = d["texts_2k_token"], d["tables"]

with open(processed / "summaries.pkl", "rb") as f:    # 요약 결과 + 이미지 Base64
    d = pickle.load(f)
    text_summaries, table_summaries = d["text_summaries"], d["table_summaries"]
    image_summaries, img_base64_list = d["image_summaries"], d["img_base64_list"]

# ── 2. 다중 벡터 검색기 생성 ──────────────────────────────
id_key = "doc_id"                                     # 요약 ↔ 원본 연결 번호표
vectorstore = Chroma(                                 # 색인 카드 서랍: 요약 임베딩 (검색용)
    collection_name="multimodal_rag",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)
store = InMemoryStore()                               # 서가: 원본 보관 (전달용, 메모리라 휘발성)
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

# ── 3. 요약과 원본을 같은 id로 저장 ───────────────────────
def add_documents(retriever, doc_summaries, doc_contents):
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]          # 원본마다 고유 id
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})  # 요약에 id를 메타데이터로 부착
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)            # 요약 → 벡터DB
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))    # (id, 원본) → docstore

# 텍스트 / 표 / 이미지 각각 색인 (⚠️ 한 번만 실행!)
for summaries, contents in [(text_summaries, texts_2k_token),
                            (table_summaries, tables),
                            (image_summaries, img_base64_list)]:
    if summaries:
        add_documents(retriever, summaries, contents)

# ── 4. 검색 결과를 이미지 / 텍스트로 분리 ─────────────────
B64_PATTERN = r"^[A-Za-z0-9+/]{100,}={0,2}$"          # Base64 문자만 길게 + 끝에 = 0~2개 → 이미지

def split_image_text_types(docs):
    b64_images, texts_out = [], []
    for doc in docs:
        content = doc.page_content if isinstance(doc, Document) else doc
        (b64_images if re.match(B64_PATTERN, content) else texts_out).append(content)
    return {"images": b64_images, "texts": texts_out}

# ── 5. 멀티모달 메시지 조립 ───────────────────────────────
def img_prompt_func(data_dict):
    messages = []
    for image in data_dict["context"]["images"]:      # ① 이미지를 먼저 (모델이 그림부터 보게)
        messages.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({"type": "text", "text": (        # ② 지시문 + 질문 + 참고 텍스트
        "자료를 근거로 질문에 한국어로 답하세요\n"
        "자료에 없는 내용은 모른다고 답하세요\n\n"
        f"질문: {data_dict['question']}\n\n"
        f"참고텍스트:\n{formatted_texts}"
    )})
    return [HumanMessage(content=messages)]

# ── 6. 최종 체인 (LCEL) ───────────────────────────────────
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)

chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),  # 요약으로 검색 → 원본 → 분리
        "question": RunnablePassthrough(),                              # 질문은 그대로 통과
    }
    | RunnableLambda(img_prompt_func)   # 멀티모달 메시지 조립
    | model                             # 이미지+텍스트를 보고 답변 생성
    | StrOutputParser()                 # AIMessage → 문자열
)

# ── 7. 비교용 텍스트 전용 RAG (Before) ────────────────────
text_only_vs = Chroma(collection_name="text_only_rag",
                      embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
text_only_vs.add_texts(texts=texts_2k_token)          # 표·이미지 제외, 텍스트만
text_only_retriever = text_only_vs.as_retriever(search_kwargs={"k": 4})

text_only_prompt = ChatPromptTemplate.from_template(  # 멀티모달과 같은 지시문 (공정 비교)
    "자료를 근거로 질문에 한국어로 답하세요\n"
    "자료에 없는 내용은 모른다고 답하세요\n\n"
    "질문: {question}\n\n참고텍스트:\n{context}"
)

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

text_only_chain = (
    {"context": text_only_retriever | RunnableLambda(format_docs),
     "question": RunnablePassthrough()}
    | text_only_prompt | model | StrOutputParser()
)

# ── 8. 실행 & Before / After 비교 ─────────────────────────
query = "과거 데이터 모의실험에서 상환 차수별 횟수와 빈도율은? 1차 상환 비율은 몇 %인가요?"

start = time.time()
print("[BEFORE] 텍스트 전용:", text_only_chain.invoke(query))  # → "자료에 없습니다"
print("[AFTER]  멀티모달  :", chain.invoke(query))             # → 1차 5009회(97.21%) ...
print(f"소요 시간: {time.time() - start:.1f}초")
```