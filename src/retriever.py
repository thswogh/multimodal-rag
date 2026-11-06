"""다중 벡터 검색기와 최종 체인: 요약으로 검색하고, 원본으로 답변을 생성합니다."""
import re
import uuid

# langchain 1.x 에서 두 클래스의 위치가 바뀌었습니다.
try:
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
except ImportError:
    print("MultiVectorRetriever 클래스가 필요합니다")

try:
    from langchain_core.stores import InMemoryStore
except ImportError:
    print("InMemoryStore 클래스가 필요합니다")

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.documents import Document as _Doc
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 요약과 원본을 연결하는 키 (= 청구기호)
id_key = "doc_id"


# 요약을 담을 벡터 DB(Chroma)와 원본을 담을 저장소(InMemoryStore)를 doc_id로 묶은 검색기를 만듭니다.
def build_retriever():
    # 요약문이 들어갈 벡터 DB (= 색인 카드 서랍)
    vectorstore = Chroma(
         collection_name="multimodal_rag" , embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    )
    # 다시 실행해도 이전 요약이 남지 않도록 컬렉션을 비운다.
    # (안 비우면 요약은 검색되는데 새 docstore엔 원본이 없어서 검색 결과가 빈 리스트가 됨)
    vectorstore.reset_collection()

    # 원본이 들어갈 저장소 (= 서가)
    store = InMemoryStore()

    retriever = MultiVectorRetriever(
         vectorstore= vectorstore, docstore= store, id_key = id_key
    )
    return retriever


# 요약은 벡터 DB에, 원본은 docstore에 같은 doc_id로 저장해 짝지어 둡니다.
def add_documents(retriever, doc_summaries, doc_contents):
    """요약본은 벡터 DB에, 원본은 docstore에 '같은 id'로 저장합니다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

    summary_docs = [
        _Doc(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]

    # 요약본 → 벡터 DB (검색용)
    retriever.vectorstore.add_documents(summary_docs)

    # 원본 → docstore (전달용)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


# 검색 결과를 Base64 이미지와 텍스트(표 포함)로 나눕니다.
# Base64 문자(영문/숫자/+/)가 100자 이상 이어지고 끝에 '='이 0~2개면 이미지로 판단합니다.
def split_image_text_types(docs):
    b64_images=[]
    texts_out = []

    for doc in docs:
        doc.page_content if isinstance(doc,Document) else doc
        if re.match("^[A-Za-z0-9+/]{100,}={0,2}$",doc):
            b64_images.append(doc)
        else:
            texts_out.append(doc)
    return {"images":b64_images, "texts" :texts_out}


# 검색된 원본 이미지·텍스트와 질문을 멀티모달 LLM이 읽을 수 있는 메시지 하나로 조립합니다.
def img_prompt_func(data_dict):
    messages = []

    # 1. 이미지를 먼저 넣는다.
    for image in data_dict["context"]["images"]:
        messages.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
        )
    # 2. 텍스트 컨텍스트와 질문을 넣는다.
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({
        "type":"text",
        "text":( "자료를 근거로 질문에 한국어로 답하세요\n"
                 "자료에 없는 내용은 모른다고 답하세요\n\n"
                 f"질문: {data_dict['question']}\n\n"
                 f"참고텍스트: \n{formatted_texts}"
                )
    })
    return [HumanMessage(content=messages)]


# 최종 멀티모달 체인을 만듭니다: 검색 → 이미지/텍스트 분리 → 메시지 조립 → LLM → 문자열 답변
def build_chain(retriever):
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)

    chain=(
        {
            "context": retriever | RunnableLambda (split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        |model
        | StrOutputParser()
    )
    return chain


# 검색된 문서들의 본문만 꺼내 줄바꿈으로 이어 붙입니다.
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


# Before/After 비교용으로, 이미지와 표 없이 텍스트만 검색하는 일반 RAG 체인을 만듭니다.
# 비교가 공정하도록 모델·임베딩·지시문은 멀티모달 체인과 같게 맞췄습니다.
def build_text_only_chain(texts_2k_token):
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)

    text_only_vectorstore = Chroma(
        collection_name="text_only_rag", embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    text_only_vectorstore.add_texts(texts=texts_2k_token)

    text_only_retriever = text_only_vectorstore.as_retriever(search_kwargs={"k":4})

    text_only_prompt = ChatPromptTemplate.from_template(
        "자료를 근거로 질문에 한국어로 답하세요\n"
        "자료에 없는 내용은 모른다고 답하세요\n\n"
        "질문: {question}\n\n"
        "참고텍스트: \n{context}"
    )

    text_only_chain = (
        {
            "context": text_only_retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        |text_only_prompt
        |model
        |StrOutputParser()
    )
    return text_only_chain
