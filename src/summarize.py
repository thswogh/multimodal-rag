"""요약 체인: 텍스트·표·이미지를 검색에 쓸 요약문으로 바꿉니다."""
import base64
import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# 텍스트와 표를 LLM(gpt-4o-mini)으로 요약해, 검색용 요약문 리스트 두 개를 반환합니다.
# summarize_texts=False 이면 텍스트는 요약하지 않고 원문을 그대로 돌려줍니다.
def generate_text_summaries(texts, tables, summarize_texts=False):

    # 한국어 요약 프롬프트 ({element} 자리에 요약할 표 또는 텍스트가 들어감)
    prompt_text_kor = """
                    "표와 텍스트를 검색용으로 요약하는 도우미 / 임베딩되어 원본 검색에 쓰임 /
                     검색에 최적화된 간결한 요약 작성. 원본텍스트 수의 15%정도 글자수로 요약해 / 요약할 표 또는 텍스트: {element}"""

    # 영어 요약 프롬프트 (필요할 때 prompt_text_kor 대신 사용)
    prompt_text_eng= """
                    "An assistant that summarizes tables and text for search purposes / The summaries are embedded and used to retrieve the original content /
                     Create a concise summary optimized for search / Table or text to be summarized: {element}"""

    prompt = ChatPromptTemplate.from_template(prompt_text_kor)

    # 수치를 그대로 옮겨야 하므로 temperature=0
    model = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini"
    )

    # 요약 체인: 입력 포장 → 프롬프트 → LLM → 문자열 추출
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    # 텍스트: summarize_texts 가 True 면 요약, 아니면 원문 그대로 사용
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency":2})
    elif texts: text_summaries = texts

    # 표: 있으면 요약 (요청 제한을 피하려고 동시 요청은 2개씩)
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency":2})

    return text_summaries, table_summaries


# 이미지 파일을 읽어 Base64 문자열로 바꿉니다. (이미지를 LLM에 문자열로 보내기 위함)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Base64 이미지 한 장을 비전 모델(gpt-4o-mini)에 보내고, 그 이미지의 요약문을 돌려받습니다.
def image_summarize(img_base64, prompt):
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    # 이미지 url 은 "data:image/jpeg;base64," 접두사 뒤에 img_base64 를 붙입니다.
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


# 폴더 안의 이미지를 모두 Base64로 인코딩하고 요약해, 두 리스트(원본 Base64, 요약문)를 반환합니다.
def generate_img_summaries(path):
    """
    이미지의 요약과 Base64 인코딩 문자열을 생성

    path: Unstructured에 의해 추출된 이미지 파일들의 경로
    """
    img_base64_list = []   # Base64로 인코딩된 이미지
    image_summaries = []   # 이미지 요약문

    # 지시문은 영어, 결과는 한국어로 (토큰 효율 + 품질)
    prompt_kor = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""

    # 영어 요약이 필요하면 이쪽을 사용
    # prompt_eng = """You are an assistant tasked with summarizing images for retrieval.
    # These summaries will be embedded and used to retrieve the raw image.
    # Provide a concise summary of the image that is well optimized for retrieval."""

    for img_file in sorted(os.listdir(path)):
        # .jpg, .png, .jpeg 확장자만 처리합니다
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt_kor))

    return img_base64_list, image_summaries
