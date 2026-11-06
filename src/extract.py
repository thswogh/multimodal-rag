"""PDF 파싱: PDF에서 텍스트·표·이미지를 추출하고, 텍스트를 검색하기 좋은 크기로 나눕니다."""
import os

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


# PDF를 텍스트·표·이미지 요소로 분리하고, 그림은 path/figures 폴더에 jpg 파일로 저장합니다.
def extract_pdf_elements(path, fname):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.

    path : 파일 경로 (추출된 이미지는 path/figures 에 저장됨)
    fname: 파일 이름
    """
    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",                  # 레이아웃 분석 모델로 정밀 추출
        extract_images_in_pdf=True,       # PDF 속 이미지를 파일로 추출
        infer_table_structure=True,       # 표의 행·열 구조를 추론
        chunking_strategy="by_title",     # 타이틀 기준으로 텍스트 분할
        max_characters=4000,              # 최대 4000자로 블록 제한
        new_after_n_chars=3800,           # 3800자 이후 새 블록 생성
        combine_text_under_n_chars=2000,  # 2000자 이하는 결합
        extract_image_block_output_dir=os.path.join(path, "figures"),  # 이미지 저장 경로 → data/figures
    )


# 추출한 요소를 타입에 따라 텍스트 리스트와 표 리스트로 나눕니다.
def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출한 요소들을 테이블과 텍스트로 분류합니다.

    raw_pdf_elements: unstructured.documents.elements 리스트
    """
    tables = []
    texts = []

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))     # 표 요소는 tables 에 저장
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))     # 텍스트 덩어리는 texts 에 저장

    return texts, tables


# 텍스트 블록을 하나로 합친 뒤 2000토큰 단위로, 200토큰씩 겹치게 다시 자릅니다.
def split_texts(texts):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,       # 각 텍스트 블록의 최대 크기 (토큰)
        chunk_overlap=200,    # 청크 간 겹칠 크기 (토큰)
    )

    joined_texts = " ".join(texts)
    texts_2k_token = text_splitter.split_text(joined_texts)

    return texts_2k_token
