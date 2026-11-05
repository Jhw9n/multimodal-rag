"""PDF에서 텍스트·표·이미지를 추출하는 함수 모음 (교안 4-1 ~ 4-4)."""
import os

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(pdf_path, figures_dir, languages=None):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.
    pdf_path:    PDF 파일 경로
    figures_dir: 추출된 이미지(.jpg)가 저장될 폴더
    languages:   OCR 언어. 한글 PDF면 ["kor", "eng"] (tesseract-lang 필요)
    """
    os.makedirs(figures_dir, exist_ok=True)
    kwargs = {"languages": languages} if languages else {}
    return partition_pdf(
        filename=pdf_path,
        strategy="hi_res",                # 레이아웃 분석 모델로 정밀 추출 (필수)
        extract_images_in_pdf=True,       # PDF에서 이미지를 추출
        infer_table_structure=True,       # 테이블 구조를 추론
        chunking_strategy="by_title",     # 타이틀 기준으로 텍스트 블록 분할
        max_characters=4000,              # 최대 4000자로 블록 제한
        new_after_n_chars=3800,           # 3800자 이후 새 블록 생성
        combine_text_under_n_chars=2000,  # 2000자 이하는 결합
        extract_image_block_output_dir=figures_dir,  # 이미지 저장 경로 (구버전 옵션명 image_output_dir_path는 무시됨)
        **kwargs,
    )


def categorize_elements(raw_pdf_elements):
    """추출한 요소를 (텍스트 리스트, 표 리스트)로 분류합니다."""
    tables = []
    texts = []
    for element in raw_pdf_elements:
        kind = type(element).__name__
        if kind in ("Table", "TableChunk"):
            tables.append(str(element))
        elif kind == "CompositeElement":
            texts.append(str(element))
    return texts, tables


def split_texts(texts, chunk_size=2000, chunk_overlap=200):
    """텍스트를 하나로 합친 뒤 토큰 기준으로 다시 자릅니다.
    chunk_overlap: 블록 경계에서 문장이 잘려 뜻이 사라지는 것을 막기 위한 겹침."""
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(" ".join(texts))
