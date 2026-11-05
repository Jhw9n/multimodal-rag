"""텍스트·표·이미지 요약 체인 (교안 5, 6)."""
import base64
import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

SUMMARY_MODEL = "gpt-4o-mini"  # 비전 기능 포함, 비용 저렴 (gpt-4-vision-preview는 종료)


def generate_text_summaries(texts, tables, summarize_texts=False):
    """텍스트·표를 검색용 요약본으로 만듭니다. 반환: (text_summaries, table_summaries)"""
    prompt_text_kor = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
    이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
    주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
    요약할 표 또는 텍스트: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text_kor)
    model = ChatOpenAI(temperature=0, model=SUMMARY_MODEL)

    # 요약 체인 = 입력 포장 → 프롬프트 → LLM → 문자열
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries


def encode_image(image_path):
    """이미지를 base64 문자열로 변환합니다."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """이미지 1장의 요약을 생성합니다."""
    chat = ChatOpenAI(model=SUMMARY_MODEL, max_tokens=1024)
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


IMG_PROMPT_KOR = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""


def generate_img_summaries(path, prompt=IMG_PROMPT_KOR):
    """폴더 안의 모든 이미지를 요약합니다. 반환: (img_base64_list, image_summaries)"""
    img_base64_list = []
    image_summaries = []

    for img_file in sorted(os.listdir(path)):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            base64_image = encode_image(os.path.join(path, img_file))
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries
