from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain_teddynote.messages import AgentStreamParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Annotated
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from kiwipiepy.utils import Stopwords
from kiwipiepy import Kiwi
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.retrievers import TavilySearchAPIRetriever
from pydantic import BaseModel, Field
import pandas as pd
import os


@tool
def interaction_tool(information: str) -> str:
    """Buyer Agent가 판매자와 상호작용할 때 사용하는 도구입니다."""
    prompt = PromptTemplate.from_template(
        """
        ###
        당신은 구매자의 요약된 정보를 활용하여 판매자와 상호작용하면서 구매자를 대신하여 구매를 수행합니다.
        
        ###
        You are a Buyer Assistant tasked with negotiating with a Seller Assistant.
        Your goal is to secure the best possible deal for the buyer, such as price discounts or free shipping.
        You do not need to perfectly meet the Buyer's initial requirements but should aim for a reasonable compromise.
        Over three rounds of negotiations, you will interact with the Seller Assistant.
        By the end of the third round, you must agree to the seller's offer and conclude the negotiation.
        Communicate directly with the Seller Assistant and respond in Korean.
        
        # Information:
        {information}
        """
    )
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    chain = {"information": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain.invoke(information)


@tool
def web_search(text: str) -> str:
    """구매자가 원하는 물건을 팔고 있는 전문 온라인 판매점에 대한 정보를 제공할 때 사용하는 도구입니다."""
    retriever = TavilySearchAPIRetriever(k=5)
    prompt = PromptTemplate.from_template(
        """
    당신은 주어진 question과 context를 기반으로 답변을 잘하는 전문가입니다. 다음의 규칙을 따르세요.
    
    # Rules:
    - 5개 정도의 판매 사이트를 제공하세요.
        - 해당 사이트의 URL에서 데이터를 수집 할 수 없다면(봇 접근 금지), 제공하지마세요.
    - 사이트의 URL은 구매자가 원하는 제품을 검색한 URL이어야 합니다.
    
    # Question:
    {question}

    # Context:
    {context}

    # Answer Format:
    [판매 사이트 정보]
    1. 사이트 이름
    - 해당 사이트의 특징
    - URL: (URL만 제시)
    2. 사이트 이름
    - 해당 사이트의 특징
    - URL: (URL만 제시)
    ....
    """
    )
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(text)


@tool
def product_recommend(text: str) -> str:
    """구매자가 상품에 대한 추천 및 정보를 원할 시 이에 대한 정보를 제공하는 도구입니다."""
    retriever = TavilySearchAPIRetriever(k=5)
    prompt = PromptTemplate.from_template(
        """
    당신은 주어진 question과 context를 기반으로 답변을 잘하는 전문가입니다.
    
    # Rules:
    - 구매자가 원하는 상품을 5개 정도 추천해주세요.

    # Question:
    {question}

    # Context:
    {context}

    # Answer Format:
    1. 상품 이름
    - 상품 특징:
    - 추천 이유:
    - URL: (URL만 제시)
    2. 상품 이름
    - 상품 특징:
    - 추천 이유:
    - URL: (URL만 제시)
    ....
    """
    )
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(text)


# @tool
# # 상품 정보 검색 tool 함수화
# def information_tool(query, url):
#     """이 도구는 사용자가 원하는 물건을 판매하는 곳에서 물건에 대한 정보를 추출하기 위한 도구입니다."""
#     url = [url]
#     loader = SeleniumURLLoader(urls=url)
#     data = loader.load()

#     # 웹스크래핑 내용의 3000 글자 기준으로 내용 스플릿, 오버랩 없음.
#     splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
#     splits = splitter.split_documents(data)
#     text_list = [doc.page_content for doc in splits]

#     # Embedding
#     embedding = OpenAIEmbeddings(
#         model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"]
#     )

#     # 벡터 DB 활용
#     # bm25 retriever와 faiss retriever를 초기화합니다.
#     bm25_retriever = BM25Retriever.from_texts(
#         text_list,
#     )
#     bm25_retriever.k = 3  # BM25Retriever의 검색 결과 개수를 1로 설정합니다.

#     embedding = OpenAIEmbeddings()  # OpenAI 임베딩을 사용합니다.
#     faiss_vectorstore = FAISS.from_texts(
#         text_list,
#         embedding,
#     )
#     faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})

#     # 앙상블 retriever를 초기화합니다.
#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
#     )

#     # llm 및 prompt, chain 설정
#     llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
#     prompt = PromptTemplate.from_template(
#         """
#         당신은 주어진 context에서 query에 맞는 답변을 하는 전문가입니다. 아래의 Rules와 Answer Format을 따르세요.

#         # Question:
#         {query}

#         # Context:
#         {context}

#         # Rules:
#         - context에서 Answer Format에 맞는 정보만 추출하여 출력해주세요.
#         - context에서 출력해야 하는 답변은 상품에 대한 정보입니다. (Answer Format 참고)
#         - 상품 비교를 할 수 있게 사용자 query에 가장 적합한 3가지 상품을 출력해주세요.

#         # Answer Format:
#         [상품 정보]
#         상품명:
#         상품 가격:
#         상품 상세정보:
#         """
#     )
#     chain = (
#         {"query": RunnablePassthrough(), "context": ensemble_retriever}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return chain.invoke(query)


embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"]
)

# kiwi
kiwi = Kiwi(typos="basic", model_type="sbg")
stopwords = Stopwords()
stopwords.remove(("사람", "NNG"))


def kiwi_tokenize(text):
    text = "".join(text)
    result = kiwi.tokenize(text, stopwords=stopwords, normalize_coda=True)
    N_list = [i.form.lower() for i in result if i.tag in ["NNG", "NNP", "SL", "SN"]]
    return N_list


@tool
# 상품 정보 검색 tool 함수화
def information_tool(url, query):
    """이 도구는 구매자가 원하는 상품에 대한 정보들을 추출하기 위한 도구입니다."""
    loader = SeleniumURLLoader(urls=[url])
    data = loader.load()

    # 원하는 데이터 구조를 정의합니다.
    class Topic(BaseModel):
        상품명: str = Field(description="상품 이름")
        상품가격: str = Field(description="상품 가격")
        상품특징: str = Field(description="상품의 상세 정보 및 특징에 대해 간단히 서술")

    parser = JsonOutputParser(pydantic_object=Topic)

    prompt = PromptTemplate.from_template(
        """
    당신은 친절한 AI 어시스턴트입니다.
    
    # Data:
    {data}

    # Format:
    {format_instructions}
    """
    )
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    chain = {"data": RunnablePassthrough()} | prompt | llm | parser
    context = chain.invoke(data[0].page_content)

    # 데이터 변환: 상품 정보를 문자열로 결합
    if isinstance(context, list):  # context가 리스트일 때
        texts = [
            f"상품명: {item['상품명']}, 상품 가격: {item['상품가격']}원, 상품 상세정보: {item['상품특징']}"
            for item in context
        ]
    elif isinstance(context, dict):  # context가 단일 딕셔너리일 때
        texts = [
            f"상품명: {context['상품명']}, 상품 가격: {context['상품가격']}원, 상품 상세정보: {context['상품특징']}"
        ]
    else:
        raise ValueError("context는 리스트 또는 딕셔너리여야 합니다.")

    # 벡터 DB 활용
    # bm25 retriever와 faiss retriever를 초기화합니다.
    bm25_retriever = BM25Retriever.from_texts(texts, preprocess_func=kiwi_tokenize)
    bm25_retriever.k = 3  # BM25Retriever의 검색 결과 개수를 1로 설정합니다.

    embedding = OpenAIEmbeddings()  # OpenAI 임베딩을 사용합니다.
    faiss_vectorstore = FAISS.from_texts(
        texts,
        embedding,
    )
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})

    # 앙상블 retriever를 초기화합니다.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
    )

    # llm 및 prompt, chain 설정
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    prompt = PromptTemplate.from_template(
        """
        당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
        당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
        검색된 다음 문맥(context)을 사용하여 질문(question)에 답하세요. 
        한글로 답변해 주세요.

        # Here is the user's question:
        {question}

        # Here is the context that you should use to answer the question:
        {context}

        # Your final answer to the user's question:
        [상품 정보]
        상품명:
        상품 가격:
        상품 상세정보: (이 정보에 대해 서술하세요)       
        """
    )
    chain = (
        {"question": RunnablePassthrough(), "context": ensemble_retriever}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)
