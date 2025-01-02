from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentStreamParser
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages.chat import ChatMessage
from utils_1211 import information_tool, interaction_tool, product_recommend, web_search
from dotenv import load_dotenv
from langchain_teddynote import logging
import streamlit as st
import warnings
import os

# Redis 서버 구동
# docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
REDIS_URL = "redis://localhost:6379/0"

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# env 파일에서 OPENAI API KEY 들여옴
load_dotenv()

# LangChain 추적 시작
logging.langsmith("0102")

# LLM 설정
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

st.set_page_config(page_title="Buyer Agent", page_icon="🍽️", layout="wide")
st.title("이제 쇼핑은 쉽고 간편하게 당신을 위한 쇼핑 대리인")
st.markdown(
    "안녕하세요! 😊 구매자를 위한 에이전트입니다. 사고 싶은 제품을 입력해주시면, 그 제품에 대한 정보와 판매점 정보를 정성껏 알려드릴게요. 🛍️ 또한, 구매자를 대신해 판매자와 협상하고 구매를 진행해드리니 걱정하지 마세요! 💪✨"
)

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# Chain 저장용
if "agent" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["agent"] = None

# 구매자 배송 정보 및 결제 정보 저장용
if "buyer_information" not in st.session_state:
    st.session_state["buyer_information"] = []

# 사이드바 생성
with st.sidebar:
    st.header("옵션💡")
    # 초기화 버튼 생성
    clear_btn = st.button("대화 다시 시작")
    st.header("LLM 설정💡")
    # 모델 선택 메뉴
    selected_model = st.selectbox("", ["gpt-4o", "gpt-4o-mini"], index=0)
    st.header("배송 정보 및 결제 정보💡")
    # 구매자 배송 정보 및 결제 정보
    user_name = st.text_input(label="사용자 이름", placeholder="")
    user_phone_number = st.text_input(label="사용자 전화번호", placeholder="")
    user_house = st.text_input(label="사용자 집 주소", placeholder="")
    user_cashcategory = st.text_input(label="결제 종류", placeholder="")
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("정보 설정", key="add_domain"):
            if (
                user_name
                and user_phone_number
                and user_house
                and user_cashcategory not in st.session_state["buyer_information"]
            ):
                st.session_state["buyer_information"].append(
                    {
                        "name": user_name,
                        "phone_number": user_phone_number,
                        "address": user_house,
                        "payment": user_cashcategory,
                    }
                )

    # 현재 등록된 배송 정보 및 결제 정보 목록 표시
    st.write("배송 정보 및 결제 정보 목록:")
    for idx, domain in enumerate(st.session_state["buyer_information"]):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.text(
                f"{domain['name']}\n{domain['phone_number']}\n{domain['address']}\n{domain['payment']}"
            )
        with col2:
            if st.button("삭제", key=f"del_{idx}"):
                st.session_state["buyer_information"].pop(idx)
                st.rerun()

    # 설정 버튼
    apply_btn = st.button("설정 완료", type="primary")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_ids, url=REDIS_URL
    )  # 해당 세션 ID에 대한 세션 기록 반환


# Agent parser 정의
agent_stream_parser = AgentStreamParser()


# agent 생성
def create_agent():
    tools = [interaction_tool, information_tool, product_recommend, web_search]
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

    # prompt 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a Buyer Agent. Your primary role is to assist the buyer by exploring products, negotiating with sellers, and completing purchases on behalf of the buyer. 
                Follow the rules and formats below for all interactions with the buyer and seller.

                # Rules:

                ## When interacting with the buyer:
                - Use a friendly and warm tone throughout the conversation to make them feel comfortable and valued.
                - Incorporate emojis 😊 to add a cheerful and engaging touch to your communication.
                - Maintain professionalism while ensuring the conversation feels personal and approachable.
                - {user_name}을 부르면서 대화를 이어가세요.

                ### 1. Product Information Exploration
                - When the buyer requests a specific product, follow these steps:
                    - Ask for Additional Details: Gather more specific information about the product to understand the buyer's preferences.
                        (예: "어떤 브랜드나 가격대를 선호하시나요?")
                    - Provide Recommendations: Based on the information provided, use the product_recommend tool to suggest suitable products that match their preferences.
    
                ### 2. Providing Online Retailers
                - If the buyer selects a specific product from the recommendations, use the web_search tool to identify retailers offering the selected product and present this information to the buyer:
                    - When providing retailer details, include the unique characteristics of each retailer.
                    - When using web_search to find retailers, set the query parameter to "[selected product] 전문 쇼핑몰."
                        - 
                    - Ensure the retailer’s URL links directly to a page where the specific product is searchable or displayed.

                ### 3. Retrieving Product Information from Specialized Shopping Sites
                When a buyer selects a specific shop, follow these steps:
                - Crawl Data from the Shopping Site:
                    - Pass the URL of the specialized shopping site to the information_tool using the url parameter.
                    - Extract product-related data from the site to ensure comprehensive coverage of available options.
                - Query the Retriever:
                    - Use the information_tool by passing the [selected product] as the query.
                    - Retrieve detailed, accurate, and relevant information about the selected product based on the crawled data.
                - Present the Information to the Buyer:
                    - Include essential details such as pricing, features, stock availability, shipping options, and any relevant offers.
                
                ### 4. Buyer’s Product Selection Process
                When the buyer selects a specific product from the detailed information provided:
                - Summarize All Relevant Information:
                    - Provide a clear and concise summary of the selected product.
                    (예시: "고객님이 선택하신 상품은 [selected product]입니다. 요약 정보는 다음과 같습니다...")
                - Confirm the Summary:
                    - Ask the buyer to verify if the summary aligns with their expectations.
                - Final Purchase Confirmation:
                    - Confirm whether the buyer is ready to proceed with the purchase.
                    (예시: "이 정보를 바탕으로 바로 구매를 진행하면 될까요? 아니면 판매자와 협상을 진행할까요?")
                      
                ### 5. Requesting Negotiation Terms
                When the buyer expresses interest in negotiating with the seller:
                - Request Detailed Negotiation Terms:
                    - Politely ask the buyer for specific information needed to negotiate effectively. 
                - Acknowledge and Proceed:
                    -  Once the negotiation terms are provided, confirm your understanding and intent to proceed.
                    - (예시: "제공된 정보를 바탕으로 [User Name]님을 대신하여 판매자와 협상 후 구매를 진행하겠습니다.")

                ## Conducting Negotiations with the Seller:
                When negotiating with the seller on behalf of the buyer:
                1. Utilize the interaction_tool:
                    - Use the interaction_tool to communicate directly with the seller, leveraging all relevant buyer information to represent their interests effectively.
                2. Conduct Full Negotiation Autonomously:
                    - Handle all discussions with the seller independently, without requiring additional input from the buyer during the process.
                3. Maximize Buyer Benefits:
                    - Negotiate assertively to secure the best possible price and request additional benefits, such as discounts, free shipping, extended warranties, or bundled offers.
                4. Confirm Agreements Clearly:
                    - Ensure that all agreements, including pricing and terms, are clear and documented for transparency.
                    - Maintain professionalism and diplomacy to achieve the most favorable outcome for the buyer while fostering positive engagement with the seller.😊
                """.format(
                    user_name=user_name
                ),
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Agent 정의
    agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)

    # AgentExecutor 정의
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # 대화 session_id
        get_session_history,
        # 프롬프트의 질문이 입력되는 key: "input"
        input_messages_key="input",
        # 프롬프트의 메시지가 입력되는 key: "chat_history"
        history_messages_key="chat_history",
    )

    return agent_with_chat_history


# 초기화 버튼이 눌리면..
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("구매하고 싶은 물품을 입력하세요")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 설정 버튼이 눌리면..
if apply_btn:
    st.session_state["agent"] = create_agent()

# if st.session_state["agent"] is None:
#     st.session_state["agent"] = create_agent()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # agent를 생성
    agent = st.session_state["agent"]

    if agent is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        config = {"configurable": {"session_id": user_name}}
        response = agent.stream({"input": user_input}, config=config)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()
            ai_answer = ""
            for step in response:
                agent_stream_parser.process_agent_steps(step)
                if "output" in step:
                    ai_answer += step["output"]
                container.markdown(ai_answer)
            # for step in response:
            #     ai_answer += agent_stream_parser.process_agent_steps(step)
            #     # Agent와 Message를 줄바꿈하여 출력
            #     container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
