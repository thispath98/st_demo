import os
import sqlite3
import pandas as pd
import streamlit as st

# from dotenv import load_dotenv
# from langchain_community.llms import OpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI, OpenAI

from BuildGraph import TourGuideRAG

# 페이지 설정
st.set_page_config(page_title="TourGuideRAG", page_icon="🎡")

# 메시지 세션 스테이트 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []


class ChatCallbackHandler(BaseCallbackHandler):
    """
    LLM이 토큰 단위로 출력할 때마다 Streamlit UI에 실시간 업데이트해주는 콜백 핸들러입니다.
    """
    message = ""

    def on_llm_start(self, *args, **kwargs):
        """LLM이 시작될 때 호출됩니다. 토큰 누적용 빈 컨테이너를 만듭니다."""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        """LLM이 종료될 때 호출됩니다. 최종 메시지를 저장합니다."""
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        """LLM이 새 토큰을 생성할 때마다 호출됩니다. 토큰을 누적해 UI에 표시합니다."""
        self.message += token
        self.message_box.markdown(self.message)


def save_message(message: str, role: str) -> None:
    """메시지를 세션 스테이트에 저장합니다."""
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message: str, role: str, save: bool = True) -> None:
    """
    채팅 UI에 메시지를 출력합니다.
    save=True인 경우, 세션 스테이트에도 메시지를 저장합니다.
    """
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history() -> None:
    """세션 스테이트에 기록된 메시지를 모두 다시 출력합니다."""
    for msg in st.session_state["messages"]:
        send_message(msg["message"], msg["role"], save=False)


# OpenAI API 키 로드
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "LangChain 프로젝트명"
# os.environ["LANGCHAIN_API_KEY"] = "LangChain API KEY 입력"
# st.write(st.secrets["LANGCHAIN_TRACING_V2"])
# st.write(st.secrets["LANGCHAIN_ENDPOINT"])
# st.write(st.secrets["LANGCHAIN_PROJECT"])
# st.write(st.secrets["LANGCHAIN_API_KEY"])

# LLM 인스턴스 준비
llm_back = OpenAI(openai_api_key=OPENAI_API_KEY)
llm_chat = ChatOpenAI(model="gpt-4o", temperature=0)
llm = OpenAI(
    streaming=True,
    callbacks=[ChatCallbackHandler()],
    openai_api_key=OPENAI_API_KEY,
)

# RAG 객체 생성
tourag = TourGuideRAG(llm_back, llm_chat, llm)
app = tourag.buildgraph()

# UI 구성
st.title("TourGuideRAG")
st.write("### Welcome!\n\nUse this chatbot to gather information for a trip to Busan~~!!!")

# 초기 메시지 및 이전 대화 기록 표시
send_message("I'm ready. Ask away!", "ai", save=False)
paint_history()

# 사용자 입력 처리
message = st.chat_input("Ask anything about Busan tour...")

if message:
    send_message(message, "human")
    inputs = {"question": message}
    with st.chat_message("ai"):
        response = app.invoke(inputs)

button = st.sidebar.button("draw image")
if button:
    with st.sidebar:
        st.image(app.get_graph().draw_mermaid_png(), caption="Sunrise by the mountains")
