import os
import json
import sqlite3
import pandas as pd
from typing import TypedDict, Literal

# 필요한 경우 활성화
import streamlit as st
# from langchain_openai import ChatOpenAI, OpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS
# from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langgraph.graph import END, StateGraph


class GraphState(TypedDict):
    """
    TourGuideRAG에서 사용하는 그래프 상태를 정의한다.
    """
    region: str
    question: str
    food_type: str
    vec_store: str
    answer: str
    end: bool


class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource.
    """
    datasource: Literal[
        "local_tourist_spots",
        "foreign_tourist_spots",
        "restaurants",
        "web search",
    ] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )


class TourGuideRAG:
    """
    부산 관광 정보를 제공하기 위해 RAG(Retrieval-Augmented Generation) 로직을 구성한 클래스.
    주어진 LLM들을 통해 사용자의 질의에 맞는 데이터를 검색하고, 적절한 답변을 생성한다.
    내부적으로 SQLite를 사용하여 CSV 데이터를 관리한다.
    """

    def __init__(self, llm_back, llm_chat, llm):
        """
        Args:
            llm_back: SQL 쿼리 생성이나 기타 백엔드 로직에 사용되는 LLM
            llm_chat: 모델 function call(Structured LLM)을 활용할 수 있는 LLM (ex. ChatOpenAI)
            llm: 최종 답변 생성을 위한 LLM
        """
        self.llm_back = llm_back
        self.llm_chat = llm_chat
        self.llm = llm
        self.database = None
        self.busan_general_knowledge = None
        self.schemas = None

        # 초기 데이터 로드
        self._load_data()

    def _load_data(self):
        """JSON에 저장된 부산 관련 일반 지식과 CSV 파일들을 로드하여 SQLite에 저장한다."""
        # 1) JSON 파일 로드
        with open("data/text_data.json", "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        self.busan_general_knowledge = loaded_data["busan_general_knowledge"]
        self.schemas = loaded_data["schemas"]

        # 2) SQLite In-memory DB 생성
        conn = sqlite3.connect(":memory:")

        # 3) CSV 파일별로 테이블 생성
        csv_files = {
            "data/내국인 관심 관광지_수정.csv": "local_tourist_spots",
            "data/외국인 관심 관광지_수정.csv": "foreign_tourist_spots",
            "data/busan_restrau_20to24_witch_eng_data.csv": "restaurants",
        }

        for file_path, table_name in csv_files.items():
            df = pd.read_csv(file_path)
            df.to_sql(table_name, conn, index=False, if_exists="replace")

        # 4) In-memory DB를 멤버 변수에 연결
        self.database = conn

    def filter_csv_with_sql(self, query: str):
        """
        인자로 받은 SQL 쿼리를 SQLite에 실행하고, 결과를 Pandas DataFrame으로 반환한다.
        """
        try:
            result = pd.read_sql_query(query, self.database)
            # print("쿼리 실행 결과:", result)  # 필요 시 디버깅용 출력
            return result
        except Exception as e:
            return f"Error executing query: {e}"

    def select_vec_store(self, state: GraphState) -> GraphState:
        """
        사용자가 입력한 질문을 기반으로 적절한 데이터 소스(datasource)를 선택한다.
        """
        question = state["question"]
        # function call(Structured) 기능이 있는 LLM을 활용한다.
        structured_llm = self.llm_chat.with_structured_output(RouteQuery)

        # 라우팅을 위한 system 프롬프트
        system_prompt = """You are an expert at routing a user question to the appropriate data source.
Based on the category the question is referring to, route it to the relevant data source.

Return "foreign_tourist_spots" if the query asks for recommendation of tourist spots where foreign people like.
Return "local_tourist_spots" if the query asks for recommendation of tourist spots but there is no foreign-related keyword.
Return "restaurants" if the query requests restaurant recommendations.
Return "web search" if it is not related to tourist attractions or restaurants, such as weather or transportation."""

        # PromptTemplate 생성
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{question}")]
        )

        # Prompt + LLM 호출 -> datasource 결정
        router = prompt | structured_llm
        response = router.invoke(question)
        st.write("LLM의 data source 결정 결과:", response)

        # 결정된 datasource를 State에 업데이트
        return GraphState(vec_store=response.datasource)

    def retrieve_from_web(self, state: GraphState) -> GraphState:
        """
        Web 검색을 해야 할 경우(관광지, 맛집 이외의 질의)에 대한 처리 함수를 정의한다.
        실제 Web 검색 로직 대신, 필수 필드만 업데이트 한 뒤 State를 반환한다.
        """
        query = state["question"]
        # 웹 검색 로직(검색 API 등)을 붙일 수도 있음
        return GraphState(question=query)

    def next_step(self, state: GraphState) -> str:
        """
        'select_vec_store' 단계 이후 라우팅을 결정하기 위한 키를 반환한다.
        """
        return state["vec_store"]

    def end_bool(self, state: GraphState) -> GraphState:
        """
        'answer' 필드에 값이 있으면 end=True로 표시, 그렇지 않으면 False로 표시한다.
        """
        is_end = bool(state["answer"])
        return GraphState(end=is_end)

    def retrieve_with_sql_filtering(self, state: GraphState) -> GraphState:
        """
        선택된 datasource(테이블)에 대해 사용자의 질문을 해석하여 적절한 SQL 쿼리를 생성하고,
        쿼리 결과를 바탕으로 최종 답변을 생성한다.
        """
        datasource = state["vec_store"]
        question = state["question"]

        # SQL 쿼리 생성을 위한 프롬프트 템플릿
        sql_prompt = PromptTemplate(
            input_variables=[
                "question",
                "datasource",
                "columns",
                "description",
                "external_knowledge",
            ],
            template="""You are an expert in generating SQL queries. Based on the user's question, external knowledge and
the specified data source, generate an SQL query.

Data source: {datasource}
Table description: {description}
Columns description: {columns}
External knowledge: {external_knowledge}

User question: {question}

Ensure the query matches the schema of the data source and retrieves
the most relevant information. Use 'LIKE' for text columns.
Use 'IN' for multiple value filtering. Provide only the SQL query as output.""",
        )

        # SQL 생성 체인
        sql_chain = LLMChain(llm=self.llm_back, prompt=sql_prompt)

        # 데이터 소스 스키마 정보 준비
        schema_details = self.schemas.get(datasource, {})
        columns = ", ".join(schema_details.get("columns", []))
        description = schema_details.get("description", "No description available.")

        # LLM을 활용한 SQL 쿼리 생성
        sql_query = sql_chain.run({
            "question": question,
            "datasource": datasource,
            "columns": columns,
            "description": description,
            "external_knowledge": self.busan_general_knowledge,
        })

        filtered_data = self.filter_csv_with_sql(sql_query)

        if isinstance(filtered_data, pd.DataFrame) and not filtered_data.empty:
            result_text = filtered_data.to_string(index=False)
        else:
            result_text = "No matching records found."

        # 최종 답변 생성을 위한 prompt
        final_query = f"""Based on the user's question: {question}
and the following retrieved information:
{result_text}
Please provide a detailed and concise answer."""
        final_answer = self.llm(final_query)

        return GraphState(answer=final_answer)

    def buildgraph(self):
        """
        StateGraph를 구성하고, 초기 진입점(entry point)을 설정한다.
        이후 compile() 호출로 최종 callable 객체를 반환한다.
        """
        # StateGraph의 제네릭 타입에 GraphState를 명시
        workflow = StateGraph(GraphState)

        # 노드 추가
        workflow.add_node("select_vec", self.select_vec_store)
        workflow.add_node("retrieve_with_sql_filtering", self.retrieve_with_sql_filtering)
        workflow.add_node("retrieve_from_web", self.retrieve_from_web)

        # 각 노드의 실행이 끝나면 종료(END)로 가는 엣지
        workflow.add_edge("retrieve_with_sql_filtering", END)
        workflow.add_edge("retrieve_from_web", END)

        # 분기 조건(conditional edges)
        workflow.add_conditional_edges(
            "select_vec",
            self.next_step,
            {
                "local_tourist_spots": "retrieve_with_sql_filtering",
                "foreign_tourist_spots": "retrieve_with_sql_filtering",
                "restaurants": "retrieve_with_sql_filtering",
                "web search": "retrieve_from_web",
            },
        )

        # 그래프 시작점 설정
        workflow.set_entry_point("select_vec")

        # 최종 Graph를 컴파일하여 반환
        app = workflow.compile()
        return app
