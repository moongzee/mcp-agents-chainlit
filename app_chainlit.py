# app_chainlit.py (도구 호출 및 최종 응답 오류 최종 해결 버전)

# --- 1. 기본 라이브러리 임포트 ---
import chainlit as cl
import json
import pandas as pd
import io
from datetime import datetime
import traceback
import time
import sqlite3
import os
import hashlib
from typing import List, Any, Dict, Optional
from mcp import ClientSession
import re
import codecs

# --- 2. LangChain 관련 라이브러리 임포트 ---
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from modules.utils import trimming_node
# Custom chain: summarization_node → agent_core
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
#from moduels.handlers import KoreanLangGraphCallbackHandler()
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import anthropic


# --- 3. MCP 타입 임포트 추가 ---
from mcp.types import CallToolResult, TextContent

# --- 3. 설정 파일 로드 및 데이터베이스 초기화 ---
from dotenv import load_dotenv
load_dotenv(override=True)

try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    SYSTEM_PROMPT = "You are a helpful assistant."
    print("경고: system_prompt.txt 파일을 찾을 수 없어 기본 프롬프트를 사용합니다.")

# 채팅 기록 저장을 위한 SQLite 데이터베이스 초기화
DB_PATH = "chat_history.db"

def init_database():
    """채팅 기록 저장용 데이터베이스 초기화"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 사용자 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 채팅 세션 테이블 (사용자 ID 추가)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """비밀번호 해시 생성"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email: str, password: str, display_name: str = None) -> bool:
    """새 사용자 생성"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        user_id = hashlib.md5(email.encode()).hexdigest()
        password_hash = hash_password(password)
        display_name = display_name or email.split('@')[0]  # 이메일의 @ 앞부분을 기본 이름으로
        
        cursor.execute('''
            INSERT INTO users (id, username, password_hash, display_name)
            VALUES (?, ?, ?, ?)
        ''', (user_id, email, password_hash, display_name))
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """사용자 인증"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    
    cursor.execute('''
        SELECT id, username, display_name
        FROM users
        WHERE username = ? AND password_hash = ?
    ''', (email, password_hash))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {
            "id": user[0],
            "username": user[1],  # 이메일
            "display_name": user[2]
        }
    return None

def save_chat_session(session_id: str, user_id: str, title: str):
    """새로운 채팅 세션 저장"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO chat_sessions (id, user_id, title, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    ''', (session_id, user_id, title))
    
    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str):
    """메시지 저장"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_messages (session_id, role, content)
        VALUES (?, ?, ?)
    ''', (session_id, role, content))
    
    # 세션의 메시지 카운트 업데이트
    cursor.execute('''
        UPDATE chat_sessions 
        SET message_count = message_count + 1, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (session_id,))
    
    conn.commit()
    conn.close()

def get_chat_sessions(user_id: str):
    """사용자의 모든 채팅 세션 목록 가져오기"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, title, created_at, message_count
        FROM chat_sessions
        WHERE user_id = ?
        ORDER BY updated_at DESC
        LIMIT 20
    ''', (user_id,))
    
    sessions = cursor.fetchall()
    conn.close()
    return sessions

def get_session_messages(session_id: str):
    """특정 세션의 메시지들 가져오기"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT role, content, timestamp
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
    ''', (session_id,))
    
    messages = cursor.fetchall()
    conn.close()
    return messages

def generate_session_title(first_message: str) -> str:
    """첫 번째 메시지를 기반으로 세션 제목 생성"""
    # 첫 번째 메시지의 처음 30자를 제목으로 사용
    title = first_message.strip()[:30]
    if len(first_message) > 30:
        title += "..."
    return title

# 데이터베이스 초기화
init_database()

# --- 4. 로그인 시스템 (데이터베이스 함수들 다음에 배치) ---

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """사용자 인증 콜백"""
    print(f"DEBUG: 로그인 시도 - 사용자명: {username}")
    
    user_data = authenticate_user(username, password)
    print(f"DEBUG: 인증 결과: {user_data}")
    
    if user_data:
        print(f"DEBUG: 로그인 성공 - {user_data['display_name']}")
        
        # 🔧 Chainlit User 객체 생성 - 모든 방법 시도
        user = cl.User(
            identifier=user_data["username"],  # 이메일을 identifier로 사용
            metadata={
                "user_id": user_data["id"],  # 실제 DB ID
                "username": user_data["username"],  # 이메일
                "display_name": user_data["display_name"],  # 표시명
                "db_id": user_data["id"]  # 백업용 ID
            }
        )
        
        # display_name 속성 직접 설정 시도
        try:
            user.display_name = user_data["display_name"]
            print(f"DEBUG: display_name 직접 설정 완료: {user_data['display_name']}")
        except Exception as e:
            print(f"DEBUG: display_name 설정 실패: {e}")
        
        print(f"DEBUG: 최종 생성된 User 객체:")
        print(f"  - identifier: {user.identifier}")
        print(f"  - display_name: {getattr(user, 'display_name', 'None')}")
        print(f"  - metadata: {user.metadata}")
        
        return user
    else:
        print(f"DEBUG: 로그인 실패 - 사용자명 또는 비밀번호 불일치")
        return None

# --- 6. MCP 연결 핸들러 정의 ---

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    
    try:
        tool_metadatas = await session.list_tools()
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = [
            {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
            for t in tool_metadatas.tools
        ]
        cl.user_session.set("mcp_tools", mcp_tools)
        
        model = cl.user_session.get("model")
        if not model:
            model = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.1,
                streaming=True
            )
            cl.user_session.set("model", model)

        from langchain_core.tools import Tool
        all_langchain_tools = []
        for conn_name, tools in mcp_tools.items():
            for tool_info in tools:
                async def tool_func(tool_input: Any, conn_name=conn_name, tool_name=tool_info['name']):
                    mcp_session, _ = cl.context.session.mcp_sessions.get(conn_name)
                    if not mcp_session: return f"Error: MCP session for '{conn_name}' not found."
                    
                    if not isinstance(tool_input, dict):
                        tool_input = {"query": tool_input}

                    try:
                        tool_result = await mcp_session.call_tool(tool_name, tool_input)
                        return str(tool_result)
                    except Exception as e: return f"Error calling tool {tool_name}: {e}"

                all_langchain_tools.append(
                    Tool(
                        name=tool_info['name'],
                        description=tool_info['description'],
                        func=tool_func,
                        coroutine=tool_func
                    )
                )

        
        # 세션별 메모리를 담아둘 저장소 (in-memory dict)
        session_memory_store = {}
        
        # 🔧 맥락 유지형 요약 노드 - 안전한 수동 요약 방식
        async def summary_node(state):
            """맥락을 유지하면서 메시지를 제한하는 노드"""
            messages = state.get("messages", [])
            session_id = cl.context.session.id
            
            # 세션 메모리 가져오기
            if session_id not in session_memory_store:
                new_memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history"
                )
                session_memory_store[session_id] = new_memory
            
            memory = session_memory_store[session_id]
            
            # 시스템 메시지와 일반 메시지 분리
            system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
            non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
            
            try:
                memory_vars = memory.load_memory_variables({})
                chat_history = memory_vars.get("chat_history", [])
                
                # 메시지 처리
                history_to_add = []
                
                if isinstance(chat_history, list) and chat_history:
                    # 대화가 길어지면 수동 요약 수행
                    if len(chat_history) > 6:  # 3번의 대화 (6개 메시지) 이상 시
                        # 앞의 4개 메시지를 요약으로 변환
                        old_messages = chat_history[:4]
                        recent_messages = chat_history[4:]
                        
                        # 🔧 API 호출하여 의미있는 요약 생성
                        try:
                            summary_content = await create_intelligent_summary(old_messages, model)
                            summary_msg = AIMessage(content=f"[이전 대화 요약]\n{summary_content}")
                            
                            history_to_add = [summary_msg] + recent_messages
                            print(f"[DEBUG] 지능형 요약 적용: {len(chat_history)} → 요약+{len(recent_messages)}개 메시지")
                        except Exception as summary_error:
                            print(f"[DEBUG] 요약 생성 실패: {summary_error}")
                            # 요약 실패 시 최근 4개 메시지만 유지
                            history_to_add = chat_history[-4:] if len(chat_history) > 4 else chat_history
                            print(f"[DEBUG] 요약 실패로 최근 메시지만 유지: {len(history_to_add)}개")
                    else:
                        # 아직 길지 않으면 전체 유지
                        history_to_add = chat_history
                
                # 최종 메시지 구성: [시스템] + [요약+최근히스토리] + [현재새메시지]
                combined_messages = system_messages + history_to_add + non_system_messages
                
                print(f"[DEBUG] summary_node: 시스템={len(system_messages)}, 히스토리={len(history_to_add)}, 새메시지={len(non_system_messages)}, 총={len(combined_messages)}")
                
                return {"messages": combined_messages}
                
            except Exception as e:
                print(f"[DEBUG] summary_node 오류: {e}")
                # 오류 시 새 메시지만 반환 (히스토리 없이)
                return {"messages": system_messages + non_system_messages}
        
        # 🔧 지능형 요약 생성 함수
        async def create_intelligent_summary(messages, model):
            """API를 호출하여 2000자 이내의 의미있는 요약 생성"""
            
            # 🔧 요약 전용 모델 생성 (경제적이고 빠른 haiku 모델 사용)
            summary_model = ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=0.1,
                streaming=False  # 요약은 스트리밍 불필요
            )
            
            # 메시지들을 텍스트로 변환
            conversation_text = ""
            for i, msg in enumerate(messages):
                if hasattr(msg, 'content'):
                    role = "사용자" if i % 2 == 0 else "AI"
                    conversation_text += f"{role}: {msg.content}\n\n"
            
            # 요약 프롬프트
            summary_prompt = f"""아래 대화 내용을 2000자 이내로 간결하고 의미있게 요약해주세요. 
주요 질문, 핵심 답변, 중요한 맥락을 포함하되 너무 길지 않게 작성해주세요.

[대화 내용]
{conversation_text}

[요약 요구사항]
- 2000자 이내
- 주요 질문과 답변의 핵심 내용 포함  
- 향후 대화에 필요한 중요 맥락 보존
- 간결하고 명확하게 작성"""

            try:
                # 🔧 haiku 모델로 요약 생성
                summary_response = await summary_model.ainvoke([HumanMessage(content=summary_prompt)])
                summary_content = summary_response.content
                
                # 2000자 초과 시 강제 자르기
                if len(summary_content) > 2000:
                    summary_content = summary_content[:1950] + "..."
                
                print(f"[DEBUG] 요약 생성 완료 (haiku): {len(conversation_text)}자 → {len(summary_content)}자")
                return summary_content
                
            except Exception as e:
                print(f"[DEBUG] 요약 API 호출 실패: {e}")
                # 실패 시 간단한 수동 요약으로 폴백
                fallback_summary = f"이전 대화 {len(messages)}개 메시지 (API 요약 실패로 생략)"
                return fallback_summary
        
        # 메모리 객체 생성 및 저장
        def get_session_history(session_id: str):
            if session_id not in session_memory_store:
                new_memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history"
                )
                session_memory_store[session_id] = new_memory
            return session_memory_store[session_id]
        
        memory = get_session_history(cl.context.session.id)
        cl.user_session.set("memory", memory)
        cl.user_session.set("session_memory_store", session_memory_store)

        # 🔧 Agent core 생성 - checkpointer=None으로 설정하여 자체 메모리 관리 비활성화
        agent_core = create_react_agent(
            model,
            all_langchain_tools,
            checkpointer=None,  # 자체 메모리 관리 비활성화
            prompt=SYSTEM_PROMPT,
        )

        # 🔧 맥락 유지형 에이전트: summary_node → agent_core
        context_preserving_agent = (
            RunnableLambda(summary_node) |  # 맥락 유지 요약 (async 지원)
            agent_core  # ReAct 반복 수행
        )

        # 세션에 agent 저장
        cl.user_session.set("agent", context_preserving_agent)

    except Exception as e:
        error_msg = f"MCP 연결 처리 중 오류 발생: {e}"
        print(error_msg)
        traceback.print_exc()
        await cl.Message(content=error_msg).send()

@cl.on_mcp_disconnect
async def on_mcp_disconnect(connection):
    await cl.Message(content=f"MCP 연결 '{connection.name}'이(가) 종료되었습니다.").send()

# --- 7. Chainlit 메시지 핸들러 ---

@cl.on_settings_update
async def setup_agent(settings):
    """설정 업데이트 시 에이전트 재설정"""
    pass

@cl.on_chat_start
async def on_chat_start():
    """채팅 시작 시 초기화 및 사이드바 설정"""
    # 현재 사용자 정보 가져오기
    user = cl.user_session.get("user")
    if not user:
        user = cl.context.session.user
        cl.user_session.set("user", user)
    
    if not user:
        await cl.Message(content="로그인이 필요합니다. 다시 로그인해주세요.").send()
        return
    
    cl.user_session.set("mcp_tools", {})
    cl.user_session.set("message_count", 0)
    
    # 환영 메시지 - 표시명 우선, 없으면 사용자명, 없으면 기본값
    display_name = user.metadata.get("display_name") or user.metadata.get("username") or "사용자"
    
    # 🔧 사용자 정보 업데이트 시도
    try:
        # Chainlit에서 사용자 표시명을 강제로 설정
        user.display_name = display_name
        
    except Exception as e:
        print(f"DEBUG: 사용자 표시명 설정 실패: {e}")
    
    await cl.Message(content=f"안녕하세요 {display_name}님! NOA 입니다. 무엇이든 물어보세요 😊").send()



@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(anthropic.APIStatusError)
)
async def astream_events_with_retry(agent, message, config):
    async for event in agent.astream_events(
        {"messages": [message]},
        config=config,
        version="v2"
    ):
        yield event


@cl.on_chat_end
async def on_chat_end():
    """채팅 종료 시 세션 저장"""
    user = cl.user_session.get("user") or cl.context.session.user
    if not user:
        return
        
    session_id = cl.context.session.id
    message_count = cl.user_session.get("message_count", 0)
    
    if message_count > 0:
        # 첫 번째 메시지를 기반으로 제목 생성
        first_message = cl.user_session.get("first_message", "새로운 대화")
        title = generate_session_title(first_message)
        
        # 실제 user_id 사용 (메타데이터에서 가져옴)
        actual_user_id = user.metadata.get("user_id", user.identifier)
        save_chat_session(session_id, actual_user_id, title)


@cl.on_message
async def on_message(message: cl.Message):
    """사용자 메시지 처리"""
    # 현재 사용자 확인
    user = cl.user_session.get("user") or cl.context.session.user
    if not user:
        await cl.Message(content="로그인이 필요합니다. 다시 로그인해주세요.").send()
        return
    
    # 메시지 카운트 및 첫 메시지 저장
    message_count = cl.user_session.get("message_count", 0)
    message_count += 1
    cl.user_session.set("message_count", message_count)
    
    if message_count == 1:
        # 첫 번째 메시지 저장 (세션 제목용)
        cl.user_session.set("first_message", message.content)
    
    # 메시지 DB에 저장
    session_id = cl.context.session.id
    save_message(session_id, "user", message.content)
    
    agent = cl.user_session.get("agent")
    if not agent:
        await cl.Message(content="에이전트가 아직 설정되지 않았습니다. 먼저 MCP 서버에 연결해주세요.").send()
        return

    # 🔧 기본 Chainlit 콜백 사용
    config = RunnableConfig(
        recursion_limit=100,
        thread_id=cl.context.session.id,
        callbacks=[cl.LangchainCallbackHandler()],
    )

    # 최종 응답 메시지 처리
    final_msg = cl.Message(content="", author="Assistant")
    has_streamed_content = False
    all_final_responses = []
    table_elements = None
    
    try:
        # Anthropic Overloaded 오류 시 최대 5회 자동 재시도(backoff)
        async for event in astream_events_with_retry(
            agent, HumanMessage(content=message.content), config
        ):
            kind = event["event"]

            if kind == "on_llm_start":
                if not final_msg.id:
                    await final_msg.send()

            elif kind == "on_llm_stream" or kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                
                # 🔧 요약 내용 필터링 - 사용자에게 보이지 않도록 처리
                chunk_content = ""
                if isinstance(chunk.content, str):
                    chunk_content = chunk.content
                elif isinstance(chunk.content, list):
                    for part in chunk.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            chunk_content += part.get("text", "")
                
                # 요약 내용은 스트리밍하지 않음
                if chunk_content and not chunk_content.startswith("[이전 대화 요약]"):
                    if chunk_content.strip():
                        await final_msg.stream_token(chunk_content)
                        has_streamed_content = True

            elif kind == "on_tool_end":
                step = cl.user_session.get(f"tool_step_{event['run_id']}")
                tool_output_object = event["data"].get("output")
                json_string_to_parse = None
                raw_content = None
                if isinstance(tool_output_object, ToolMessage):
                    raw_content = tool_output_object.content
                if isinstance(raw_content, CallToolResult):
                    for content_item in raw_content.content:
                        if isinstance(content_item, TextContent):
                            json_string_to_parse = content_item.text
                            break
                elif isinstance(raw_content, str):
                    match = re.search(r"text='(.*?)', annotations=None", raw_content, re.DOTALL)
                    if match:
                        json_string_to_parse = match.group(1)
                    else:
                        print("정규표현식 매칭 실패! 파싱 시도하지 않음.")
                        json_string_to_parse = None

                if step:
                    step.output = json_string_to_parse or str(raw_content)

                if event["name"] == "run_cortex_agents" and json_string_to_parse and json_string_to_parse.strip():
                    try:
                        unescaped = codecs.decode(json_string_to_parse, "unicode_escape")
                        if isinstance(unescaped, str):
                            unescaped = unescaped.encode("latin1").decode("utf-8")
                        tool_result_json = json.loads(unescaped)
                        chart_data = tool_result_json.get("results", {})
                        elements = []
                        if chart_data and chart_data.get("data"):
                            columns = [col["name"] for col in chart_data["resultSetMetaData"]["rowType"]]
                            df = pd.DataFrame(chart_data["data"], columns=columns)
                            elements.append(cl.Dataframe(data=df, name="상세 데이터", display="inline"))
                            string_io = io.StringIO()
                            df.to_csv(string_io, index=False, encoding='utf-8-sig')
                            csv_bytes = string_io.getvalue().encode('utf-8-sig')
                            elements.append(cl.File(
                                name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                content=csv_bytes,
                                display="inline"
                            ))
                        if elements and table_elements is None:
                            table_elements = elements
                    except Exception as e:
                        print(f"'{event['name']}' 툴 결과 처리 중 오류 발생:")
                        traceback.print_exc()

                if step:
                    await step.update()

            elif kind == "on_llm_end" or kind == "on_chat_model_end":
                output = event["data"].get("output")
                if output:
                    final_answer = ""
                    if hasattr(output, 'content'):
                        if isinstance(output.content, list):
                            for part in output.content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    final_answer += part.get("text", "")
                        elif isinstance(output.content, str):
                            final_answer = output.content
                    
                    # 🔧 요약 내용 필터링 - 최종 응답에서도 요약 제외
                    if final_answer.strip() and not final_answer.startswith("[이전 대화 요약]"):
                        all_final_responses.append(final_answer)
        #--- for loop 종료 ---
    except anthropic.APIStatusError as e:
        if getattr(e, 'error', {}).get('type') == 'overloaded_error':
            await cl.Message(content="Anthropic 서버 과부하로 일시적으로 응답이 지연되고 있습니다. 잠시 후 다시 시도해주세요.").send()
            return
    except Exception as e:
        await cl.Message(content=f"알수없는 오류가 발생했습니다: {str(e)}").send()
        return

    if table_elements:
        await cl.Message(
            content="DATA",
            elements=table_elements,
        ).send()
        table_elements = None

    final_response = ""
    if not has_streamed_content and all_final_responses:
        # 🔧 스트리밍되지 않은 응답이 있는 경우 출력
        final_response = "\n\n".join(all_final_responses)
        # 요약 내용 제외 처리
        if not final_response.startswith("[이전 대화 요약]"):
            await cl.Message(content=final_response, author="Assistant").send()
    elif not has_streamed_content and not all_final_responses:
        final_response = "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다."
        await cl.Message(content=final_response, author="Assistant").send()
    else:
        # 🔧 스트리밍된 경우 final_msg 내용 확인 및 완료 처리
        final_response = final_msg.content
        
        # 스트리밍이 비어있고 all_final_responses가 있으면 추가
        if not final_response.strip() and all_final_responses:
            additional_content = "\n\n".join(all_final_responses)
            # 요약 내용이 아닌 경우만 추가
            if not additional_content.startswith("[이전 대화 요약]"):
                await final_msg.stream_token(additional_content)
                final_response = final_msg.content

    if final_response:
        # 메시지 끝의 공백 제거 (Anthropic API 에러 방지)
        clean_input = message.content.strip()
        clean_output = final_response.strip()
        
        # 🔧 요약 내용은 메모리에 저장하지 않음
        if clean_output.startswith("[이전 대화 요약]"):
            print(f"[DEBUG] 요약 내용 감지 - 메모리 저장 건너뜀")
            await final_msg.update()
            return
        
        try:
            save_message(session_id, "assistant", clean_output)
            
            # 세션 메모리 스토어에서 실제 메모리 객체 가져오기
            session_memory_store = cl.user_session.get("session_memory_store", {})
            if session_id in session_memory_store:
                actual_memory = session_memory_store[session_id]
                
                # 🔧 안전한 메모리 저장 처리
                try:
                    # 빈 메시지 확인
                    if not clean_input.strip() or not clean_output.strip():
                        print(f"[DEBUG] 빈 메시지 감지 - 메모리 저장 건너뜀")
                    else:
                        # 🔧 간단한 ConversationBufferMemory 사용
                        try:
                            actual_memory.save_context(
                                {"input": clean_input},
                                {"output": clean_output}
                            )
                            print(f"[DEBUG] 메모리 저장 완료")
                        except Exception as save_error:
                            print(f"[DEBUG] save_context 실패: {save_error}")
                            # ConversationBufferMemory는 chat_memory를 직접 사용
                            try:
                                actual_memory.chat_memory.add_user_message(clean_input)
                                actual_memory.chat_memory.add_ai_message(clean_output)
                                print("[DEBUG] chat_memory 직접 저장 완료")
                            except Exception as chat_error:
                                print(f"[DEBUG] chat_memory 저장도 실패: {chat_error}")
                        
                except Exception as save_error:
                    print(f"[DEBUG] 메모리 저장 처리 중 오류: {save_error}")
            else:
                print(f"[DEBUG] 세션 메모리를 찾을 수 없음: {session_id}")

        except Exception as memory_error:
            print(f"[DEBUG] 메모리 저장 중 전체 오류: {memory_error}")

    await final_msg.update()


class AsyncConversationBufferMemory:
    """ConversationBufferMemory를 RunnableWithMessageHistory에서 사용할 수 있도록 하는 래퍼"""
    def __init__(self, memory):
        self.memory = memory

    async def aget_messages(self):
        variables = self.memory.load_memory_variables({})
        chat_history = variables.get("chat_history", [])
        # 리스트 형태로 반환 (ConversationBufferMemory는 항상 리스트)
        return chat_history if isinstance(chat_history, list) else []
    
    async def aadd_messages(self, messages, *args, **kwargs):
        # 동기 메서드를 비동기로 감싸서 호출
        print(f"[DEBUG][aadd_messages] called with {len(messages)} messages")
        for msg in messages:
            if hasattr(msg, 'type'):
                if msg.type == "human":
                    self.memory.save_context({"input": msg.content}, {"output": ""})
                elif msg.type == "ai":
                    # 이전 입력이 없으면 빈 입력으로 저장
                    self.memory.save_context({"input": ""}, {"output": msg.content})

    def __getattr__(self, name):
        return getattr(self.memory, name)


