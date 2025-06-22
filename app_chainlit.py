# app_chainlit.py (메모리 관리 로직 및 함수 시그니처 수정 완료 버전)

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
# Custom chain: summarization_node → agent_core
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
#from moduels.handlers import KoreanLangGraphCallbackHandler()
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import anthropic
import asyncio

# 네트워크 관련
import httpx
from httpx import ReadError as HttpxReadError
import httpcore
from httpcore import ReadError as HttpcoreReadError


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

# 🔧 전역 세션 메모리 스토어 추가
session_memory_store = {}

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

# 🔧 에러 발생 시 대화 초기화 안내를 위한 유틸리티 함수들
async def send_error_with_reset_guidance(error_message: str, error_type: str = "일반"):
    """에러 메시지와 함께 대화 초기화 안내를 보내는 함수"""
    
    # 🎨 에러 타입별 맞춤 메시지
    if error_type == "네트워크":
        reset_guidance = """
🔄 **해결 방법:**
1. **새로운 대화 시작**: 브라우저를 새로고침하거나 새 탭에서 대화를 시작해보세요
2. **잠시 후 재시도**: 네트워크가 안정된 후 다시 시도해주세요
3. **짧은 메시지로 시도**: 긴 질문 대신 짧은 메시지로 시작해보세요

💡 **팁**: 새로고침 후에도 같은 문제가 반복되면, 몇 분 후에 다시 접속해보세요.
"""
    elif error_type == "API":
        reset_guidance = """
🔄 **해결 방법:**
1. **새로운 대화 시작**: 페이지를 새로고침하여 새로운 대화를 시작해주세요
2. **잠시 대기**: API 서버가 안정화될 때까지 5-10분 정도 기다려주세요
3. **간단한 질문부터**: 복잡한 작업보다는 간단한 질문부터 시작해보세요

💡 **팁**: 서버 과부하가 해소되면 정상적으로 이용하실 수 있습니다.
"""
    elif error_type == "메모리":
        reset_guidance = """
🔄 **해결 방법:**
1. **즉시 새로고침**: 브라우저를 새로고침하여 메모리를 초기화해주세요
2. **대화 기록 정리**: 너무 긴 대화는 새로운 세션에서 시작하는 것이 좋습니다
3. **단계별 질문**: 복합적인 질문보다는 단계별로 나누어 질문해주세요

💡 **팁**: 새 대화에서는 이전 내용을 간단히 요약해서 다시 질문해주세요.
"""
    else:  # 일반 에러
        reset_guidance = """
🔄 **해결 방법:**
1. **페이지 새로고침**: F5키나 브라우저 새로고침 버튼을 눌러주세요
2. **새 탭에서 시작**: 새로운 브라우저 탭에서 대화를 시작해보세요
3. **잠시 후 재시도**: 몇 분 후에 다시 시도해보세요

💡 **지속적인 문제**: 계속 같은 오류가 발생하면 브라우저 캐시를 삭제해보세요.
"""
    
    # 🚨 에러 메시지 + 안내사항 결합
    full_message = f"""⚠️ **오류 발생**: {error_message}

{reset_guidance}

🔧 **빠른 해결**: 지금 바로 **Ctrl+F5** (Windows) 또는 **Cmd+R** (Mac)을 눌러 새로고침해보세요!
"""
    
    await cl.Message(content=full_message).send()

async def send_critical_error_guidance():
    """심각한 에러 발생 시 상세한 안내를 보내는 함수"""
    
    critical_guidance = """
🚨 **시스템 오류가 발생했습니다**

이 문제를 해결하기 위해 다음 단계를 따라해주세요:

### 🔄 **즉시 해결 방법**
1. **브라우저 새로고침**: `F5` 또는 `Ctrl+F5`를 눌러주세요
2. **새 탭 열기**: 새로운 브라우저 탭에서 다시 접속해주세요
3. **5분 대기**: 시스템이 안정화될 때까지 잠시 기다려주세요

### 🛠️ **추가 해결 방법**
- **브라우저 캐시 삭제**: 설정 > 개인정보 > 인터넷 사용 기록 삭제
- **다른 브라우저 사용**: Chrome, Firefox, Safari 등 다른 브라우저 시도
- **인터넷 연결 확인**: Wi-Fi 또는 네트워크 연결 상태 점검

### 💬 **새 대화 시작하기**
문제가 해결되면 간단한 인사말("안녕하세요")로 새 대화를 시작해보세요!

---
*문제가 계속 발생하면 잠시 후 다시 시도해주세요. 시스템이 자동으로 복구됩니다.*
"""
    
    await cl.Message(content=critical_guidance).send()


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((HttpxReadError, HttpcoreReadError, ConnectionError))
)
async def create_intelligent_summary_silent(messages_to_summarize, existing_summary=None):
    """API를 호출하여 의미있는 요약 생성"""
    summary_model = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.1,
        streaming=False,
        timeout=60.0,
        max_retries=2
    )

    conversation_text = "\n\n".join(
        f"{'사용자' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in messages_to_summarize
    )

    if existing_summary:
        summary_prompt = f"""다음은 지금까지의 대화 요약과 새로 추가된 대화 내용입니다. 기존 요약을 바탕으로 새로운 대화 내용을 자연스럽게 통합하여 업데이트된 전체 요약본을 만들어주세요.

[기존 요약]
{existing_summary}

[새로운 대화 내용]
{conversation_text}

[업데이트된 전체 요약]:"""
    else:
        summary_prompt = f"""다음 대화 내용을 한국어로 간결하게 핵심만 요약해주세요. 이 요약은 대화의 맥락을 유지하기 위한 내부 정보로 사용됩니다.

[대화 내용]
{conversation_text}

[요약]:"""

    try:
        summary_response = await summary_model.ainvoke([HumanMessage(content=summary_prompt)])
        summary_content = summary_response.content.strip()
        print(f"[DEBUG] 요약 생성/업데이트 완료.")
        return summary_content
    except Exception as e:
        print(f"[DEBUG] 요약 API 호출 실패: {e}")
        return existing_summary or f"이전 대화 {len(messages_to_summarize)}개 (요약 생성 중 오류 발생)"

async def preprocess_with_silent_summary(input_data):
    """누적 요약 기능이 포함된 전처리기"""
    global session_memory_store
    session_id = cl.context.session.id

    if session_id not in session_memory_store:
        session_memory_store[session_id] = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history"
        )
    memory = session_memory_store[session_id]

    user_message = input_data.get("messages", [])
    system_prompt_message = [SystemMessage(content=SYSTEM_PROMPT)]

    try:
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])

        existing_summary = None
        # 대화 기록의 첫 번째가 시스템 메시지이고, 요약본이라면 추출
        if chat_history and isinstance(chat_history[0], SystemMessage):
            # content가 문자열인지 확인
            if isinstance(chat_history[0].content, str) and chat_history[0].content.startswith("--- 누적 요약 ---"):
                existing_summary = chat_history[0].content
                chat_history = chat_history[1:]  # 실제 대화만 남김
                print(f"[DEBUG] 기존 요약 발견. 실제 대화 길이: {len(chat_history)}")

        # 실제 대화 기록이 4개 이상일 때만 요약 로직 실행
        if len(chat_history) >= 4:
            messages_to_summarize = chat_history[:-2]
            recent_messages = chat_history[-2:]

            print(f"[DEBUG] 누적 요약 시작. 기존 요약:{'있음' if existing_summary else '없음'}, 요약 대상:{len(messages_to_summarize)}, 유지:{len(recent_messages)}")
            
            # 기존 요약과 함께 새로운 요약 생성
            updated_summary_content = await create_intelligent_summary_silent(messages_to_summarize, existing_summary)
            
            # 새로운 요약 메시지와 최신 대화로 메모리를 완전히 재구성
            summary_message = SystemMessage(content=f"--- 누적 요약 ---\n{updated_summary_content}")
            memory.chat_memory.messages = [summary_message] + recent_messages
            
            print(f"[DEBUG] 메모리 재구성 완료. 현재 메모리: 요약 1개 + 최신 대화 {len(recent_messages)}개")
            
            # LLM에는 현재 턴의 요약본과 최신 대화를 전달
            messages_for_llm = [summary_message] + recent_messages

        else:
            # 요약할 만큼 길지 않으면, 기존 요약(있다면)과 전체 기록을 그대로 사용
            messages_for_llm = ([SystemMessage(content=existing_summary)] if existing_summary else []) + chat_history

        final_messages_to_agent = system_prompt_message + messages_for_llm + user_message
        
        print(f"[DEBUG] 최종 전달 메시지 수: {len(final_messages_to_agent)}")
        return {"messages": final_messages_to_agent}

    except Exception as e:
        print(f"[DEBUG] 전처리 중 심각한 오류 발생: {e}")
        traceback.print_exc()
        return {"messages": system_prompt_message + user_message}

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


def create_robust_anthropic_model():
    """네트워크 안정성을 위한 강화된 Anthropic 모델 생성"""
    
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.1,
        streaming=True,
        max_tokens=16000,
        max_retries=3,  # API 레벨 재시도
        timeout=180.0,  # 전체 요청 타임아웃 3분
    )
    return model


# --- 6. MCP 연결 핸들러 정의 ---
@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    global session_memory_store  # 전역 변수 사용
    
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
            model = create_robust_anthropic_model()
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

        # 🔧 순수 Agent core만 생성 (요약은 전처리에서 처리)
        agent_core = create_react_agent(
            model,
            all_langchain_tools,
            checkpointer=None,
            prompt=SYSTEM_PROMPT,
        )

        # 🔧 순수 agent 저장 (RunnableLambda 체인 없음)
        cl.user_session.set("agent", agent_core)
        
        print("[DEBUG] MCP 연결 완료 - 순수 agent 설정됨")

    except Exception as e:
        error_msg = f"MCP 연결 처리 중 오류가 발생했습니다: {e}"
        print(error_msg)
        traceback.print_exc()
        
        # MCP 연결 실패 시 안내
        await send_error_with_reset_guidance(
            "도구 연결에 실패했습니다. 일부 기능이 제한될 수 있습니다.", 
            "일반"
        )

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
    retry=retry_if_exception_type((
        anthropic.APIStatusError,
        HttpxReadError,  # httpx 읽기 에러
        HttpcoreReadError,  # httpcore 읽기 에러
        anthropic.APIConnectionError,  # API 연결 에러
        anthropic.APITimeoutError,  # API 타임아웃
        ConnectionError,  # 일반 연결 에러
        TimeoutError  # 일반 타임아웃
    ))
)
async def astream_events_with_retry(agent, messages, config):
    """강화된 재시도 로직으로 스트림 이벤트 처리"""
    try:
        async for event in agent.astream_events(
            {"messages": messages},
            config=config,
            version="v2"
        ):
            yield event
    except (HttpxReadError, HttpcoreReadError) as e:
        print(f"[DEBUG] 네트워크 읽기 에러 발생: {e}")
        # 짧은 대기 후 재시도
        await asyncio.sleep(1)
        raise  # 재시도 데코레이터가 처리

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
    """사용자 메시지 처리 - 메모리 저장 순서 수정"""
    global session_memory_store
    
    # 현재 사용자 확인
    user = cl.user_session.get("user") or cl.context.session.user
    if not user:
        await send_error_with_reset_guidance(
            "로그인이 필요합니다. 다시 로그인해주세요.", 
            "일반"
        )
        return
    
    # 메시지 카운트 및 첫 메시지 저장
    message_count = cl.user_session.get("message_count", 0)
    message_count += 1
    cl.user_session.set("message_count", message_count)
    
    if message_count == 1:
        cl.user_session.set("first_message", message.content)
    
    # 메시지 DB에 저장
    session_id = cl.context.session.id
    try:
        save_message(session_id, "user", message.content)
    except Exception as db_error:
        print(f"[DEBUG] DB 저장 실패: {db_error}")
    
    agent = cl.user_session.get("agent")
    if not agent:
        await send_error_with_reset_guidance(
            "에이전트가 아직 설정되지 않았습니다. MCP 서버 연결을 확인해주세요.", 
            "일반"
        )
        return

    # 🔧 1. 먼저 요약을 전처리에서 조용히 처리
    input_data = {"messages": [HumanMessage(content=message.content)]}
    processed_input = await preprocess_with_silent_summary(input_data)
    
    # 🔧 2. 처리된 메시지로 agent 실행
    config = RunnableConfig(
        recursion_limit=100,
        thread_id=cl.context.session.id,
        callbacks=[cl.LangchainCallbackHandler()],
    )

    final_msg = cl.Message(content="", author="Assistant")
    has_streamed_content = False
    all_final_responses = []
    table_elements = None
    
    max_network_retries = 3
    network_retry_count = 0
    
    while network_retry_count < max_network_retries:
        try:
            # 🔧 전처리된 메시지로 agent 실행 (전체 메시지 배열 전달)
            async for event in astream_events_with_retry(
                agent, processed_input["messages"], config
            ):
                kind = event["event"]

                if kind == "on_llm_start":
                    if not final_msg.id:
                        await final_msg.send()

                elif kind == "on_llm_stream" or kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    
                    chunk_content = ""
                    if isinstance(chunk.content, str):
                        chunk_content = chunk.content
                    elif isinstance(chunk.content, list):
                        for part in chunk.content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                chunk_content += part.get("text", "")
                    
                    if chunk_content and chunk_content.strip():
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
                        
                        if final_answer.strip():
                            all_final_responses.append(final_answer)
            
            # 성공하면 루프 종료
            break
            
        except (HttpxReadError, HttpcoreReadError, ConnectionError) as network_error:
            network_retry_count += 1
            print(f"[DEBUG] 네트워크 에러 발생 ({network_retry_count}/{max_network_retries}): {network_error}")
            
            if network_retry_count < max_network_retries:
                # 재시도 전 대기
                wait_time = 2 ** network_retry_count  # 지수 백오프
                await asyncio.sleep(wait_time)
                
                # 사용자에게 재시도 알림
                await cl.Message(
                    content=f"🔄 네트워크 연결이 불안정합니다. 재시도 중... ({network_retry_count}/{max_network_retries})"
                ).send()
                continue
            else:
                # 최대 재시도 횟수 초과 - 네트워크 에러 안내
                await send_error_with_reset_guidance(
                    "네트워크 연결이 불안정하여 응답을 완료할 수 없습니다.", 
                    "네트워크"
                )
                return
                
        except anthropic.APIStatusError as e:
            if getattr(e, 'error', {}).get('type') == 'overloaded_error':
                await send_error_with_reset_guidance(
                    "Anthropic 서버가 과부하 상태입니다.", 
                    "API"
                )
                return
            else:
                await send_error_with_reset_guidance(
                    f"API 오류가 발생했습니다: {str(e)}", 
                    "API"
                )
                return
                
        except MemoryError as memory_error:
            print(f"[DEBUG] 메모리 에러: {memory_error}")
            await send_error_with_reset_guidance(
                "메모리 부족으로 처리할 수 없습니다.", 
                "메모리"
            )
            return
            
        except RecursionError as recursion_error:
            print(f"[DEBUG] 재귀 에러: {recursion_error}")
            await send_error_with_reset_guidance(
                "처리 과정이 너무 복잡해졌습니다.", 
                "메모리"
            )
            return
            
        except Exception as e:
            print(f"[DEBUG] 예상치 못한 오류: {e}")
            traceback.print_exc()
            
            # 🚨 알 수 없는 오류에 대한 상세 안내
            await send_critical_error_guidance()
            return

    # 테이블 요소 처리
    if table_elements:
        await cl.Message(
            content="📊 **데이터 결과**",
            elements=table_elements,
        ).send()
        table_elements = None

    # 🔧 3. 응답 완료 후 메모리에 대화 저장 (가장 중요!)
    final_response = ""
    if not has_streamed_content and all_final_responses:
        final_response = "\n\n".join(all_final_responses)
        await cl.Message(content=final_response, author="Assistant").send()
    elif not has_streamed_content and not all_final_responses:
        await send_error_with_reset_guidance(
            "응답을 생성하는 데 문제가 발생했습니다.", 
            "일반"
        )
        return
    else:
        final_response = final_msg.content
        
        if not final_response.strip() and all_final_responses:
            additional_content = "\n\n".join(all_final_responses)
            await final_msg.stream_token(additional_content)
            final_response = final_msg.content

    # 🔧 4. 메모리에 현재 대화 저장 (핵심!)
    if final_response and final_response.strip():
        clean_input = message.content.strip()
        clean_output = final_response.strip()
        
        try:
            save_message(session_id, "assistant", clean_output)
            
            # 🔧 메모리에 현재 대화 즉시 저장
            if session_id in session_memory_store:
                actual_memory = session_memory_store[session_id]
                
                try:
                    if clean_input and clean_output:
                        # 🔧 현재 대화를 메모리에 저장
                        actual_memory.save_context(
                            {"input": clean_input},
                            {"output": clean_output}
                        )
                        print(f"[DEBUG] 현재 대화 메모리 저장 완료: Q={clean_input[:30]}, A={clean_output[:30]}")
                        
                        # 🔧 메모리 상태 확인
                        updated_vars = actual_memory.load_memory_variables({})
                        updated_history = updated_vars.get("chat_history", [])
                        print(f"[DEBUG] 업데이트된 메모리 크기: {len(updated_history)}개 메시지")
                        
                except Exception as save_error:
                    print(f"[DEBUG] 메모리 저장 실패: {save_error}")
                    # 🔧 대안 방법으로 직접 메모리에 추가
                    try:
                        actual_memory.chat_memory.add_user_message(clean_input)
                        actual_memory.chat_memory.add_ai_message(clean_output)
                        print("[DEBUG] chat_memory 직접 저장 완료")
                    except Exception as chat_error:
                        print(f"[DEBUG] chat_memory 저장도 실패: {chat_error}")
            else:
                print(f"[DEBUG] 세션 메모리를 찾을 수 없음: {session_id}")
                # 🔧 메모리가 없으면 새로 생성
                new_memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history"
                )
                session_memory_store[session_id] = new_memory
                new_memory.save_context(
                    {"input": clean_input},
                    {"output": clean_output}
                )
                print(f"[DEBUG] 새 메모리 생성 및 저장 완료")

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