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

# --- 2. LangChain 관련 라이브러리 임포트 ---
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from utils import trimming_node

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
    #print(f"DEBUG: 로그인 시도 - 사용자명: {username}")
    
    user_data = authenticate_user(username, password)
    #print(f"DEBUG: 인증 결과: {user_data}")
    
    if user_data:
        #print(f"DEBUG: 로그인 성공 - {user_data['display_name']}")
        
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
            #print(f"DEBUG: display_name 직접 설정 완료: {user_data['display_name']}")
        except Exception as e:
            print(f"DEBUG: display_name 설정 실패: {e}")
        
        #print(f"DEBUG: 최종 생성된 User 객체:")
        #print(f"  - identifier: {user.identifier}")
        #print(f"  - display_name: {getattr(user, 'display_name', 'None')}")
        #print(f"  - metadata: {user.metadata}")
        
        return user
    else:
        print(f"DEBUG: 로그인 실패 - 사용자명 또는 비밀번호 불일치")
        return None

# --- 5. 커스텀 콜백 핸들러 클래스 ---
class KoreanLangGraphCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.main_step: Optional[cl.Step] = None
        self.start_time = None
        self.chain_count = 0
        self.step_timings: Dict[str, float] = {}
        self.current_steps: Dict[str, cl.Step] = {}
        
    async def on_chain_start(self, serialized, inputs, **kwargs):
        """체인 시작"""
        self.chain_count += 1
        run_id = str(kwargs.get("run_id", ""))
        
        if run_id:
            self.step_timings[run_id] = time.time()
        
        # 첫 번째 체인 시작 시에만 메인 단계 생성
        if self.chain_count == 1:
            self.main_step = cl.Step(name="생각하는중", type="run")
            await self.main_step.send()
            self.start_time = time.time()
        
        # 하위 체인들도 단계로 표시
        elif self.main_step:
            chain_name = serialized.get("name", "Chain")
            step = cl.Step(
                name=f"Used {chain_name}",
                type="chain",
                parent_id=self.main_step.id
            )
            await step.send()
            self.current_steps[run_id] = step
    
    async def on_chain_end(self, outputs, **kwargs):
        """체인 종료"""
        self.chain_count -= 1
        run_id = str(kwargs.get("run_id", ""))
        
        # 하위 체인 완료 처리
        if run_id in self.step_timings and run_id in self.current_steps:
            elapsed = time.time() - self.step_timings[run_id]
            step = self.current_steps[run_id]
            step.output = f"완료 (소요시간: {elapsed:.2f}초)"
            await step.update()
            del self.current_steps[run_id]
        
        # 모든 체인이 끝났을 때 메인 단계 업데이트
        if self.chain_count == 0 and self.main_step and self.start_time:
            elapsed = time.time() - self.start_time
            self.main_step.name = "생각하는과정표시"
            self.main_step.output = f"총 소요시간: {elapsed:.2f}초"
            await self.main_step.update()
    
    async def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 시작"""
        run_id = str(kwargs.get("run_id", ""))
        if run_id:
            self.step_timings[run_id] = time.time()
            
        if self.main_step:
            step = cl.Step(
                name="Used llm",
                type="llm",
                parent_id=self.main_step.id
            )
            await step.send()
            self.current_steps[run_id] = step
    
    async def on_llm_end(self, response, **kwargs):
        """LLM 종료"""
        run_id = str(kwargs.get("run_id", ""))
        
        if run_id in self.step_timings and run_id in self.current_steps:
            elapsed = time.time() - self.step_timings[run_id]
            step = self.current_steps[run_id]
            step.output = f"완료 (소요시간: {elapsed:.2f}초)"
            await step.update()
            del self.current_steps[run_id]
    
    async def on_tool_start(self, serialized, input_str, **kwargs):
        """도구 시작"""
        run_id = str(kwargs.get("run_id", ""))
        tool_name = serialized.get("name", "tool")
        
        if run_id:
            self.step_timings[run_id] = time.time()
            
        if self.main_step:
            step = cl.Step(
                name=f"Used {tool_name}",
                type="tool",
                parent_id=self.main_step.id
            )
            step.input = str(input_str)
            await step.send()
            self.current_steps[run_id] = step
    
    async def on_tool_end(self, output, **kwargs):
        """도구 종료"""
        run_id = str(kwargs.get("run_id", ""))
        
        if run_id in self.step_timings and run_id in self.current_steps:
            elapsed = time.time() - self.step_timings[run_id]
            step = self.current_steps[run_id]
            output_preview = str(output)[:200] + "..." if len(str(output)) > 200 else str(output)
            step.output = f"완료 (소요시간: {elapsed:.2f}초)\n\n결과: {output_preview}"
            await step.update()
            del self.current_steps[run_id]

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

        agent = create_react_agent(
            model,
            all_langchain_tools,
            checkpointer=MemorySaver(),
            prompt=SYSTEM_PROMPT,
            pre_model_hook=trimming_node
        )
        cl.user_session.set("agent", agent)
        

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

# config.json에서 MCP 서버 기본값 읽기
DEFAULT_MCP_NAME = None
DEFAULT_MCP_COMMAND = None
DEFAULT_MCP_ARGS = None
DEFAULT_MCP_TRANSPORT = None

if os.path.exists("config.json"):
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        for name, info in config.items():
            DEFAULT_MCP_NAME = name
            DEFAULT_MCP_COMMAND = info.get("command")
            DEFAULT_MCP_ARGS = info.get("args", [])
            DEFAULT_MCP_TRANSPORT = info.get("transport")
            break

@cl.on_chat_start
async def on_chat_start():
    """채팅 시작 시 초기화 및 사이드바 설정"""
    user = cl.user_session.get("user")
    if not user:
        user = cl.context.session.user
        cl.user_session.set("user", user)
    
    if not user:
        await cl.Message(content="로그인이 필요합니다. 다시 로그인해주세요.").send()
        return
    
    #print(f"DEBUG: 사용자 정보 - {user}")
    #print(f"DEBUG: 사용자 메타데이터 - {user.metadata}")
    #print(f"DEBUG: user_id 확인 - {user.metadata.get('user_id')}")
    
    cl.user_session.set("mcp_tools", {})
    cl.user_session.set("message_count", 0)

    # config.json의 MCP 서버를 기본값으로 등록
    if DEFAULT_MCP_NAME and DEFAULT_MCP_COMMAND:
        mcp_servers = cl.user_session.get("mcp_servers", {})
        mcp_servers[DEFAULT_MCP_NAME] = {
            "command": DEFAULT_MCP_COMMAND,
            "args": DEFAULT_MCP_ARGS,
            "transport": DEFAULT_MCP_TRANSPORT
        }
        cl.user_session.set("mcp_servers", mcp_servers)
    
    # 사용자별 채팅 기록 사이드바 설정
    await setup_chat_history_sidebar(user.identifier)
    
    display_name = user.metadata.get("display_name") or user.metadata.get("username") or "사용자"
    try:
        user.display_name = display_name
    except Exception as e:
        print(f"DEBUG: 사용자 표시명 설정 실패: {e}")
    await cl.Message(content=f"안녕하세요 {display_name}님! NOA 입니다. 무엇이든 물어보세요 😊").send()

## 채팅기록 보여주기용 기능 안
async def setup_chat_history_sidebar(user_identifier: str):
    """사용자별 채팅 기록을 초기 메시지로 표시"""
    user = cl.user_session.get("user") or cl.context.session.user
    if not user:
        return
    
    # 여러 방법으로 user_id 찾기
    actual_user_id = None
    
    # 방법 1: metadata에서 user_id 찾기
    actual_user_id = user.metadata.get("user_id")
    if actual_user_id:
        print(f"DEBUG: metadata에서 user_id 찾음: {actual_user_id}")
    
    # 방법 2: metadata에서 db_id 찾기 (백업)
    if not actual_user_id:
        actual_user_id = user.metadata.get("db_id")
        if actual_user_id:
            print(f"DEBUG: metadata에서 db_id 찾음: {actual_user_id}")
    
    # 방법 3: 이메일로 직접 데이터베이스에서 조회
    if not actual_user_id:
        username = user.metadata.get("username") or user.identifier
        if username and "@" in username:  # 이메일인 경우
            print(f"DEBUG: 이메일로 user_id 조회 시도: {username}")
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            conn.close()
            if result:
                actual_user_id = result[0]
                print(f"DEBUG: 데이터베이스에서 user_id 찾음: {actual_user_id}")
    
    # 방법 4: identifier가 해시값인 경우 그대로 사용
    if not actual_user_id and len(user.identifier) == 32:  # MD5 해시 길이
        actual_user_id = user.identifier
        print(f"DEBUG: identifier를 user_id로 사용: {actual_user_id}")
    
    if not actual_user_id:
        print("DEBUG: 모든 방법으로 user_id를 찾을 수 없음")
        return
        
    sessions = get_chat_sessions(actual_user_id)
    print(f"DEBUG: 사용자 {user.metadata.get('display_name', 'Unknown')}의 채팅 기록 {len(sessions)}개 로드됨")
    
    # if sessions:
    #     history_content = "## 📚 채팅 기록\n\n"
    #     history_content += "*이전 대화들:*\n\n"
        
    #     for session_id, title, created_at, message_count in sessions:
    #         created_date = datetime.fromisoformat(created_at).strftime("%m/%d %H:%M")
    #         history_content += f"• **{title}** ({message_count}개 메시지) - {created_date}\n"
        
    #     history_content += f"\n총 {len(sessions)}개의 대화가 저장되어 있습니다."
    #     history_content += "\n\n*새로운 대화를 시작하거나 `/기록`을 입력하여 상세 정보를 확인하세요.*"
        
    #     # 채팅 기록을 시스템 메시지로 표시
    #     await cl.Message(
    #         content=history_content,
    #         author="📚 System"
    #     ).send()
    
    # 세션에 저장 (실제 user_id 포함)
    cl.user_session.set("chat_sessions", sessions)
    cl.user_session.set("actual_user_id", actual_user_id)

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

    # Chainlit의 기본 LangGraph 트레이스를 비활성화하고 커스텀만 사용
    korean_callback = KoreanLangGraphCallbackHandler()
    
    config = RunnableConfig(
        recursion_limit=100,
        thread_id=cl.context.session.id,
        callbacks=[korean_callback]  # 커스텀 콜백만 사용 (기본 Chainlit 콜백 제거)
    )
    
    # 최종 응답 메시지 처리
    final_msg = cl.Message(content="", author="Assistant")
    has_streamed_content = False
    all_final_responses = []
    
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=message.content)]},
        config=config,
        version="v2"
    ):
        kind = event["event"]
        print(f"DEBUG: Event - {kind}")
        
        if kind == "on_llm_start":
            if not final_msg.id:
                await final_msg.send()
                
        elif kind == "on_llm_stream" or kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(f"DEBUG: {kind} - chunk: {chunk}")
            if isinstance(chunk.content, str) and chunk.content.strip():
                await final_msg.stream_token(chunk.content)
                has_streamed_content = True
            elif isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text", "").strip():
                        await final_msg.stream_token(part.get("text", ""))
                        has_streamed_content = True
        
        elif kind == "on_llm_end" or kind == "on_chat_model_end":
            output = event["data"].get("output")
            print(f"DEBUG: {kind} - output: {output}")
            print(f"DEBUG: {kind} - has_streamed_content: {has_streamed_content}")
            
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
                    print(f"DEBUG: 수집된 최종 응답: {final_answer[:100]}...")
    
    # 스트림 완료 후 최종 응답 처리
    print(f"DEBUG: 스트림 완료 - has_streamed_content: {has_streamed_content}")
    print(f"DEBUG: 스트림 완료 - all_final_responses: {len(all_final_responses)}개")
    
    final_response = ""
    if not has_streamed_content and all_final_responses:
        # 스트리밍이 없었다면 수집된 모든 응답을 합쳐서 전송
        final_response = "\n\n".join(all_final_responses)
        print(f"DEBUG: 최종 응답 전송: {final_response[:200]}...")
        await cl.Message(content=final_response, author="Assistant").send()
    elif not has_streamed_content and not all_final_responses:
        # 아무 응답도 없었다면 기본 메시지
        final_response = "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다."
        print("DEBUG: 응답이 없어서 기본 메시지 전송")
        await cl.Message(content=final_response, author="Assistant").send()
    else:
        # 스트리밍된 내용이 있는 경우 final_msg의 내용을 사용
        final_response = final_msg.content
    
    # 어시스턴트 응답을 DB에 저장
    if final_response:
        save_message(session_id, "assistant", final_response)


