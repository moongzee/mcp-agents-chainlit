# app_chainlit.py (리팩토링 완료 버전)

# --- 1. 기본 라이브러리 임포트 ---
import chainlit as cl
import chainlit.types as cl_types
import chainlit.server as cl_server
import json
import pandas as pd
import io
from datetime import datetime
import traceback
import time
import asyncio
import re
import codecs
from typing import Any, List

# --- 2. MCP 및 Langchain 관련 타입 임포트 ---
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableConfig
from google.api_core.exceptions import ServiceUnavailable
import anthropic
# 네트워크 관련
import httpx
from httpx import ReadError as HttpxReadError
import httpcore
from httpcore import ReadError as HttpcoreReadError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# --- 3. 분리된 모듈 임포트 ---
from dotenv import load_dotenv
import utils.db_utils as db_utils
import utils.ui_utils as ui_utils
import utils.llm_setup as llm_setup
import utils.memory_manager as memory_manager

# --- 4. 초기 설정 ---
load_dotenv(override=True)
db_utils.init_database()


# --- 5. 로그인 및 인증 ---
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """사용자 인증 콜백"""
    print(f"DEBUG: 로그인 시도 - 사용자명: {username}")
    user_data = db_utils.authenticate_user(username, password)
    print(f"DEBUG: 인증 결과: {user_data}")

    if user_data:
        print(f"DEBUG: 로그인 성공 - {user_data['display_name']}")
        user = cl.User(
            identifier=user_data["username"],
            metadata={
                "user_id": user_data["id"],
                "username": user_data["username"],
                "display_name": user_data["display_name"],
                "db_id": user_data["id"]
            }
        )
        try:
            user.display_name = user_data["display_name"]
            print(f"DEBUG: display_name 직접 설정 완료: {user_data['display_name']}")
        except Exception as e:
            print(f"DEBUG: display_name 설정 실패: {e}")
        return user
    else:
        print(f"DEBUG: 로그인 실패 - 사용자명 또는 비밀번호 불일치")
        return None

# --- 6. MCP 연결 핸들러 ---
@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """MCP 연결 시 도구 및 에이전트 설정"""
    try:
        tool_metadatas = await session.list_tools()
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = [
            {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
            for t in tool_metadatas.tools
        ]
        cl.user_session.set("mcp_tools", mcp_tools)

        model_name = cl.user_session.get("model_name", "claude-sonnet-4-20250514")

        from langchain_core.tools import Tool
        all_langchain_tools = []
        for conn_name, tools in mcp_tools.items():
            for tool_info in tools:
                async def tool_func(tool_input: Any, conn_name=conn_name, tool_name=tool_info['name']):
                    mcp_session, _ = cl.context.session.mcp_sessions.get(conn_name)
                    if not mcp_session: return f"Error: MCP session for '{conn_name}' not found."
                    if not isinstance(tool_input, dict): tool_input = {"query": tool_input}
                    try:
                        return str(await mcp_session.call_tool(tool_name, tool_input))
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
        session_id = cl.context.session.id
        if session_id not in memory_manager.session_memory_store:
            memory_manager.session_memory_store[session_id] = ConversationBufferMemory(
                return_messages=True, memory_key="chat_history"
            )
        
        # Agent core 생성
        agent_core = llm_setup.create_agent(model_name, all_langchain_tools)
        cl.user_session.set("agent", agent_core)

        print("[DEBUG] MCP 연결 완료 - 순수 agent 설정됨")

    except Exception as e:
        error_msg = f"MCP 연결 처리 중 오류가 발생했습니다: {e}"
        print(error_msg)
        traceback.print_exc()
        await ui_utils.send_error_with_reset_guidance("도구 연결에 실패했습니다.", "일반")

@cl.on_mcp_disconnect
async def on_mcp_disconnect(connection):
    await cl.Message(content=f"MCP 연결 '{connection}'이(가) 종료되었습니다.").send()

# --- 7. Chat Profile 및 시작 설정 ---
def load_mcp_servers_from_config():
    """config.json에서 MCP 서버 설정 읽기"""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return [{"name": name, "command": f"{conf['command']} {' '.join(conf.get('args', []))}".strip()}
                for name, conf in config.items()]
    except Exception as e:
        print(f"[DEBUG] config.json 읽기 실패: {e}")
        return []

@cl.set_chat_profiles
async def chat_profiles():
    """다양한 모델과 프롬프트 조합의 챗 프로필 설정"""
    return [
        cl.ChatProfile(name="Claude 4 + GoT deep", markdown_description="Claude Sonnet 4 + GoT 방법론 프롬프트 (Deep)"),
        cl.ChatProfile(name="Claude 4 + General", markdown_description="Claude Sonnet 4 + 일반 프롬프트"),
        cl.ChatProfile(name="Claude 3.7 + GoT deep", markdown_description="Claude 3.7 Sonnet + GoT 방법론 프롬프트 (Deep)"),
        cl.ChatProfile(name="Claude 3.7 + General", markdown_description="Claude 3.7 Sonnet + 일반 프롬프트"),
        cl.ChatProfile(name="Gemini 2.5 Pro + GoT deep", markdown_description="Gemini 2.5 Pro + GoT 방법론 프롬프트 (Deep)"),
        cl.ChatProfile(name="Gemini 2.5 Pro + General", markdown_description="Gemini 2.5 Pro + 일반 프롬프트"),
        cl.ChatProfile(name="Gemini 2.5 Flash + GoT deep", markdown_description="Gemini 2.5 Flash + GoT 방법론 프롬프트 (Deep)"),
        cl.ChatProfile(name="Gemini 2.5 Flash + General", markdown_description="Gemini 2.5 Flash + 일반 프롬프트"),
    ]

@cl.on_chat_start
async def on_chat_start():
    """채팅 시작 시 모델, 프롬프트, MCP 연결 설정"""
    profile_map = {
        "Claude 4 + GoT deep": ("claude-sonnet-4-20250514", "got_deep"),
        "Claude 4 + General": ("claude-sonnet-4-20250514", "general"),
        "Claude 3.7 + GoT deep": ("claude-3-7-sonnet-latest", "got_deep"),
        "Claude 3.7 + General": ("claude-3-7-sonnet-latest", "general"),
        "Gemini 2.5 Flash + GoT deep": ("gemini-2.5-flash", "got_deep"),
        "Gemini 2.5 Flash + General": ("gemini-2.5-flash", "general"),
        "Gemini 2.5 Pro + GoT deep": ("gemini-2.5-pro", "got_deep"),
        "Gemini 2.5 Pro + General": ("gemini-2.5-pro", "general"),
    }
    chat_profile = cl.user_session.get("chat_profile", "Claude 4 + GoT deep")
    model_name, prompt_type = profile_map.get(chat_profile, profile_map["Claude 4 + GoT deep"])

    cl.user_session.set("model_name", model_name)
    cl.user_session.set("prompt_type", prompt_type)

    mcp_servers = load_mcp_servers_from_config()
    try:
        print("[DEBUG] 직접 MCP 연결 시도...")
        for i, server in enumerate(mcp_servers, 1):
            try:
                conn_request = cl_types.ConnectStdioMCPRequest(
                    sessionId=cl.context.session.id,
                    clientType="stdio", name=server["name"], fullCommand=server["command"]
                )
                await cl_server.connect_mcp(conn_request, cl.context.session.user)
                print(f"[DEBUG] MCP 서버 {i} ({server['name']}) 연결 성공")
                if i < len(mcp_servers): await asyncio.sleep(0.5)
            except Exception as e:
                print(f"[DEBUG] MCP 서버 {i} ({server['name']}) 연결 실패: {e}")
    except Exception as e:
        print(f"[DEBUG] MCP 연결 전체 실패: {e}")
        try:
            await cl.mcp.autoconnect(); print("[DEBUG] Fallback autoconnect 성공")
        except Exception as fallback_error:
            print(f"[DEBUG] Fallback autoconnect도 실패: {fallback_error}")

    user = cl.user_session.get("user") or cl.context.session.user
    if not user:
        await cl.Message(content="로그인이 필요합니다. 다시 로그인해주세요.").send()
        return
    
    cl.user_session.set("mcp_tools", {})
    cl.user_session.set("message_count", 0)
    display_name = user.metadata.get("display_name", "사용자")
    await cl.Message(content=f"안녕하세요 {display_name}님! NOA 입니다. 무엇이든 물어보세요 😊").send()

@cl.on_chat_end
async def on_chat_end():
    """채팅 종료 시 세션 저장"""
    user = cl.user_session.get("user") or cl.context.session.user
    if not user: return

    if cl.user_session.get("message_count", 0) > 0:
        first_message = cl.user_session.get("first_message", "새로운 대화")
        title = db_utils.generate_session_title(first_message)
        actual_user_id = user.metadata.get("user_id", user.identifier)
        db_utils.save_chat_session(cl.context.session.id, actual_user_id, title)

# --- 8. 메시지 처리 핸들러 ---
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5),
    retry=retry_if_exception_type((HttpxReadError, HttpcoreReadError, ConnectionError, anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError, TimeoutError))
)
async def astream_events_with_retry(agent, messages, config):
    """강화된 재시도 로직으로 스트림 이벤트 처리"""
    async for event in agent.astream_events({"messages": messages}, config=config, version="v2"):
        yield event

@cl.on_message
async def on_message(message: cl.Message):
    """사용자 메시지 처리"""
    user = cl.user_session.get("user") or cl.context.session.user
    if not user:
        await ui_utils.send_error_with_reset_guidance("로그인이 필요합니다.", "일반")
        return

    # 메시지 카운트 및 DB 저장
    msg_count = cl.user_session.get("message_count", 0) + 1
    cl.user_session.set("message_count", msg_count)
    if msg_count == 1: cl.user_session.set("first_message", message.content)
    
    session_id = cl.context.session.id
    try:
        db_utils.save_message(session_id, "user", message.content)
    except Exception as e:
        print(f"[DEBUG] DB 저장 실패: {e}")

    agent = cl.user_session.get("agent")
    if not agent:
        await ui_utils.send_error_with_reset_guidance("에이전트가 설정되지 않았습니다.", "일반")
        return

    # 1. 요약 및 메시지 전처리
    input_data = {"messages": [HumanMessage(content=message.content)]}
    processed_input = await memory_manager.preprocess_with_silent_summary(input_data, llm_setup.load_system_prompt)

    # 2. 에이전트 실행
    config = RunnableConfig(recursion_limit=100, thread_id=session_id, callbacks=[cl.LangchainCallbackHandler()])
    final_msg = cl.Message(content="", author="Assistant")
    has_streamed, all_responses, table_elements = False, [], None
    max_retries, retry_count = 3, 0

    while retry_count < max_retries:
        try:
            async for event in astream_events_with_retry(agent, processed_input["messages"], config):
                kind = event["event"]
                if kind == "on_llm_start" and not final_msg.id: await final_msg.send()
                elif kind in ("on_llm_stream", "on_chat_model_stream"):
                    chunk_content = event["data"]["chunk"].content
                    if isinstance(chunk_content, list): # Gemini 1.5 Pro
                        chunk_content = "".join(part.get("text", "") for part in chunk_content if part.get("type") == "text")
                    if chunk_content: await final_msg.stream_token(chunk_content); has_streamed = True
                elif kind == "on_tool_end":
                    step = cl.user_session.get(f"tool_step_{event['run_id']}")
                    output_obj = event["data"].get("output")
                    raw_content = output_obj.content if isinstance(output_obj, ToolMessage) else None
                    json_str = None
                    if isinstance(raw_content, CallToolResult):
                        json_str = next((c.text for c in raw_content.content if isinstance(c, TextContent)), None)
                    elif isinstance(raw_content, str):
                        match = re.search(r"text='(.*?)'", raw_content, re.DOTALL)
                        if match: json_str = match.group(1)
                    
                    if step: step.output = json_str or str(raw_content)

                    if event["name"] == "run_cortex_agents" and json_str:
                        try:
                            unescaped = codecs.decode(json_str, "unicode_escape").encode("latin1").decode("utf-8")
                            data = json.loads(unescaped).get("results", {})
                            if data.get("data"):
                                cols = [c["name"] for c in data["resultSetMetaData"]["rowType"]]
                                df = pd.DataFrame(data["data"], columns=cols)
                                csv_bytes = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                                table_elements = [
                                    cl.Dataframe(data=df, name="상세 데이터", display="inline"),
                                    cl.File(name=f"report_{datetime.now():%Y%m%d_%H%M%S}.csv", content=csv_bytes, display="inline")
                                ]
                        except Exception as e: print(f"툴 결과 처리 오류: {e}\n{traceback.format_exc()}")
                    if step: await step.update()
                elif kind in ("on_llm_end", "on_chat_model_end"):
                    output = event["data"].get("output")
                    if output:
                        content = output.content if hasattr(output, 'content') else ""
                        if isinstance(content, list): content = "".join(p.get("text", "") for p in content if p.get("type")=="text")
                        if content: all_responses.append(content)
            break
        except (HttpxReadError, HttpcoreReadError, ConnectionError) as e:
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(2 ** retry_count)
                await cl.Message(content=f"🔄 네트워크 연결이 불안정하여 재시도합니다... ({retry_count}/{max_retries})").send()
            else: await ui_utils.send_error_with_reset_guidance("네트워크가 불안정하여 응답할 수 없습니다.", "네트워크"); return
        except (anthropic.APIStatusError, ServiceUnavailable) as e:
            msg, err_type = ("서버 과부하 상태입니다.", "API")
            if isinstance(e, ServiceUnavailable) and not ("overloaded" in str(e) or "503" in str(e)):
                msg, err_type = (f"API 오류: {e}", "API")
            await ui_utils.send_error_with_reset_guidance(msg, err_type); return
        except (MemoryError, RecursionError) as e:
            err_type = "메모리" if isinstance(e, MemoryError) else "복잡도"
            await ui_utils.send_error_with_reset_guidance(f"{err_type} 문제로 처리할 수 없습니다.", "메모리"); return
        except Exception as e:
            print(f"[DEBUG] 예상치 못한 오류: {e}\n{traceback.format_exc()}")
            await ui_utils.send_critical_error_guidance(); return

    if table_elements:
        await cl.Message(content="📊 **데이터 결과**", elements=table_elements).send()

    # 3. 최종 응답 및 메모리 저장
    final_response = final_msg.content.strip()
    if not has_streamed and all_responses: final_response = "\n\n".join(all_responses)
    if not final_response:
        await ui_utils.send_error_with_reset_guidance("응답 생성에 실패했습니다.", "일반"); return
    
    await final_msg.update()

    try:
        db_utils.save_message(session_id, "assistant", final_response)
        if session_id in memory_manager.session_memory_store:
            memory = memory_manager.session_memory_store[session_id]
            memory.save_context({"input": message.content.strip()}, {"output": final_response})
            print(f"[DEBUG] 대화 저장 완료. 메모리 크기: {len(memory.load_memory_variables({})['chat_history'])}개")
    except Exception as e:
        print(f"[DEBUG] 최종 응답 저장 실패: {e}")
