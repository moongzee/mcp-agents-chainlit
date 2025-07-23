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
from typing import Any, List, Optional

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
        
        # Agent core 생성 -> Plan-and-Execute Graph 생성으로 변경
        prompt_type = cl.user_session.get("prompt_type", "general") # 프롬프트 타입 가져오기
        graph = llm_setup.create_plan_and_execute_graph(model_name, all_langchain_tools, prompt_type)
        cl.user_session.set("agent", graph)

        print("[DEBUG] MCP 연결 완료 - Plan-and-Execute Graph 설정됨")

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
        cl.ChatProfile(name="Claude 4 + General", markdown_description="Claude Sonnet 4 + 일반 프롬프트"),
        cl.ChatProfile(name="Claude 4 + GoT deep", markdown_description="Claude Sonnet 4 + GoT 방법론 프롬프트 (Deep)"),
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
        "Claude 4 + General": ("claude-sonnet-4-20250514", "general"),
        "Claude 4 + GoT deep": ("claude-sonnet-4-20250514", "got_deep"),
        "Claude 3.7 + GoT deep": ("claude-3-7-sonnet-latest", "got_deep"),
        "Claude 3.7 + General": ("claude-3-7-sonnet-latest", "general"),
        "Gemini 2.5 Flash + GoT deep": ("gemini-2.5-flash", "got_deep"),
        "Gemini 2.5 Flash + General": ("gemini-2.5-flash", "general"),
        "Gemini 2.5 Pro + GoT deep": ("gemini-2.5-pro", "got_deep"),
        "Gemini 2.5 Pro + General": ("gemini-2.5-pro", "general"),
    }
    chat_profile = cl.user_session.get("chat_profile", "Claude 4 + General")
    model_name, prompt_type = profile_map.get(chat_profile, profile_map["Claude 4 + General"])

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
async def astream_events_with_retry(agent, graph_input, config):
    """강화된 재시도 로직으로 스트림 이벤트 처리 (입력 형식 일반화)"""
    async for event in agent.astream_events(graph_input, config=config, version="v2"):
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

    # 1. Plan-and-Execute 그래프 입력 준비 (대화 기록 추가)
    memory = memory_manager.session_memory_store.get(session_id)
    chat_history = []
    if memory:
        # ConversationBufferMemory에서 BaseMessage 객체 리스트를 직접 가져옵니다.
        chat_history = memory.load_memory_variables({}).get("chat_history", [])

    # chat_history를 PlanExecute 상태의 일부로 전달합니다.
    graph_input = {
        "input": message.content,
        "chat_history": chat_history,
        "summary": cl.user_session.get("summary", "") # 세션에서 요약본 로드
    }
    
    # 2. 그래프 실행 및 UI 렌더링
    config = RunnableConfig(recursion_limit=100, thread_id=session_id, callbacks=[cl.LangchainCallbackHandler()])
    
    final_response = "" # 전체 응답을 저장할 변수 (스트리밍을 위해 ""로 초기화)
    plan_msg = None
    agent_step = None # 현재 실행 중인 에이전트 스텝을 추적
    response_msg = cl.Message(content="", author="Assistant") # 스트리밍을 위한 빈 메시지 객체
    
    try:
        async for event in astream_events_with_retry(agent, graph_input, config):
                kind = event["event"]
                name = event["name"]

                if kind == "on_chain_start":
                    if name == "planner":
                        pass
                        #await cl.Message(content="🤔 **계획 수립 중...**", author="System").send()
                    elif name == "agent":
                        plan = event["data"].get("input", {}).get("plan", [])
                        if plan:
                            task = plan[0]
                            agent_step = cl.Step(name=f"🚀 실행: {task}", type="run")
                            await agent_step.send()
                    elif name == "replan":
                        if agent_step:
                            await agent_step.remove() # 이전 에이전트 스텝 UI 정리
                            agent_step = None
                        #await cl.Message(content="🔄 **계획 검토 및 다음 단계 준비 중...**", author="System").send()

                elif kind == "on_chain_stream":
                    # 최종 응답 스트리밍 처리
                    chunk = event["data"].get("chunk")
                    response_chunk = None
                    if isinstance(chunk, str):
                        response_chunk = chunk
                    elif isinstance(chunk, dict) and "response" in chunk:
                        # LangChain의 Plan-and-Execute 그래프에서 최종 응답이 딕셔너리 형태로 올 수 있음
                        response_chunk = chunk["response"]

                    if response_chunk:
                        # 에이전트의 청크가 누적된 전체 응답일 수 있으므로,
                        # 이미 스트리밍된 부분(final_response)과 비교하여 새로운 부분(delta)만 스트리밍합니다.
                        if response_chunk.startswith(final_response):
                            delta = response_chunk[len(final_response):]
                        else:
                            delta = response_chunk # 예상치 못한 경우, 그냥 청크 전체를 보냅니다.
                        
                        if delta:
                            final_response += delta
                            await response_msg.stream_token(delta)

                elif kind == "on_chain_end":
                    if name == "planner":
                        plan = event["data"].get("output", {}).get("plan", [])
                        if plan: # 빈 계획은 표시하지 않음
                            plan_text = "\n".join(f"- {step}" for step in plan)
                            plan_msg = cl.Message(content=f"**📝 계획:**\n{plan_text}", author="System")
                            await plan_msg.send()
                    elif name == "agent":
                         if agent_step:
                            result = event["data"].get("output", {}).get("past_steps", [("", "")])[-1][1]
                            agent_step.output = result
                            await agent_step.update()
                    elif name == "replan" or name == "__end__" or name == "agent":
                        output = event["data"].get("output", {})
                        
                        # 스트리밍이 이미 진행되었다면, 여기서는 특별한 처리를 하지 않습니다.
                        if final_response:
                            continue
                        
                        # 'replan', '__end__', 'agent' 노드 모두에서 최종 응답이 'response' 키로 전달될 수 있습니다.
                        if "response" in output:
                            final_response = output["response"]

                        # 'replan'의 경우, 다음 계획을 업데이트합니다.
                        elif "plan" in output and plan_msg:
                            new_plan = output["plan"]
                            if new_plan:
                                plan_text = "\n".join(f"- {step}" for step in new_plan)
                                plan_msg.content = f"📝 **다음 계획:**\n{plan_text}"
                                await plan_msg.update()
                            else:
                                # 새로운 계획이 비어있으면, 모든 계획이 완료되었음을 의미하므로 UI를 정리합니다.
                                await plan_msg.remove()
        
        # 스트리밍이 완료되었거나, 스트리밍 없이 최종 응답만 있는 경우 처리
        if response_msg.content: 
            await response_msg.update()
        elif final_response: 
             await cl.Message(content=final_response, author="Assistant").send()


    except (HttpxReadError, HttpcoreReadError, ConnectionError) as e:
        await ui_utils.send_error_with_reset_guidance("네트워크가 불안정하여 응답할 수 없습니다.", "네트워크"); return
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
    if agent_step: await agent_step.remove()
    # 3. 최종 응답 및 메모리 저장
    if final_response:
        # await cl.Message(content=final_response, author="Assistant").send() # 스트리밍으로 대체되었으므로 주석 처리
        try:
            db_utils.save_message(session_id, "assistant", final_response)
            if session_id in memory_manager.session_memory_store:
                memory = memory_manager.session_memory_store[session_id]
                memory.save_context({"input": message.content.strip()}, {"output": final_response})
                print(f"[DEBUG] 대화 저장 완료. 메모리 크기: {len(memory.load_memory_variables({})['chat_history'])}개")
        except Exception as e:
            print(f"[DEBUG] 최종 응답 저장 실패: {e}")
    elif not response_msg.content: # 스트리밍도, 최종응답도 없는 경우
        await ui_utils.send_error_with_reset_guidance("응답 생성에 실패했습니다.", "일반"); return
    
    # 요약 업데이트 로직 추가
    if final_response:
        # 1. 현재 세션의 전체 대화 기록 가져오기
        memory = memory_manager.session_memory_store.get(session_id)
        chat_history_for_summary = []
        if memory:
            chat_history_for_summary = memory.load_memory_variables({}).get("chat_history", [])

        # 2. 요약할 메시지가 충분히 있을 경우에만 요약 실행
        if len(chat_history_for_summary) > 2:
            try:
                new_summary = await llm_setup.create_intelligent_summary_silent(
                    messages_to_summarize=chat_history_for_summary,
                    existing_summary=cl.user_session.get("summary", "")
                )
                cl.user_session.set("summary", new_summary)
                print(f"[DEBUG] 대화 요약 업데이트 완료.")
            except Exception as e:
                print(f"[DEBUG] 대화 요약 업데이트 중 오류 발생: {e}")
