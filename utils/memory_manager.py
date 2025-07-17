import chainlit as cl
import traceback
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import anthropic
import httpx
from httpx import ReadError as HttpxReadError
import httpcore
from httpcore import ReadError as HttpcoreReadError
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage

# 🔧 전역 세션 메모리 스토어
session_memory_store = {}
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

async def preprocess_with_silent_summary(input_data, load_system_prompt_func):
    """누적 요약 기능이 포함된 전처리기"""
    global session_memory_store
    session_id = cl.context.session.id

    if session_id not in session_memory_store:
        session_memory_store[session_id] = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history"
        )
    memory = session_memory_store[session_id]

    user_message = input_data.get("messages", [])

    prompt_type = cl.user_session.get("prompt_type", "general")
    system_prompt = load_system_prompt_func(prompt_type)
    system_prompt_message = [SystemMessage(content=system_prompt)]

    try:
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])

        existing_summary = None
        if chat_history and isinstance(chat_history[0], SystemMessage):
            if isinstance(chat_history[0].content, str) and chat_history[0].content.startswith("--- 누적 요약 ---"):
                existing_summary = chat_history[0].content
                chat_history = chat_history[1:]
                print(f"[DEBUG] 기존 요약 발견. 실제 대화 길이: {len(chat_history)}")

        if len(chat_history) >= 4:
            messages_to_summarize = chat_history[:-2]
            recent_messages = chat_history[-2:]

            print(f"[DEBUG] 누적 요약 시작. 기존 요약:{'있음' if existing_summary else '없음'}, 요약 대상:{len(messages_to_summarize)}, 유지:{len(recent_messages)}")

            updated_summary_content = await create_intelligent_summary_silent(messages_to_summarize, existing_summary)

            summary_message = SystemMessage(content=f"--- 누적 요약 ---\n{updated_summary_content}")
            memory.chat_memory.messages = [summary_message] + recent_messages

            print(f"[DEBUG] 메모리 재구성 완료. 현재 메모리: 요약 1개 + 최신 대화 {len(recent_messages)}개")

            messages_for_llm = [summary_message] + recent_messages
        else:
            messages_for_llm = ([SystemMessage(content=existing_summary)] if existing_summary else []) + chat_history

        final_messages_to_agent = system_prompt_message + messages_for_llm + user_message

        print(f"[DEBUG] 최종 전달 메시지 수: {len(final_messages_to_agent)}")
        return {"messages": final_messages_to_agent}

    except Exception as e:
        print(f"[DEBUG] 전처리 중 심각한 오류 발생: {e}")
        traceback.print_exc()
        return {"messages": system_prompt_message + user_message}

class AsyncConversationBufferMemory:
    """ConversationBufferMemory를 RunnableWithMessageHistory에서 사용할 수 있도록 하는 래퍼"""
    def __init__(self, memory):
        self.memory = memory

    async def aget_messages(self):
        variables = self.memory.load_memory_variables({})
        chat_history = variables.get("chat_history", [])
        return chat_history if isinstance(chat_history, list) else []

    async def aadd_messages(self, messages, *args, **kwargs):
        print(f"[DEBUG][aadd_messages] called with {len(messages)} messages")
        for msg in messages:
            if hasattr(msg, 'type'):
                if msg.type == "human":
                    self.memory.save_context({"input": msg.content}, {"output": ""})
                elif msg.type == "ai":
                    self.memory.save_context({"input": ""}, {"output": msg.content})

    def __getattr__(self, name):
        return getattr(self.memory, name) 