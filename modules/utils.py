from typing import Any, Dict, List, Callable, Optional
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
import uuid
#import tiktoken 
import pprint
import re
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

# trimming_node: trim_messages를 활용해 최근 N개 메시지만 LLM에 전달
async def trimming_node(state, max_tokens=5000):
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed_messages}



def build_llm_input(memory, user_input, last_n=1):
    variables = memory.load_memory_variables({})
    summary = variables.get("chat_history", "")
    buffer = getattr(memory, "buffer", [])
    recent = buffer[-last_n:] if last_n > 0 else []
    llm_input = []
    if summary:
        llm_input.append(HumanMessage(content=f"[이전 대화 요약]\n{summary}"))
    llm_input.extend(recent)
    llm_input.append(HumanMessage(content=user_input))
    #print(f"DEBUG: 빌드된 llm_input: {llm_input}")
    return llm_input


async def summarization_node(state, max_summary_tokens=2000, buffer_trigger_tokens=6000, last_n=1):
    memory = state["memory"]
    user_input = state["user_input"]
    model = state["model"]

    # 1. 현 시점 LLM input (요약+buffer+입력) 구성
    llm_input = build_llm_input(memory, user_input, last_n=last_n)
    total_tokens = sum(count_tokens_approximately(m.content) for m in llm_input)

    # 2. input 전체가 buffer_trigger_tokens(6000) 초과시만 buffer를 요약(즉, save_context 트리거)
    if total_tokens > buffer_trigger_tokens:
        # 'dummy_output'은 AI 응답 dummy (buffer 요약 트리거용)
        memory.save_context({"input": user_input}, {"output": "[BUFFER 요약 트리거용]"})
        # buffer가 summary로 변환됨
        print(f"[DEBUG] buffer summary triggered (input {total_tokens} tokens)")

    # 3. 요약본이 2000 초과면 meta-summary
    variables = memory.load_memory_variables({})
    summary = variables.get("chat_history", "")
    if summary and count_tokens_approximately(summary) > max_summary_tokens:
        prompt = f"""
아래 요약문을 2000토큰(한글 기준 약 1500~2000자) 이내로 아주 간결하게 다시 요약하세요.

[요약문]
{summary}
"""
        short_summary = model.invoke([HumanMessage(content=prompt)]).content
        memory.moving_summary_buffer = short_summary
        print(f"[DEBUG] summary meta-summary triggered ({count_tokens_approximately(summary)} -> {count_tokens_approximately(short_summary)}) tokens")

    # 4. 최종 input(요약+최신2+현재입력)으로 재구성
    llm_input = build_llm_input(memory, user_input, last_n=last_n)
    return {"messages": llm_input}

