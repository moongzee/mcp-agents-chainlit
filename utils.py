from typing import Any, Dict, List, Callable, Optional
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
import uuid
#import tiktoken 
import pprint
import re
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


# trimming_node: trim_messages를 활용해 최근 N개 메시지만 LLM에 전달
async def trimming_node(state, max_tokens=15000):
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed_messages}




