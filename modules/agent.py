from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from modules.utils import trimming_node

class AsyncConversationSummaryBufferMemory:
    def __init__(self, memory):
        self.memory = memory

    async def aget_messages(self):
        variables = self.memory.load_memory_variables({})
        summary = variables["chat_history"]
        if isinstance(summary, str):
            return [AIMessage(content=summary)]
        return summary

    def __getattr__(self, name):
        return getattr(self.memory, name)

# 에이전트, 메모리 생성 등은 기존 app_chainlit.py에서 이 모듈로 옮겨서 사용하도록 리팩토링 예정 