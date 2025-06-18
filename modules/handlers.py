from langchain_core.callbacks import BaseCallbackHandler
import chainlit as cl
import time
from typing import Optional, Dict

class KoreanLangGraphCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.main_step: Optional[cl.Step] = None
        self.start_time = None
        self.chain_count = 0
        self.step_timings: Dict[str, float] = {}
        self.current_steps: Dict[str, cl.Step] = {}
        
    async def on_chain_start(self, serialized, inputs, **kwargs):
        self.chain_count += 1
        run_id = str(kwargs.get("run_id", ""))
        if run_id:
            self.step_timings[run_id] = time.time()
        if self.chain_count == 1:
            self.main_step = cl.Step(name="소요시간로그", type="run")
            await self.main_step.send()
            self.start_time = time.time()
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
        self.chain_count -= 1
        run_id = str(kwargs.get("run_id", ""))
        if run_id in self.step_timings and run_id in self.current_steps:
            elapsed = time.time() - self.step_timings[run_id]
            step = self.current_steps[run_id]
            step.output = f"완료 (소요시간: {elapsed:.2f}초)"
            await step.update()
            del self.current_steps[run_id]
        if self.chain_count == 0 and self.main_step and self.start_time:
            elapsed = time.time() - self.start_time
            self.main_step.name = "소요시간"
            self.main_step.output = f"총 소요시간: {elapsed:.2f}초"
            await self.main_step.update()
    async def on_llm_start(self, serialized, prompts, **kwargs):
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
        run_id = str(kwargs.get("run_id", ""))
        if run_id in self.step_timings and run_id in self.current_steps:
            elapsed = time.time() - self.step_timings[run_id]
            step = self.current_steps[run_id]
            step.output = f"완료 (소요시간: {elapsed:.2f}초)"
            await step.update()
            del self.current_steps[run_id]
    async def on_tool_start(self, serialized, input_str, **kwargs):
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
            await step.send()
            self.current_steps[run_id] = step
    async def on_tool_end(self, output, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        if run_id in self.step_timings and run_id in self.current_steps:
            elapsed = time.time() - self.step_timings[run_id]
            step = self.current_steps[run_id]
            step.output = f"완료 (소요시간: {elapsed:.2f}초)"
            await step.update()
            del self.current_steps[run_id] 