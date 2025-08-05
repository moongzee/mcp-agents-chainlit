from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from datetime import datetime, timezone, timedelta
import re
import json

# --- Plan-and-Execute 관련 추가 임포트 ---
import operator
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
# --- 요약 기능 추가 ---
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import anthropic
import asyncio
from httpx import ReadError as HttpxReadError
from httpcore import ReadError as HttpcoreReadError
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage


# --- 설정 파일 로드 ---
def load_routing_config(config_path="prompt/routing_config.json"):
    """JSON 설정 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config file: {e}")
        # 설정 파일 로드 실패 시 비상용 기본값
        return {}
ROUTING_CONFIG = load_routing_config()

def load_file_content(file_path: str) -> str:
    """주어진 경로의 파일 내용을 읽어 반환합니다."""
    if not file_path:
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}")
        return ""

def load_system_prompt(prompt_type: str) -> str:
    """프롬프트 타입에 따라 시스템 프롬프트를 로드합니다."""
    KST = timezone(timedelta(hours=9))
    current_time = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

    if prompt_type == "general":
        prompt_path = "prompt/system_prompt_general.txt"
    elif prompt_type == "got_deep":
        prompt_path = "prompt/system_prompt_got_deep.txt"
    else:
        # 기본 프롬프트 또는 오류 처리
        prompt_path = "prompt/system_prompt_general.txt"
        print(f"Warning: Unsupported prompt type '{prompt_type}'. Falling back to 'general'.")

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().format(current_time=current_time)
        return content
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}")
        # 비상용 기본 프롬프트
        return "You are a helpful assistant."

def parse_system_prompt(prompt_content: str) -> dict:
    """시스템 프롬프트를 각 섹션별로 파싱하여 딕셔너리로 반환합니다."""
    sections = {}
    # 정규식을 사용하여 <TAG_NAME> 형식의 태그를 찾습니다.
    pattern = re.compile(r"<([A-Z_]+)>", re.DOTALL)
    parts = pattern.split(prompt_content)
    
    if len(parts) > 1:
        # 첫 번째 부분은 보통 비어있으므로 무시하고, 태그와 내용을 짝지어 저장합니다.
        for i in range(1, len(parts), 2):
            tag = parts[i]
            content = parts[i+1].strip()
            # 다음 태그가 나오기 전까지의 내용을 모두 포함합니다.
            if '----' in content:
                content = content.split('----')[0].strip()
            sections[tag] = f"<{tag}>\n{content}\n</{tag}>"
    return sections

# --- Plan-and-Execute 상태 및 모델 정의 ---
class PlanExecute(TypedDict):
    """Plan-and-Execute 그래프의 상태를 정의합니다."""
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    routing_mode: str # 라우팅 모드를 저장 (general, feedback_report, cortex)
    chat_history: List[BaseMessage] # 전체 대화 기록
    summary: str # 누적 요약

class Plan(BaseModel):
    """Planner가 생성할 계획의 구조입니다."""
    steps: List[str] = Field(
        description="실행할 단계들의 목록, 순서대로 정렬되어야 합니다."
    )

class Response(BaseModel):
    """Replanner가 최종 답변을 생성할 때의 구조입니다."""
    response: str

class Act(BaseModel):
    """Replanner의 행동을 결정합니다. (계획 수정 또는 최종 답변)"""
    action: Union[Response, Plan] = Field(
        description="수행할 액션입니다. 사용자에게 답변하려면 Response를, 추가 작업이 필요하면 Plan을 사용하세요."
    )


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((HttpxReadError, HttpcoreReadError, ConnectionError, anthropic.APIError))
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

    # BaseMessage 객체에서 content 속성을 추출하여 텍스트로 변환
    conversation_text = "\n\n".join(
        f"{'사용자' if msg.type == 'human' else 'AI'}: {msg.content}"
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
        # 오류 발생 시 기존 요약 또는 대체 텍스트 반환
        return existing_summary or f"이전 대화 {len(messages_to_summarize)}개 (요약 생성 중 오류 발생)"


### LLM 모델 생성함수
def create_llm_model(model_name: str):
    """
    모델 이름에 따라 Anthropic(Claude) 또는 Google(Gemini) 모델을 생성합니다.
    """
    model_provider = model_name.lower() # 소문자로 변환하여 비교
    
    if "claude" in model_provider:
        model = ChatAnthropic(
            model=model_name,
            temperature=0.1,
            streaming=True,
            max_tokens=16000,
            max_retries=3,
        )
    elif "gemini" in model_provider:
        try:
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.1,
                max_output_tokens=16000,
                max_retries=3,
            )
            print(f"[DEBUG] Google 모델 생성 성공: {model_name}")
        except Exception as e:
            print(f"[ERROR] Google 모델 생성 실패: {e}")
            raise ValueError(f"Google 모델 생성 실패: {model_name} - {e}")

    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
    
    return model

def create_plan_and_execute_graph(model_name: str, all_langchain_tools, prompt_type: str):
    """
    주어진 모델과 도구를 사용하여 조건부 라우팅 기능이 포함된
    Plan-and-Execute LangGraph 워크플로우를 생성합니다.
    """
    # 0. 시스템 프롬프트 로드 및 파싱
    prompt_content = load_system_prompt(prompt_type)
    prompt_sections = parse_system_prompt(prompt_content)
    
    # 1. 실행 에이전트 (기존 ReAct 에이전트)
    execution_llm = create_llm_model(model_name)
    agent_executor = create_react_agent(execution_llm, all_langchain_tools)

    # 2. Planner LLM (will be used to build dynamic planners)
    planner_llm = create_llm_model(model_name)

    # 3. Replanner 설정 (이 LLM은 동적 프롬프트로 최종 답변 생성 시에도 사용됨)
    replanner_llm = create_llm_model(model_name)

    # 4. 그래프 노드 정의
    async def summarization_node(state: PlanExecute):
        """대화 기록을 바탕으로 요약을 생성하거나 업데이트합니다."""
        chat_history = state.get("chat_history", [])
        existing_summary = state.get("summary", "")

        # 요약은 대화 기록이 4개 이상일 때만 의미가 있습니다.
        if len(chat_history) < 4:
            print("[DEBUG] 대화 기록이 짧아 요약을 건너뜁니다.")
            return {"summary": existing_summary, "chat_history": chat_history}

        try:
            # 요약 대상(마지막 2개 제외)과 유지할 최신 대화 분리
            messages_to_summarize = chat_history[:-2]
            recent_messages = chat_history[-2:]
            
            print(f"[DEBUG] 누적 요약 시작. 대상: {len(messages_to_summarize)}개, 유지: {len(recent_messages)}개")

            updated_summary = await create_intelligent_summary_silent(messages_to_summarize, existing_summary)

            # 요약본과 최신 대화로 새로운 대화 기록 구성
            new_chat_history = [SystemMessage(content=f"--- 누적 요약 ---\n{updated_summary}")] + recent_messages
            
            print("[DEBUG] 요약 완료. 새로운 대화 기록으로 업데이트합니다.")
            return {"summary": updated_summary, "chat_history": new_chat_history}

        except Exception as e:
            print(f"[ERROR] 요약 노드 실행 중 오류 발생: {e}")
            # 오류 발생 시에도 원래 상태를 최대한 보존하여 다음 단계로 진행
            return {"summary": existing_summary, "chat_history": chat_history}


    async def initial_router(state: PlanExecute):
        """사용자 입력에 따라 초기 라우팅 모드를 설정합니다."""
        # 사용자의 입력에서 공백을 제거하여 "정량피드백" 키워드를 유연하게 감지
        normalized_input = state["input"].replace(" ", "")
        if "정량피드백" in normalized_input:
            print("[DEBUG] '정량 피드백' 키워드 감지. 라우팅 모드를 'feedback_report'로 설정합니다.")
            mode = "feedback_report"
            return {"routing_mode": mode}
        else:
            mode = "general"
            print("[DEBUG] 'general' 모드: Planner를 건너뛰고, 사용자 입력을 바로 실행 계획으로 설정합니다.")
            return {"routing_mode": mode, "plan": [state["input"]]}

    async def execute_step(state: PlanExecute):
        """계획의 첫 단계를 실행합니다."""
        task = state["plan"][0]
        
        # 요약본과 대화 기록을 프롬프트에 포함하여 컨텍스트 강화
        summary = state.get("summary", "")
        chat_history = state.get("chat_history", [])
        
        # 실행 에이전트를 위한 시스템 프롬프트 구성
        execution_system_prompt_parts = [
            prompt_sections.get("ROLE", ""),
            prompt_sections.get("INSTRUCTIONS", ""),
        ]
        execution_system_prompt = "\n".join(part for part in execution_system_prompt_parts if part)
        
        # HumanMessage, AIMessage 등을 포함한 전체 대화 기록을 컨텍스트로 활용
        context_messages = []
        if execution_system_prompt:
            context_messages.append(SystemMessage(content=execution_system_prompt))

        if summary:
            context_messages.append(SystemMessage(content=f"--- 대화 요약 ---\n{summary}"))
        
        # chat_history에 있는 메시지를 그대로 사용
        context_messages.extend(chat_history)
        
        # 현재 task를 HumanMessage로 추가
        context_messages.append(HumanMessage(content=task))

        # ReAct 에이전트 호출 시 messages 형식에 맞게 전달
        agent_response = await agent_executor.ainvoke({"messages": context_messages})

        # 모델 응답이 리스트일 수 있는 경우를 처리 (Gemini 등)
        raw_content = agent_response["messages"][-1].content
        if isinstance(raw_content, list):
            # 텍스트 부분만 추출하여 하나의 문자열로 합칩니다.
            content_str = "".join(part.get("text", "") for part in raw_content if part.get("type") == "text")
        else:
            content_str = str(raw_content)

        # 어떤 도구가 사용되었는지 확인 (더 안정적인 방식으로 수정)
        tool_name = ""
        from langchain_core.messages import AIMessage
        for message in reversed(agent_response.get("messages", [])):
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_name = message.tool_calls[0].get("name", "")
                break

        return {"past_steps": [(task, content_str, tool_name)]} # 도구 이름도 함께 저장

    async def plan_step(state: PlanExecute):
        """사용자 입력을 바탕으로 계획 수립"""
        
        # 시스템 프롬프트에 요약본 추가
        summary = state.get("summary", "")
        summary_prompt_part = ""
        if summary:
            summary_prompt_part = f"<CONVERSATION_SUMMARY>\n{summary}\n</CONVERSATION_SUMMARY>"

        # 기본 시스템 프롬프트 구성
        system_prompt_parts = [
            prompt_sections.get("<ROLE", ""),
            prompt_sections.get("GRAPH_OF_THOUGHTS_METHODOLOGY", ""),
            summary_prompt_part # 요약 부분 추가
        ]
        # 기본 사용자 프롬프트
        user_prompt_template = "주어진 목표에 대해 5단계별 계획을 세워주세요. 마지막은 값이 맞는지 검증단계로 구성해주세요.: {input}"

        # 라우팅 모드에 따라 Planner 프롬프트 강화
        routing_mode = state.get("routing_mode", "general")
        config = ROUTING_CONFIG.get(routing_mode, ROUTING_CONFIG["general"])
        
        business_rule_paths = config.get("planner_business_rules", [])
        all_rules_content = [load_file_content(path) for path in business_rule_paths if path]

        # CONFIG 설정에 따라, planner가 응답 형식을 참고할지 결정
        if config.get("add_response_format_to_planner", False):
            response_format_content = load_file_content(config["response_format"])
            if response_format_content:
                print(f"[DEBUG] {routing_mode} 모드: 응답 포맷을 Planner에게 전달합니다.")
                all_rules_content.append(response_format_content)

        if any(all_rules_content):
            print(f"[DEBUG] '{routing_mode}' 모드로 Planner를 구성합니다.")
            system_prompt_parts.extend(filter(None, all_rules_content))

            # CONFIG에서 Planner 최종 지시사항 가져오기
            final_instruction = config.get("planner_final_instruction")
            if final_instruction:
                system_prompt_parts.append(final_instruction)
                
            user_prompt_template = "다음 사용자 요청에 대한 실행 계획을 수립하세요: {input}"
            if routing_mode == "feedback_report":
                user_prompt_template = "다음 사용자 요청에 대한 실행 계획을 최대 5단계 수립하세요. 사용자가 언급한 특정 브랜드가 있다면, 각 계획 단계에 해당 브랜드를 명시적으로 포함해야 합니다. 예: '[브랜드명] 판매량 조회'. 요청: {input}"

        final_system_prompt = "\n".join(part for part in system_prompt_parts if part)
        
        # 대화 기록을 포함하여 Planner 호출
        chat_history_for_planner = state.get("chat_history", [])
        
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", final_system_prompt),
            # 이전 대화 내용도 함께 전달하여 맥락 이해도 높임
            *chat_history_for_planner, 
            ("user", user_prompt_template),
        ])

        planner = planner_prompt | planner_llm.with_structured_output(Plan)
        
        plan_result = await planner.ainvoke({"input": state["input"]})

        # 방어 코드
        if plan_result is None:
            print("[WARNING] Planner가 유효한 계획을 생성하지 못했습니다. 빈 계획으로 진행합니다.")
            return {"plan": []}
        return {"plan": plan_result.steps}

    async def replan_step(state: PlanExecute):
        """실행 결과를 바탕으로 계획을 수정하거나 최종 답변을 생성합니다."""
        routing_mode = state.get("routing_mode", "general")
        config = ROUTING_CONFIG.get(routing_mode, ROUTING_CONFIG["general"])
        chat_history = state.get("chat_history", [])
        summary = state.get("summary", "")

        # General 모드일 때와, 다른 모드의 마지막 단계일 때 최종 응답 생성
        if routing_mode == "general" or len(state["plan"]) <= 1:
            # 1. 기본 시스템 프롬프트 및 라우팅에 따른 응답 형식 로드
            final_prompt_parts = [
                prompt_sections.get("ROLE", ""),
                load_file_content(config["response_format"])
            ]
            
            # 2. Cortex 도구 사용이 감지되면, 최종 응답 생성 시 Cortex 규칙을 동적으로 추가
            is_cortex_tool_used = any(step[2] in ["fashion_analyst", "market_analyst"] for step in state["past_steps"])
            if is_cortex_tool_used:
                print("[DEBUG] Cortex 도구 사용 감지. Cortex 규칙을 최종 응답 생성에 적용합니다.")
                cortex_rules = load_file_content("prompt/business_rules/cortex.txt")
                if cortex_rules:
                    final_prompt_parts.append(cortex_rules)
            
            # 3. CONFIG에서 사용자 프롬프트 템플릿 가져오기
            user_prompt_template = config["replanner_user_prompt"]

            final_prompt_str = "\n".join(part for part in final_prompt_parts if part)
            
            # 요약 및 대화 기록을 프롬프트에 추가
            context_messages = []
            if summary:
                context_messages.append(SystemMessage(content=f"<대화 요약>\n{summary}\n</대화 요약>"))
            context_messages.extend(chat_history)


            # General 모드에서는 Act를 사용하여 Replan 또는 Response를 결정
            if routing_mode == "general":
                replanner_prompt = ChatPromptTemplate.from_messages([
                    ("system", final_prompt_str + """
You are a 'replanner'. Your role is to analyze the user's request and the execution history to determine the next action.
Based on the provided `past_steps`, decide whether to provide a final response to the user or to revise the plan for further action.
If the execution result is sufficient to answer the user's question, generate a final response using the `Response` model.
If the result is insufficient or requires further steps, create a new plan using the `Plan` model.
                    """),
                    *context_messages, # 대화기록과 요약 추가
                    ("user", user_prompt_template)
                ])
                
                replanner = replanner_prompt | replanner_llm.with_structured_output(Act)
                
                # 스트리밍 처리를 위해 astream 사용
                response_chunks = []
                async for chunk in replanner.astream({"input": state["input"], "past_steps": state["past_steps"]}):
                    if isinstance(chunk.action, Response):
                        yield {"response": chunk.action.response}
                    elif isinstance(chunk.action, Plan):
                        # Plan이 완전히 생성될 때까지 기다렸다가 반환
                        yield {"plan": chunk.action.steps}

            # 다른 모드에서는 바로 최종 답변 스트리밍
            else:
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system", final_prompt_str),
                    *context_messages, # 대화기록과 요약 추가
                    ("user", user_prompt_template)
                ])
                final_responder = final_prompt | replanner_llm
                
                # 스트리밍을 위해 astream 사용
                response_content = ""
                async for chunk in final_responder.astream({"input": state["input"], "past_steps": state["past_steps"]}):
                    response_content += chunk.content
                    yield {"response": response_content}
                # 스트리밍 완료 후 추가적인 yield를 방지하여 루프 발생을 막습니다.
                return

        # 아직 계획을 더 실행해야 하는 경우 (non-general mode)
        else:
            new_plan = state["plan"][1:]
            yield {"plan": new_plan}

    # 5. 그래프 조건부 엣지(분기) 로직
    def decide_after_summarization(state: PlanExecute):
        """요약 노드 이후, 라우팅 모드에 따라 분기합니다."""
        if state.get("routing_mode") == "general":
            print("[DEBUG] 'general' 모드 감지. Planner를 건너뛰고 바로 agent로 이동합니다.")
            return "agent"
        else:
            print(f"[DEBUG] '{state.get('routing_mode')}' 모드 감지. Planner로 이동합니다.")
            return "planner"
            
    def decide_after_planning(state: PlanExecute):
        """플래너가 계획을 생성했는지 여부에 따라 분기합니다."""
        if not state.get("plan"):
            print("[DEBUG] Planner가 계획을 생성하지 못했으므로, 바로 Replanner로 이동합니다.")
            return "replan"
        return "agent"
        
    def should_continue(state: PlanExecute) -> str:
        """계획이 남아있는지, 아니면 종료해야 하는지 결정합니다."""
        if not state["plan"] or ("response" in state and state["response"]):
            print("[DEBUG] 조건 확인(should_continue): 종료 조건 충족. (plan 없음 또는 response 있음)")
            return END
        print("[DEBUG] 조건 확인(should_continue): 계속 진행. (plan 남음, response 없음)")
        return "agent"

    # 6. 그래프 구성
    workflow = StateGraph(PlanExecute)
    workflow.add_node("initial_router", initial_router)
    workflow.add_node("summarizer", summarization_node) # 요약 노드 추가
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "initial_router")
    
    # initial_router -> summarizer로 직접 연결
    workflow.add_edge("initial_router", "summarizer")

    # summarizer 이후에 조건부 분기 추가
    workflow.add_conditional_edges(
        "summarizer",
        decide_after_summarization,
        {"planner": "planner", "agent": "agent"},
    )
    
    # planner 이후에 조건부 분기 추가
    workflow.add_conditional_edges(
        "planner",
        decide_after_planning,
        {"agent": "agent", "replan": "replan"},
    )
    workflow.add_edge("agent", "replan")
    
    workflow.add_conditional_edges(
        "replan", 
        should_continue, 
        {
            END: END, 
            "agent": "agent"
        }
    )

    return workflow.compile() 
