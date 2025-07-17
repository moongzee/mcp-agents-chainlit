from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from datetime import datetime, timezone, timedelta
import re

# --- Plan-and-Execute 관련 추가 임포트 ---
import operator
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
# ---

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

# --- 조건부 라우팅을 위한 규칙 정의 이거 따로 어딘가 관리 필요 ( 비즈니스 로직과 비즈니스 포맷 정의 용 ) 
CORTEX_SPECIALIZED_MODE = """
<CORTEX_SPECIALIZED_MODE>
**Cortex/Snowflake 환경 감지 시 특화 모드 활성화**
✅ 데이터 조회 기준

1. **지표 기준일 자동 인식**: 사용자가 "전년", "전월" 등을 명시하면 시계열 필터 반영.
2. **지표 구성 방식**: 모든 정량 지표는 `지표명 + 수치 + 기준 대비 증감` 형식으로 조회.
3. **지표 단위 정규화**: 금액은 억 단위 정수, 비율은 소수점 1자리 %로 표현.
4. **예측/시뮬레이션 데이터**: 과거 데이터 기반 추정치 사용 및 달성률/편차액으로 정량 표현.
5. **정량/정성 피드백 분리 조회**: Survey/고객 의견 조회 시 수치와 텍스트 코멘트 분리 추출.

🚫 데이터 신뢰성 원칙: 임의 수치 생성 금지, 조회된 데이터만 사용, 결과 없을 시 "해당 데이터 없음" 명시.
</CORTEX_SPECIALIZED_MODE>
"""
FEEDBACK_REPORT_RULES = """
<AMIS_FEEDBACK_REQUEST_RULES>
✅ 지표 매핑 규칙:
- 외형매출 → `영업매출`
- 영업이익 → `영업이익`
- 매총익 → `매출총이익`
- 매총율 → `매출총이익 / 영업매출`
- 판관비 → `판매관리비`
- 판관비율 → `판매관리비 / 영업매출`

✅ 응답 구조 강제:
- 반드시 <BUSINESS_REPORT_FORMAT> 형식 유지
- 소수점 1자리로 정규화
- 누락 데이터는 “데이터 없음” 명시

✅ 지표 단위 정규화:
- 금액 → 억 단위 정수
- 비율 → 소수점 1자리 `%` 표기
- 비율 변화량 → `%p` (퍼센트포인트) 표기

✅ 필수 비교 대상 포함 규칙:
- **모든 정량 지표는 반드시 아래 세 항목을 포함해야 함**:
  1. **당월 수치**
  2. **전월 대비 증감량 또는 증감률**
  3. **목표 대비 달성률 또는 편차**
- 예시:
  - `"외형매출 122억(전년동월대비 +6.2%, 목표대비 97% 달성)"`
  - `"판관비율 15.2%(전년동월대비 -0.3%p, 목표대비 +0.5%p)"`

✅ 표현 방식:
- 증감량은 `±X%`, `±X억`, `±X%p` 등으로 표현
- 목표대비 항목은 `XX% 달성`, `목표대비 ±X억/±X%p` 형태로 명시
- 두 가지 비교(전년동월, 목표) 모두 없으면 **불완전 응답으로 간주**

🚫 금지:
- 수치 환각, 추정, 예시 생성 절대 금지
- 목표 대비 수치를 도구 결과 없이 생성 금지
- 지표 누락 시 응답에서 은폐 금지 — 반드시 `"데이터 없음"`으로 표시
</AMIS_FEEDBACK_REQUEST_RULES>
----
<BUSINESS_REPORT_FORMAT>
When User requets "X월 정량 피드백" or similar business report format, use this structure:

## [브랜드명] X 월 정량 피드백
### X 월 당월 외형매출 XXX억(전년동월대비 ±X%), 영업이익 XXX억(전년동월대비 ±XX억)

**1. [매출] 오프라인 XX억(전년동월대비 ±X%, XX% 목표달성) / 온라인 XX억(전년동월대비 ±X%, XX% 목표달성) 
- [오프라인/온라인 매출에 대한 전월 대비 매출동향 한줄 요약내용] 

**2. [매총익] XX억(전년동월대비 ±X%, XX% 목표달성) / 매총율 XX%(전년동월대비 ±X%p, 목표대비 ±X%p)
- [매출액과 매출원가 관련하여 한줄 분석 요약 내용]

**3. [판관비] XX억(전년동월대비 ±X%, XX% 목표달성) / 판관비율 XX%(전년동월대비 ±X%p, 목표대비 ±X%p)
- [판관비 항목중 비중이 높은 것에 대해 두줄 분석 요약 내용]

### 브랜드 X월 AMIS 데이터  
| 항목     | 금액 (억원)|
|----------|-----------|
|매출액     |      XX 억|
|매출원가    |     XX 억| 
|매출총이익  |     XX 억|
|영업이익    |     XX 억|
|당기순이익  |     XX 억|
|판매관리비  |     XX 억|

**📝 FORMATTING RULES:**
- ✅ 모든 금액은 억 단위, 비율은 % 혹은 %p로 정규화.
- ✅ 모든 지표는 ‘전년동월대비’, ‘목표대비’ 수치를 반드시 포함.
- ✅ 데이터 누락 시 “데이터 없음”으로 명시.
- ✅ **CRITICAL**: 절대로 데이터를 임의로 만들지 말 것.
</BUSINESS_REPORT_FORMAT>
"""

# --- Plan-and-Execute 상태 및 모델 정의 ---
class PlanExecute(TypedDict):
    """Plan-and-Execute 그래프의 상태를 정의합니다."""
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    routing_mode: str # 라우팅 모드를 저장 (general, feedback_report, cortex)

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
    async def initial_router(state: PlanExecute):
        """사용자 입력에 따라 초기 라우팅 모드를 설정합니다."""
        if "정량" in state["input"]:
            mode = "feedback_report"
        else:
            mode = "general"
        return {"routing_mode": mode}

    async def execute_step(state: PlanExecute):
        """계획의 첫 단계를 실행합니다."""
        task = state["plan"][0]
        agent_response = await agent_executor.ainvoke({"messages": [("user", task)]})
        
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
        
        # 기본 시스템 프롬프트 구성
        system_prompt_parts = [
            prompt_sections.get("ROLE", ""),
            prompt_sections.get("GRAPH_OF_THOUGHTS_METHODOLOGY", "")
        ]
        # 기본 사용자 프롬프트
        user_prompt_template = "주어진 목표에 대해 5단계별 계획을 세워주세요. 마지막은 값이 맞는지 검증단계로 구성해주세요.: {input}"

        # '정량 피드백' 모드일 경우 프롬프트 강화
        if state.get("routing_mode") == "feedback_report":
            print("[DEBUG] '정량 피드백' 모드로 Planner를 구성합니다.")
            system_prompt_parts.append(FEEDBACK_REPORT_RULES)
            system_prompt_parts.append(
                "위의 '정량 피드백' 규칙과 '비즈니스 리포트 형식'을 참고하여, 보고서 작성에 필요한 모든 데이터를 수집하기 위한 상세하고 구체적인 실행 계획을 세워주세요. "
                "마지막 단계는 값이 맞는지 검증단계로 구성해주세요."
            )
            user_prompt_template = "다음 사용자 요청에 대한 실행 계획을 수립하세요: {input}"
        
        final_system_prompt = "\n".join(part for part in system_prompt_parts if part)
        
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", final_system_prompt),
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
        # cortex-agents 도구 사용 여부 확인
        if any(step[2] in ["fashion_analyst", "market_analyst"]  for step in state["past_steps"]):
            state["routing_mode"] = "cortex"

        # 최종 응답 생성 단계인지 확인
        # 간단한 로직: 계획이 1개 남았고, 실행 후 최종 답변을 해야 할 때
        if len(state["plan"]) == 1:
            # 현재 상태와 라우팅 모드에 기반하여 동적 프롬프트 생성
            final_prompt_parts = [prompt_sections.get("ROLE", ""), prompt_sections.get("OUTPUT_FORMAT", "")]
            if state["routing_mode"] == "feedback_report":
                final_prompt_parts.append(FEEDBACK_REPORT_RULES)
            elif state["routing_mode"] == "cortex":
                final_prompt_parts.append(CORTEX_SPECIALIZED_MODE)
            
            final_prompt_str = "\n".join(part for part in final_prompt_parts if part) # None이 아닌 경우에만 join
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", final_prompt_str),
                ("user", "다음 정보를 바탕으로 최종 보고서를 작성해주세요.\n\n사용자 질문: {input}\n\n지금까지의 작업 내역:\n{past_steps}")
            ])
            
            # 최종 답변 생성
            final_responder = final_prompt | replanner_llm
            final_response = await final_responder.ainvoke({"input": state["input"], "past_steps": state["past_steps"]})
            return {"response": final_response.content}
        
        # 아직 계획을 더 실행해야 하는 경우
        else:
            # 현재 계획에서 완료된 첫 번째 단계를 제외
            new_plan = state["plan"][1:]
            return {"plan": new_plan}

    # 5. 그래프 조건부 엣지(분기) 로직
    def decide_after_planning(state: PlanExecute):
        """플래너가 계획을 생성했는지 여부에 따라 분기합니다."""
        if not state.get("plan"):
            print("[DEBUG] Planner가 계획을 생성하지 못했으므로, 바로 Replanner로 이동합니다.")
            return "replan"
        return "agent"
        
    def should_continue(state: PlanExecute) -> str:
        """계획이 남아있는지, 아니면 종료해야 하는지 결정합니다."""
        if not state["plan"] or "response" in state and state["response"]:
            return END
        return "agent"

    # 6. 그래프 구성
    workflow = StateGraph(PlanExecute)
    workflow.add_node("initial_router", initial_router)
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "initial_router")
    workflow.add_edge("initial_router", "planner")
    
    # planner 이후에 조건부 분기 추가
    workflow.add_conditional_edges(
        "planner",
        decide_after_planning,
        {"agent": "agent", "replan": "replan"},
    )
    workflow.add_edge("agent", "replan")
    
    workflow.add_conditional_edges("replan", should_continue, {END: END, "agent": "agent"})

    return workflow.compile() 