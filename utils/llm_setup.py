from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from datetime import datetime, timezone, timedelta


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

def create_agent(model_name: str, all_langchain_tools):
    """
    주어진 모델과 도구를 사용하여 LangGraph 에이전트를 생성합니다.
    시스템 프롬프트는 전처리 단계에서 메시지에 추가됩니다.
    """
    model = create_llm_model(model_name)
    
    agent_core = create_react_agent(
        model,
        all_langchain_tools
    )
    return agent_core 