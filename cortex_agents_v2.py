from typing import Any, Dict, Tuple, List, Optional
import httpx
from mcp.server.fastmcp import FastMCP
import os
import json
import uuid
from dotenv import load_dotenv, find_dotenv
import logging
import sys
import traceback

# Setup logging to stderr only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)

# Load environment
load_dotenv(find_dotenv())

# Initialize FastMCP server
mcp = FastMCP("cortex_agent")

# Constants
SEMANTIC_MODEL_FILE = os.getenv("SEMANTIC_MODEL_FILE", default="@CORTEX_ANALYST_DEMO.LLM_POC.CORTEX_STAGE/fashion_streamlit.yaml")
CRAWLING_SEMANTIC_MODEL_FILE = os.getenv("CRAWLING_SEMANTIC_MODEL_FILE", default="@CORTEX_ANALYST_DEMO.LLM_POC.CORTEX_STAGE/fashion_keyword.yaml")
CORTEX_SEARCH_SERVICE = os.getenv("CORTEX_SEARCH_SERVICE", default="CORTEX_ANALYST_DEMO.LLM_POC.DOCS_MEETING")
SNOWFLAKE_ACCOUNT_URL = os.getenv("SNOWFLAKE_ACCOUNT_URL", default="https://a1041574869271-eland-partner.snowflakecomputing.com")
SNOWFLAKE_PAT = os.getenv("SNOWFLAKE_PAT", default="yJraWQiOiIyMTA0NTc2OTI3OTg5ODIiLCJhbGciOiJFUzI1NiJ9.eyJwIjoiMTI1NDQyNjA6MzIxMTMzMDA1MyIsImlzcyI6IlNGOjEwMzIiLCJleHAiOjE3ODE4MzA2NDZ9.zIsLtQrtoIp_WYZJyv_bU6U_KQ8Vr0T8MICvAjpgOb0wilLCwPcSAvH4wKzE4OGg8lFdEN7tR1X7KmfE56XxKg")

if not SNOWFLAKE_PAT:
    raise RuntimeError("Set SNOWFLAKE_PAT environment variable")
if not SNOWFLAKE_ACCOUNT_URL:
    raise RuntimeError("Set SNOWFLAKE_ACCOUNT_URL environment variable")

# Headers for API requests
API_HEADERS = {
    "Authorization": f"Bearer {SNOWFLAKE_PAT}",
    "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    "Content-Type": "application/json",
}


async def process_sse_response(resp: httpx.Response) -> Tuple[str, str, List[Dict]]:
    """Process Server-Sent Events response from Cortex Agent."""
    text, sql, citations = "", "", []
    async for raw_line in resp.aiter_lines():
        if not raw_line:
            continue
        raw_line = raw_line.strip()
        if not raw_line.startswith("data:"):
            continue
        payload = raw_line[len("data:") :].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            evt = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = evt.get("delta") or evt.get("data", {}).get("delta")
        if not isinstance(delta, dict):
            continue
        for item in delta.get("content", []):
            t = item.get("type")
            if t == "text":
                text += item.get("text", "")
            elif t == "tool_results":
                for result in item["tool_results"].get("content", []):
                    if result.get("type") == "json":
                        j = result["json"]
                        text += j.get("text", "")
                        if "sql" in j:
                            sql = j["sql"]
                        for s in j.get("searchResults", []):
                            citations.append({
                                "source_id": s.get("source_id"),
                                "doc_id": s.get("doc_id"),
                            })
    return text, sql, citations


async def execute_sql(sql: str) -> Dict[str, Any]:
    """Execute SQL query on Snowflake."""
    try:
        request_id = str(uuid.uuid4())
        sql_api_url = f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/statements"
        sql_payload = {
            "statement": sql.replace(";", ""),
            "timeout": 120
        }
        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            response = await client.post(
                sql_api_url,
                json=sql_payload,
                headers=API_HEADERS,
                params={"requestId": request_id},
            )
        
        logging.info("SQL API 요청: %s %s", sql_api_url, sql_payload)
        logging.info("SQL API 응답: %s %s", response.status_code, response.text)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"SQL API error: {response.status_code} {response.text}"}
    except Exception as e:
        logging.error("SQL execution error: %s\n%s", e, traceback.format_exc())
        return {"error": f"SQL execution error: {e}\n{traceback.format_exc()}"}


@mcp.tool(description="내부 비즈니스 데이터 분석 - 매출, 수익, 고객 지표, 운영 KPI 등 회사의 내부 데이터를 분석합니다.")
async def analyze_business_data(query: str) -> Dict[str, Any]:
    """
    내부 비즈니스 데이터 분석 도구
    
    회사의 내부 비즈니스 데이터를 분석하여 SQL 쿼리 결과만 반환합니다:
    - 매출 성과 및 수익 분석
    - 운영 KPI 및 성과 지표
    - 재무 보고서 및 비즈니스 인텔리전스
    - 내부 데이터 분석 및 리포팅
    
    Args:
        query (str): 내부 비즈니스 데이터, 매출, 수익, 회사 성과에 대한 질문
        
    Returns:
        dict: SQL 쿼리 실행 결과만 포함
    """
    
    payload = {
        "model": "claude-4-sonnet",
        "response_instruction": (
            "당신은 내부 회사 데이터 전문 비즈니스 인텔리전스 분석가입니다. "
            "Text-to-SQL 도구를 사용하여 매출, 수익, 고객, 운영 데이터를 분석하세요. "
            "내부 회사 성과 데이터로부터 실행 가능한 비즈니스 인사이트를 제공하는 데 집중하세요. "
            "트렌드, 패턴, 핵심 비즈니스 지표를 드러내는 SQL 쿼리를 생성하세요. "
            "항상 SQL 쿼리를 실행하여 구체적인 데이터 결과를 포함하세요."
        ),
        "experimental": {},
        "tools": [
            {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "BusinessDataAnalyzer"}},
            {"tool_spec": {"type": "sql_exec", "name": "SQLExecution"}},
        ],
        "tool_resources": {
            "BusinessDataAnalyzer": {"semantic_model_file": SEMANTIC_MODEL_FILE},
        },
        "tool_choice": {"type": "auto"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": query}]}
        ],
    }

    request_id = str(uuid.uuid4())
    url = f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/cortex/agent:run"
    headers = {
        **API_HEADERS,
        "Accept": "text/event-stream",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                params={"requestId": request_id},
            ) as resp:
                resp.raise_for_status()
                text, sql, citations = await process_sse_response(resp)

        # Execute the generated SQL if available
        results = None
        if sql:
            results = await execute_sql(sql)
            # Return only the SQL execution results
            if results and "error" not in results:
                return {
                    "results": results,
                    "sql": sql,
                    "analysis_type": "internal_business",
                    "success": True
                }
        
        return {
            "error": "SQL 쿼리가 생성되지 않았거나 실행에 실패했습니다.",
            "text": text,
            "sql": sql,
            "citations": citations,
            "success": False
        }
        
    except Exception as e:
        logging.error("Business data analysis error: %s\n%s", e, traceback.format_exc())
        return {
            "error": f"비즈니스 데이터 분석 실패: {e}",
            "success": False
        }


@mcp.tool(description="외부 시장 데이터 분석 - 경쟁사 분석, 시장 트렌드, 소비자 리뷰, 가격 정보 등 외부 시장 인텔리전스를 분석합니다.")
async def analyze_market_intelligence(query: str) -> Dict[str, Any]:
    """
    외부 시장 데이터 분석 도구
    
    크롤링된 외부 시장 인텔리전스 데이터를 분석하여 SQL 쿼리 결과만 반환합니다:
    - 경쟁사 분석 및 벤치마킹
    - 시장 트렌드 및 업계 인사이트
    - 소비자 리뷰 및 감정 분석
    - 가격 인텔리전스 및 경쟁 가격 분석
    - 브랜드 인식 및 시장 포지셔닝
    
    Args:
        query (str): 경쟁사, 시장 트렌드, 가격, 외부 시장 인텔리전스에 대한 질문
        
    Returns:
        dict: SQL 쿼리 실행 결과만 포함
    """
    
    payload = {
        "model": "claude-4-sonnet",
        "response_instruction": (
            "당신은 경쟁사 및 외부 시장 데이터 전문 시장 인텔리전스 분석가입니다. "
            "Text-to-SQL 도구를 사용하여 경쟁사 성과, 시장 트렌드, 소비자 인사이트를 분석하기 위한 SQL 쿼리를 생성하세요. "
            "항상 SQL 쿼리를 실행하여 구체적인 데이터 결과를 포함하세요."
        ),
        "experimental": {},
        "tools": [
            {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "MarketIntelligenceAnalyzer"}},
            {"tool_spec": {"type": "sql_exec", "name": "SQLExecution"}},
        ],
        "tool_resources": {
            "MarketIntelligenceAnalyzer": {"semantic_model_file": CRAWLING_SEMANTIC_MODEL_FILE},
        },
        "tool_choice": {"type": "auto"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": query}]}
        ],
    }

    request_id = str(uuid.uuid4())
    url = f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/cortex/agent:run"
    headers = {
        **API_HEADERS,
        "Accept": "text/event-stream",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                params={"requestId": request_id},
            ) as resp:
                resp.raise_for_status()
                text, sql, citations = await process_sse_response(resp)

        # Execute the generated SQL if available
        results = None
        if sql:
            results = await execute_sql(sql)
            # Return only the SQL execution results
            if results and "error" not in results:
                return {
                    "results": results,
                    "sql": sql,
                    "analysis_type": "market_intelligence",
                    "success": True
                }
        
        return {
            "error": "SQL 쿼리가 생성되지 않았거나 실행에 실패했습니다.",
            "text": text,
            "sql": sql,
            "citations": citations,
            "success": False
        }
        
    except Exception as e:
        logging.error("Market intelligence analysis error: %s\n%s", e, traceback.format_exc())
        return {
            "error": f"시장 인텔리전스 분석 실패: {e}",
            "success": False
        }


if __name__ == "__main__":
    try:
        logging.info("Cortex Agent MCP 서버 시작...")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logging.info("Cortex Agent MCP 서버가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logging.error(f"Cortex Agent MCP 서버 실행 중 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)