from typing import Any, Dict, Tuple, List, Optional
import os
import sys
import json
import uuid
import httpx
import asyncio
import requests
import logging
import traceback
from dotenv import load_dotenv, find_dotenv
from mcp.server.fastmcp import FastMCP
from typing import Literal
from openai import AsyncOpenAI

# Setup logging to stderr only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)

# Load environment
load_dotenv(find_dotenv())

client = AsyncOpenAI()
# Initialize FastMCP server
mcp = FastMCP("cortex_agent")

# Constants
SNOWFLAKE_ACCOUNT_URL = os.getenv("SNOWFLAKE_ACCOUNT_URL")
SNOWFLAKE_PAT = os.getenv("SNOWFLAKE_PAT")
SEMANTIC_MODEL_PATH = os.getenv("SEMANTIC_MODEL_PATH", "@CORTEX_ANALYST_DEMO.LLM_POC.CORTEX_STAGE/")

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

def safe_string_convert(value: Any) -> str:
    """안전하게 값을 문자열로 변환"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


async def call_cortex_analyst(query: str, semantic_model_file: Optional[str] = None):
    """Cortex Analyst API 호출"""
    try:
        request_id = str(uuid.uuid4())
        url = f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/cortex/analyst/message"

        headers = {
            **API_HEADERS,
            "Accept": "text/event-stream",
        }

        messages = [
            {"role": "user", "content": [{"type": "text", "text": query}]}
        ]

        payload = {
            "messages": messages,
            "semantic_model_file": SEMANTIC_MODEL_PATH + semantic_model_file + '.yaml'
        }

        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    params={"requestId": request_id},
                )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Cortex Analyst API error: {response.status_code} {response.text}"}
    except Exception as e:
        logging.error(f"Call Cortex Analyst error: {e}")
        return {"error": f"Call Cortex Analyst error: {e}"}

async def execute_sql(sql: str) -> Dict[str, Any]:
    """SQL을 실행하고 결과를 반환"""
    try:
        request_id = str(uuid.uuid4())
        url = f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/statements"
        payload = {
            "statement": sql.replace(";", ""),
            "timeout": 300
        }
        async with httpx.AsyncClient(verify=False, timeout=120.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers=API_HEADERS,
                params={"requestId": request_id},
            )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"SQL API error: {response.status_code} {response.text}"}
    except Exception as e:
        logging.error("SQL execution error: %s\n%s", e, traceback.format_exc())
        return {"error": f"SQL execution error: {e}\n{traceback.format_exc()}"}

@mcp.tool(description="Query internal business data including sales performance, revenue, customer metrics, operational KPIs, and financial reports. Use this for questions about company's own business performance and internal data.")
async def fashion_analyst(query: str) -> Dict[str, Any]:
    """
    회사 내부 비즈니스 데이터 조회 도구

    아래와 같은 쿼리일 때 해당 도구를 사용합니다.
    - 매출 실적 및 수익 분석
    - 고객 행동 및 관련 지표
    - 운영 KPI 및 성과 지표
    - 재무 보고서 및 비즈니스 인사이트
    - 내부 데이터 분석 및 리포트

    Args:
        query (str)

    Returns:
        result (dict) : {
            "text": "Text2SQL 결과",
            "data": "SQL 실행 결과"
            "sql": "SQL 쿼리",
        }
    """

    analyst_result =  await call_cortex_analyst(
        query=query,
        semantic_model_file='fashion_streamlit'
    )
    logging.info(f"analyst_result: {analyst_result}")

    content = {c['type']: c for c in analyst_result['message']['content']}
    text = content.get('text', {}).get('text')
    sql = content.get('sql', {}).get('statement')
    execute_result = await execute_sql(sql)
    logging.info(f"execute_result: {execute_result}")

    return {
        "text": text,
        "data": execute_result['data'],
        "sql": sql
    }

@mcp.tool(description="Query external market data including competitor analysis, market trends, consumer reviews, pricing intelligence, and industry insights. Use this for questions about competitors, market research, and external market intelligence.")
async def market_analyst(query: str) -> Dict[str, Any]:
    """
    회사 외부 시장 데이터 조회 도구

    아래와 같은 쿼리일 때 해당 도구를 사용합니다.
    - 경쟁사 분석 및 벤치마킹
    - 시장 동향 및 업계 인사이트
    - 소비자 리뷰 및 감정 분석
    - 플랫폼 가격 정보 분석
    - 브랜드 인식 및 시장 포지셔닝
    - 업계 연구 및 시장 조사

    Args:
        query (str)

    Returns:
        result (dict) : {
            "text": "Text2SQL 결과",
            "data": "SQL 실행 결과"
            "sql": "SQL 쿼리",
        }
    """

    analyst_result =  await call_cortex_analyst(
        query=query,
        semantic_model_file='fashion_keyword'
    )
    logging.info(f"analyst_result: {analyst_result}")

    content = {c['type']: c for c in analyst_result['message']['content']}
    text = content.get('text', {}).get('text')
    sql = content.get('sql', {}).get('statement')
    execute_result = await execute_sql(sql)
    logging.info(f"execute_result: {execute_result}")

    return {
        "text": text,
        "data": execute_result['data'],
        "sql": sql
    }

if __name__ == "__main__":

    try:
        logging.info("Optimized Cortex Agent MCP 서버 시작...")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logging.info("Cortex Agent MCP 서버가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logging.error(f"Cortex Agent MCP 서버 실행 중 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)
