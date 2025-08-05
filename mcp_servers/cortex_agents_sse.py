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
# from fastapi import APIRouter
from fastapi.responses import JSONResponse
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

# Initialize FastMCP server with metadata
mcp = FastMCP(
    "cortex_agent",
    metadata={
        "name": "Cortex MCP Server",
        "description": "FASTMCP 기반 내부/외부 데이터 분석용 MCP 서버",
        "version": "1.0.0"
    },
    host="0.0.0.0",
    port=3001,
    debug=True
)


# Constants
SNOWFLAKE_ACCOUNT_URL = os.getenv("SNOWFLAKE_ACCOUNT_URL")
SNOWFLAKE_PAT = os.getenv("SNOWFLAKE_PAT")
SEMANTIC_MODEL_PATH = os.getenv("SEMANTIC_MODEL_PATH", "@CORTEX_ANALYST_DEMO.LLM_POC.CORTEX_STAGE/")

if not SNOWFLAKE_PAT:
    raise RuntimeError("Set SNOWFLAKE_PAT environment variable")
if not SNOWFLAKE_ACCOUNT_URL:
    raise RuntimeError("Set SNOWFLAKE_ACCOUNT_URL environment variable")

API_HEADERS = {
    "Authorization": f"Bearer {SNOWFLAKE_PAT}",
    "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    "Content-Type": "application/json",
}


def safe_string_convert(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


async def call_cortex_analyst(query: str, semantic_model_file: Optional[str] = None):
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


@mcp.tool(description="Query internal business data including sales performance, revenue, customer metrics, operational KPIs, and financial reports.")
async def fashion_analyst(query: str):
    yield "💡 내부 분석을 시작합니다..."
    await asyncio.sleep(0.3)
    analyst_result = await call_cortex_analyst(query=query, semantic_model_file='fashion_streamlit')
    content = {c['type']: c for c in analyst_result['message']['content']}
    text = content.get('text', {}).get('text')
    sql = content.get('sql', {}).get('statement')
    yield f"📄 변환된 SQL: {sql}"
    await asyncio.sleep(0.3)
    execute_result = await execute_sql(sql)
    yield f"✅ 결과 반환 완료"
    yield json.dumps({"text": text, "data": execute_result['data'], "sql": sql})


@mcp.tool(description="Query external market data including competitor analysis, market trends, consumer reviews, pricing intelligence, and industry insights.")
async def market_analyst(query: str):
    yield "🌍 외부 시장 분석을 시작합니다..."
    await asyncio.sleep(0.3)
    analyst_result = await call_cortex_analyst(query=query, semantic_model_file='fashion_keyword')
    content = {c['type']: c for c in analyst_result['message']['content']}
    text = content.get('text', {}).get('text')
    sql = content.get('sql', {}).get('statement')
    yield f"📄 변환된 SQL: {sql}"
    await asyncio.sleep(0.3)
    execute_result = await execute_sql(sql)
    yield f"✅ 결과 반환 완료"
    yield json.dumps({"text": text, "data": execute_result['data'], "sql": sql})

if __name__ == "__main__":
    try:
        logging.info("Cortex Agent MCP 서버 시작 (SSE 모드)...")
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        logging.info("서버 종료됨")
    except Exception as e:
        logging.error(f"서버 실행 중 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)
