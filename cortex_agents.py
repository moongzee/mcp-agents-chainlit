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
SNOWFLAKE_ACCOUNT_URL = os.getenv("SNOWFLAKE_ACCOUNT_URL", default="https://a1041574869271-eland-partner.snowflakecomputing.com")
SNOWFLAKE_PAT = os.getenv("SNOWFLAKE_PAT", default="yJraWQiOiIyMTA0NTc2OTI3OTg5ODIiLCJhbGciOiJFUzI1NiJ9.eyJwIjoiMTI1NDQyNjA6MzIxMTMzMDA1MyIsImlzcyI6IlNGOjEwMzIiLCJleHAiOjE3ODE4MzA2NDZ9.zIsLtQrtoIp_WYZJyv_bU6U_KQ8Vr0T8MICvAjpgOb0wilLCwPcSAvH4wKzE4OGg8lFdEN7tR1X7KmfE56XxKg")
SEMANTIC_MODEL_PATH = os.getenv("SEMANTIC_MODEL_PATH", default="@CORTEX_ANALYST_DEMO.LLM_POC.CORTEX_STAGE/")
SEARCH_SERVICE_SCHEMA = os.getenv("SEARCH_SERVICE_SCHEMA", default="CORTEX_ANALYST_DEMO.LLM_POC")

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

async def agent_run_request(
    query: str,
    description: Optional[str] = None,
    semantic_model_file: Optional[str] = None,
    search_table: Optional[str] = None,
    search_type: Optional[str] = None,
    search_filters: Optional[Dict] = None,
    max_search_results: int = 5
) -> Dict[str, Any]:
    """
    공통 agent 호출 함수. semantic_model_file이 있으면 text_to_sql, search_table이 있으면 search로 분기.
    """
    if semantic_model_file:
        if description:
            response_instruction = description
        else:
            response_instruction = "cortex_analyst_text_to_sql"
        tools = [
            {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "Analyst1"}},
            {"tool_spec": {"type": "sql_exec", "name": "sql_execution_tool"}}
        ]
        tool_resources = {
            "Analyst1": {"semantic_model_file": semantic_model_file}
        }
    elif search_table and search_type:
        if description:
            response_instruction = description
        else:
            response_instruction = "cortex_search"
        tools = [
            {"tool_spec": {
                "type": "cortex_search",
                "name": search_type,
                "description": f"Search for {search_type} documents."
            }}
        ]
        tool_resources = {
            search_type: {
                "name": search_table,
                "max_results": max_search_results,
                "title_column": "relative_path",
                "id_column": "doc_id",
                "filter": search_filters or {}
            }
        }
    else:
        raise ValueError("Either semantic_model_file or (search_table and search_type) must be provided.")

    payload = {
        "model": "claude-4-sonnet",
        "response_instruction": response_instruction,
        "experimental": {},
        "tools": tools,
        "tool_resources": tool_resources,
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
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                params={"requestId": request_id},
            ) as resp:
                resp.raise_for_status()
                text, sql, citations = await process_sse_response(resp)

        # Execute any SQL that was generated during the search
        results = None
        if sql:
            results = await execute_sql(sql)

        return {
            "text": text,
            "sql": sql,
            "citations": citations,
            "results": results,
            "analysis_type": semantic_model_file or search_table or None,
            "success": True
        }
    except Exception as e:
        logging.error(f"Cortex Agent 호출 중 오류 발생: {e}")
        return {"error": f"Cortex Agent 호출 중 오류 발생: {e}", "success": False}

def safe_string_convert(value: Any) -> str:
    """안전하게 값을 문자열로 변환"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)

async def process_sse_response(resp: httpx.Response) -> Tuple[str, str, List[Dict]]:
    """SSE 응답을 처리하여 텍스트, SQL, citations를 추출"""
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
                                "source_id": safe_string_convert(s.get("source_id")),
                                "doc_id": safe_string_convert(s.get("doc_id")),
                                "content": safe_string_convert(s.get("content")),
                                "title": safe_string_convert(s.get("title")),
                            })
    return text, sql, citations

async def execute_sql(sql: str) -> Dict[str, Any]:
    """SQL을 실행하고 결과를 반환"""
    try:
        request_id = str(uuid.uuid4())
        sql_api_url = f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/statements"
        sql_payload = {
            "statement": sql.replace(";", ""),
            "timeout": 300
        }
        async with httpx.AsyncClient(verify=False, timeout=120.0) as client:
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

@mcp.tool(description="Classifies the user query to determine the most appropriate tool among internal fashion data, mixed internal-external analysis, business documents, or HR policies.")
async def detect_query_intent(query: str) -> Literal["fashion_analyst", "market_analyst", "business_document_search", "hr_document_search"]:
    """
    Determines the most suitable tool based on the user's query.
    - fashion_analyst: internal fashion operations data (e.g. sales, inventory, financials)
    - market_analyst: combined internal/external data (e.g. competitor prices, reviews)
    - business_document_search: reference documents such as terminology or business logic
    - hr_document_search: internal HR documents (e.g. vacation, org chart, policies)
    """

    prompt = f"""
You are a tool selection assistant. Choose the most appropriate tool based on the user's query.

Available tools:
- fashion_analyst: Use this tool for queries related to fashion product sales, top-selling items, inventory, customer visits, and financial statements.
  Queries about who made the best-selling product, sales performance, or product-related employee data belong here.
  This includes internal performance data of E-Land brands such as SPAO, New Balance, Shoopen, Roem, and MIXXO.

- keyword_analyst: For analysis involving competitor pricing, discount rates, and customer reviews, using both internal and external data.
  This tool is used for queries involving external competitors such as Musinsa, Ably, Uniqlo, and Handsome Mall, or for comparing E-Land brands to competitors.

- business_search: For referencing internal company documents such as terminology, report formats, or business logic.

- hr_search: For internal HR documents such as leave policies, benefits, organizational charts, vacations, onboarding data, and general employee management policies.
  Note: HR search is NOT for product sales, brand performance, or market analytics.

Respond with only the tool name.

Query: "{query}"
"""

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a tool selector."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    tool_name = response.choices[0].message.content.strip()
    logging.info(f"Selected tool: {tool_name}")

    return tool_name

@mcp.tool(description="Analyze internal business data including sales performance, revenue, customer metrics, operational KPIs, and financial reports. Use this for questions about company's own business performance and internal data.")
async def fashion_analyst(query: str) -> Dict[str, Any]:
    """
    Analyze internal business data including sales, revenue, customer metrics, and operational performance.

    This tool specializes in analyzing your company's internal business data such as:
    - Sales performance and revenue analysis
    - Customer behavior and metrics
    - Operational KPIs and performance indicators
    - Financial reports and business intelligence
    - Internal data analytics and reporting

    Use this tool when you need to analyze your own company's business performance data.

    Args:
        query (str): Question about internal business data, sales, revenue, or company performance

    Returns:
        dict: Contains SQL query, execution results, and business insights
    """

    description = (
            "You are a business intelligence analyst specializing in internal company data. "
            "Use the Text-to-SQL tool to analyze sales, revenue, customer, and operational data. "
            "Focus on providing actionable business insights from internal company performance data. "
            "Generate SQL queries that reveal trends, patterns, and key business metrics. "
            "Provide clear explanations of business performance and strategic recommendations."
        )
    return await agent_run_request(
        query=query,
        description=description,
        semantic_model_file=SEMANTIC_MODEL_PATH + 'fashion_streamlit.yaml'
    )

@mcp.tool(description="Analyze external market data including competitor analysis, market trends, consumer reviews, pricing intelligence, and industry insights. Use this for questions about competitors, market research, and external market intelligence.")
async def market_analyst(query: str) -> Dict[str, Any]:
    """
    Analyze external market data including competitor analysis, market trends, and consumer insights.

    This tool specializes in analyzing external market intelligence data such as:
    - Competitor analysis and benchmarking
    - Market trends and industry insights
    - Consumer reviews and sentiment analysis
    - Pricing intelligence and competitive pricing
    - Brand perception and market positioning
    - Industry research and market studies

    Use this tool when you need insights about competitors, market conditions, or external market data.

    Args:
        query (str): Question about competitors, market trends, pricing, or external market intelligence

    Returns:
        dict: Contains SQL query, execution results, and market intelligence insights
    """
    description = (
            "You are a market intelligence analyst specializing in competitor and external market data. "
            "Use the Text-to-SQL tool to analyze competitor performance, market trends, and consumer insights. "
            "Focus on providing strategic market intelligence and competitive analysis. "
            "Generate SQL queries that reveal market opportunities, competitive threats, and consumer preferences. "
            "Provide actionable insights for competitive strategy and market positioning."
        )
    return await agent_run_request(
        query=query,
        description=description,
        semantic_model_file=SEMANTIC_MODEL_PATH + 'fashion_keyword.yaml'
    )

@mcp.tool(description="Search and analyze documents, reports, policies, and knowledge base content. Use this for finding specific information, guidelines, procedures, or when you need to reference existing documentation.")
async def business_document_search(query: str, search_filters: Optional[Dict] = None, max_search_results: int = 5) -> Dict[str, Any]:
    """
     Search through documents and analyze content from reports, policies, and knowledge base.

    This tool specializes in document search and content analysis including:
    - Company policies and procedures
    - Research reports and studies
    - Guidelines and best practices
    - Historical documentation
    - Knowledge base articles
    - Reference materials and manuals

    Use this tool when you need to find specific information from existing documents or when the answer requires referencing established procedures, policies, or documented knowledge.

    Args:
        query (str): Question about specific information, procedures, policies, or documentation
        max_results (int): Maximum number of search results to return (default: 5)

    Returns:
        dict: Contains search results, relevant document excerpts, and any additional SQL analysis
    """
    description = (
            "You are a knowledge management specialist focused on document search and analysis. "
            "Use the Search tool to find relevant documents, policies, procedures, and reference materials. "
            "Extract key information and provide comprehensive answers based on documented knowledge. "
            "If the search results suggest data analysis is needed, use SQL execution appropriately. "
            "Provide clear citations and references to source documents for all information provided."
        )
    return await agent_run_request(
        query=query,
        description=description,
        search_table='CORTEX_ANALYST_DEMO.LLM_POC.DOCS_MEETING',
        search_type='Business Search',
        search_filters=search_filters,
        max_search_results=max_search_results
    )

@mcp.tool(description="Retrieves HR-related documents such as policies, benefits, onboarding, and organizational charts.")
async def hr_document_search(query: str, search_filters: Optional[Dict] = None, max_search_results: int = 5) -> Dict[str, Any]:
    """
    Retrieves internal HR documents such as leave policies, onboarding materials, and organizational information.

    This tool specializes in HR document search and policy referencing:
    - Vacation, leave, and remote work policies
    - Employee benefits and welfare information
    - Onboarding guides and welcome kits
    - Organization charts and internal team structures
    - HR handbook and FAQs

    Use this tool when the question is about HR policies, onboarding, or internal employee guidance.

    Args:
        query (str): HR-related question such as policy, benefits, or internal structure.
        department (str, optional): Filter by department or role-specific policy.
        limit (int, optional): Max number of HR documents to return (default: 3).

    Returns:
        dict: Relevant HR policies, structured content excerpts, and document links or IDs.
    """
    return await agent_run_request(
        query=query,
        search_table='CORTEX_ANALYST_DEMO.LLM_POC.HR_DOCS_MEETING',
        search_type='HR Search',
        search_filters=search_filters,
        max_search_results=max_search_results
    )

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