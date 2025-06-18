from typing import Any, Dict, Tuple, List
import httpx
from mcp.server.fastmcp import FastMCP
import os
import json
import uuid
from dotenv import load_dotenv, find_dotenv
import asyncio
import requests
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
SEMANTIC_MODEL_FILE = os.getenv("SEMANTIC_MODEL_FILE")
CORTEX_SEARCH_SERVICE = os.getenv("CORTEX_SEARCH_SERVICE")
SNOWFLAKE_ACCOUNT_URL = os.getenv("SNOWFLAKE_ACCOUNT_URL")
SNOWFLAKE_PAT = os.getenv("SNOWFLAKE_PAT")

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


@mcp.tool(description="Run a Snowflake Cortex Agent to analyze a user query, generate SQL, and return query results, including access to historical weather data.")
async def run_cortex_agents(query: str) -> Dict[str, Any]:
    """
    Run a Snowflake Cortex Agent to analyze a user query, generate SQL, and return query results.

    This tool uses Snowflake Cortex's LLM-based agent to convert a natural language question
    into a SQL query using a semantic model file. It executes the generated SQL on Snowflake
    and returns the result along with the SQL and any citations found.

    Args:
        query (str): The user's natural language question to analyze.

    Returns:
        dict: A dictionary containing:
            - text (str): The natural language response from the agent
            - sql (str): The generated SQL query
            - citations (List[dict]): List of document sources cited
            - results (dict): Raw execution result from the Snowflake SQL API
    """
    
    payload = {
        "model": "claude-3-5-sonnet",
        "response_instruction": "You are a helpful AI assistant.",
        "experimental": {},
        "tools": [
            {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "Analyst1"}},
            {"tool_spec": {"type": "cortex_search",            "name": "Search1"}},
            {"tool_spec": {"type": "sql_exec", "name": "sql_execution_tool"}},
        ],
        "tool_resources": {
            "Analyst1": {"semantic_model_file": SEMANTIC_MODEL_FILE},
            "Search1": {
                "name": CORTEX_SEARCH_SERVICE,
                "max_results": 10,
                "title_column": "relative_path",
                "id_column": "doc_id",
                "filter": {"@eq": {"language": "Korean"}}
                }
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

    results = await execute_sql(sql) if sql else None

    return {
        "text": text,
        "citations": citations,
        "sql": sql,
        "results": results,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")  # Claude는 stdio 기반 JSON-RPC 만 허용함

