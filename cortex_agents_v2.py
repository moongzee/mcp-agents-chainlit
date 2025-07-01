from typing import Any, Dict, Tuple, List, Optional
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
SEMANTIC_MODEL_FILE = os.getenv("SEMANTIC_MODEL_FILE", default="@CORTEX_ANALYST_DEMO.LLM_POC.CORTEX_STAGE/fashion_streamlit.yaml")
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


async def call_cortex_agent(
    query: str,
    use_analyst: bool = True,
    use_search: bool = True,
    max_search_results: int = 15,
    search_filters: Optional[Dict] = None,
    custom_instruction: Optional[str] = None
) -> Tuple[str, str, List[Dict]]:
    """
    통합된 Cortex Agent 호출 함수

    Args:
        query: 사용자 질문
        use_analyst: Analyst 도구 사용 여부
        use_search: Search 도구 사용 여부
        max_search_results: 최대 검색 결과 수
        search_filters: 검색 필터
        custom_instruction: 커스텀 응답 지시사항
    """

    # 기본 응답 지시사항
    if custom_instruction is None:
        if use_analyst and use_search:
            response_instruction = (
                "You are a helpful data analytics agent. "
                "1. Use the Analyst1 tool to convert user questions into SQL when data analysis is needed. "
                "2. Use the Search1 tool to find relevant documents when document search is needed. "
                "3. Combine results appropriately and provide structured answers."
            )
        elif use_analyst:
            response_instruction = (
                "You are a data analyst. Use the Analyst1 tool to convert questions into SQL and analyze data."
            )
        elif use_search:
            response_instruction = (
                "You are a document search assistant. Use the Search1 tool to find relevant information."
            )
        else:
            response_instruction = "Answer the question based on available context."
    else:
        response_instruction = custom_instruction

    # 도구 구성
    tools = []
    tool_resources = {}

    if use_analyst:
        tools.append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "Analyst1"}})
        tool_resources["Analyst1"] = {"semantic_model_file": SEMANTIC_MODEL_FILE}

    if use_search:
        tools.append({"tool_spec": {
            "type": "cortex_search",
            "name": "Business Search",
            "description": "Search for business-related documents such as sales reports, performance records, financial data, and business analysis."}})
        tools.append({"tool_spec": {
            "type": "cortex_search",
            "name": "HR Search",
            "description": "Search for internal HR documents, including employee policies, benefits, organizational charts, and HR guidelines."}})

        # 기본 검색 필터
        default_filters = {} # {"@eq": {"language": "Korean"}}
        if search_filters:
            default_filters = search_filters

        tool_resources["Business Search"] = {
            "name": 'CORTEX_ANALYST_DEMO.LLM_POC.DOCS_MEETING',
            "max_results": max_search_results,
            "title_column": "relative_path",
            "id_column": "doc_id",
            "filter": default_filters
        }
        tool_resources["HR Search"] = {
            "name": 'CORTEX_ANALYST_DEMO.LLM_POC.HR_DOCS_MEETING',
            "max_results": max_search_results,
            "title_column": "relative_path",
            "id_column": "doc_id",
            "filter": default_filters
        }

    if use_analyst:
        tools.append({"tool_spec": {"type": "sql_exec", "name": "sql_execution_tool"}})

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

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            params={"requestId": request_id},
        ) as resp:
            resp.raise_for_status()
            return await process_sse_response(resp)


def group_citations_by_document(citations: List[Dict]) -> Dict[str, Dict]:
    """Citations를 문서별로 그룹화"""
    documents_by_path = {}
    for citation in citations:
        relative_path = citation.get("source_id", "")
        if not relative_path:
            continue

        if relative_path not in documents_by_path:
            documents_by_path[relative_path] = {
                "title": citation.get("title", ""),
                "chunks": [],
                "source": relative_path
            }

        content = citation.get("content", "")
        if content:
            documents_by_path[relative_path]["chunks"].append(content)

    return documents_by_path


async def generate_styled_report(query: str, data_results: Dict, citations: List[Dict]) -> str:
    """문서 템플릿을 참고하여 스타일이 적용된 보고서 생성"""

    # 문서별로 청크들을 그룹화
    documents_by_path = group_citations_by_document(citations)

    # 각 문서의 전체 내용을 재구성하여 템플릿 스타일 분석
    document_styles = []
    for path, doc_info in documents_by_path.items():
        combined_content = " ".join(doc_info["chunks"])[:1000]
        document_styles.append({
            "title": doc_info["title"],
            "content_sample": combined_content,
            "source": path,
            "chunk_count": len(doc_info["chunks"])
        })

    # 보고서 생성을 위한 프롬프트 구성
    template_instruction = f"""
    다음 {len(document_styles)}개 문서들의 형태와 어투를 참고하여 데이터 분석 보고서를 작성해주세요:

    참고 문서 스타일:
    """

    for i, style in enumerate(document_styles):
        template_instruction += f"""
        문서 {i+1}: {style['title']} (청크 수: {style['chunk_count']})
        내용 샘플: {style['content_sample']}
        출처: {style['source']}
        ---
        """

    template_instruction += f"""

    위 문서들의 형식, 어투, 구조를 참고하여 다음 데이터 분석 결과를 바탕으로 보고서를 작성해주세요:

    사용자 질문: {query}
    데이터 분석 결과: {json.dumps(data_results, ensure_ascii=False, indent=2)}

    보고서 작성 시 다음 사항을 고려해주세요:
    1. 참고 문서의 어투와 형식을 따라주세요
    2. 데이터에서 도출된 핵심 인사이트를 강조해주세요
    3. 구체적인 수치와 함께 설명해주세요
    4. 결론과 제언을 포함해주세요
    5. 각 문서에서 추출한 스타일 요소들을 종합적으로 반영해주세요
    """

    # Cortex Complete API를 사용하여 보고서 생성
    payload = {
        "model": "claude-4-sonnet",
        "messages": [
            {"role": "user", "content": template_instruction}
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        request_id = str(uuid.uuid4())
        url = f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/cortex/complete"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers=API_HEADERS,
                params={"requestId": request_id},
            )

        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "보고서 생성에 실패했습니다.")
        else:
            return f"보고서 생성 API 오류: {response.status_code} {response.text}"

    except Exception as e:
        logging.error("Report generation error: %s\n%s", e, traceback.format_exc())
        return f"보고서 생성 중 오류 발생: {e}"


def detect_query_intent(query: str) -> str:
    """사용자 질문의 의도를 파악 (DEBUG)"""
    query_lower = safe_string_convert(query).lower()
    logging.debug(f"[detect_query_intent] query_lower: {query_lower}")

    # 보고서 목록 요청 키워드들
    report_list_keywords = [
        "보고서목록", "보고서 목록", "사용가능한 보고서", "available reports",
        "문서목록", "문서 목록", "어떤 문서", "어떤 보고서", "뭐가 있어", "뭐가있어",
        "목록", "리스트", "list", "documents", "reports", "파일목록", "파일 목록"
    ]

    if any(keyword in query_lower for keyword in report_list_keywords):
        logging.debug("[detect_query_intent] Detected: report_list")
        return "report_list"

    # 데이터 분석 키워드들
    analysis_keywords = [
        "분석", "조회", "데이터", "통계", "집계", "합계", "평균", "최대", "최소",
        "sql", "select", "where", "group by", "분석해줘", "보여줘"
    ]

    # 문서 검색 키워드들
    search_keywords = [
        "찾아줘", "알려줘","검색", "문서", "내용", "정보", "자료", "참고", "방법", "규정"
    ]

    business_keywords = [
        "매출", "영업", "실적", "성과", "비즈니스", "사업", "report", "business", "재무", "finance", "판매", "고객", "시장", "분기", "연간"
    ]

    # 사내 지식(HR) 관련 키워드
    hr_keywords = [
        "인사", "휴가", "복지", "채용", "인원", "연차", "근무", "급여", "인사제도", "hr", "인사팀", "조직도", "사내", "직원", "경조"
    ]

    if any(keyword in query_lower for keyword in analysis_keywords):
        logging.debug("[detect_query_intent] Detected: data_analysis")
        if not any(keyword in query_lower for keyword in hr_keywords):
            return "data_analysis"


    if any(keyword in query_lower for keyword in search_keywords):
        # 비즈니스 보고서 관련 키워드

        if any(keyword in query_lower for keyword in business_keywords):
            logging.debug("[detect_query_intent] Detected: business_search")
            return "business_search"
        elif any(keyword in query_lower for keyword in hr_keywords):
            logging.debug("[detect_query_intent] Detected: hr_search")
            return "hr_search"

        logging.debug("[detect_query_intent] Detected: document_search")
        return "document_search" # 통합 검색

    # 기본값은 통합 분석
    logging.debug("[detect_query_intent] Detected: integrated_analysis (default)")
    return "integrated_analysis"


@mcp.tool(description="Unified Cortex Agent that handles data analysis, document search, and report generation based on query intent.")
async def cortex_unified_agent(
    query: str,
    force_mode: Optional[str] = None,
    include_styled_report: bool = False,
    max_search_results: int = 15
) -> Dict[str, Any]:
    """
    통합된 Cortex Agent 도구 - 질문 의도에 따라 적절한 기능을 자동 선택

    Args:
        query: 사용자 질문
        force_mode: 강제 모드 ("data_analysis", "document_search", "report_list", "integrated_analysis")
        include_styled_report: 스타일이 적용된 보고서 생성 여부
        max_search_results: 최대 검색 결과 수

    Returns:
        dict: 통합된 분석 결과
    """

    # 질문 의도 파악
    intent = force_mode if force_mode else detect_query_intent(query)

    try:
        if intent == "report_list":
            # 보고서 목록 반환
            text, sql, citations = await call_cortex_agent(
                "시스템에 있는 모든 문서와 보고서 목록을 보여주세요",
                use_analyst=False,
                use_search=True,
                max_search_results=50
            )

            # 중복 제거된 문서 목록 생성
            unique_documents = {}
            for citation in citations:
                relative_path = citation.get("source_id", "")
                if relative_path and relative_path not in unique_documents:
                    content = citation.get("content", "")
                    unique_documents[relative_path] = {
                        "relative_path": relative_path,
                        "title": citation.get("title", ""),
                        "doc_id": citation.get("doc_id", ""),
                        "sample_content": content[:200] if content else ""
                    }

            # 문서 타입별 분류
            document_types = {}
            for path, doc_info in unique_documents.items():
                if "." in path:
                    ext = path.split(".")[-1].split("_")[0]
                else:
                    ext = "unknown"

                if ext not in document_types:
                    document_types[ext] = []
                document_types[ext].append(doc_info)

            # 사용자 친화적인 응답 생성
            if unique_documents:
                response_text = f"📋 **사용 가능한 보고서 목록** (총 {len(unique_documents)}개)\n\n"

                for doc_type, docs in document_types.items():
                    response_text += f"**{doc_type.upper()} 파일 ({len(docs)}개):**\n"
                    for doc in docs:
                        response_text += f"• {doc['relative_path']}\n"
                        if doc['sample_content']:
                            response_text += f"  └ 내용 미리보기: {doc['sample_content']}...\n"
                    response_text += "\n"

                response_text += "위 보고서들을 활용하여 데이터 분석이나 문서 검색을 요청하실 수 있습니다."
            else:
                response_text = "❌ 현재 사용 가능한 보고서를 찾을 수 없습니다."

            return {
                "intent": intent,
                "response": response_text,
                "unique_documents": list(unique_documents.values()),
                "document_types": document_types,
                "total_documents": len(unique_documents)
            }

        elif intent == "data_analysis":
            logging.debug("[cortex_unified_agent] data_analysis branch")
            # 데이터 분석 수행
            text, sql, citations = await call_cortex_agent(
                query,
                use_analyst=True,
                use_search=False,
                max_search_results=max_search_results
            )

            # SQL 실행
            results = await execute_sql(sql) if sql else None

            response_data = {
                "intent": intent,
                "response": text,
                "sql": sql,
                "results": results,
                "citations": citations
            }

            # 스타일이 적용된 보고서 생성 (요청 시)
            if include_styled_report and citations:
                styled_report = await generate_styled_report(query, results or {}, citations)
                response_data["styled_report"] = styled_report

            logging.debug(f"[cortex_unified_agent] data_analysis response: {response_data}")
            return response_data

        elif 'search' in intent :
            logging.debug(f"[cortex_unified_agent] search branch: {intent}")
            if intent == "hr_search":
                search_name = "HR Search"
            else:
                search_name = "Business Search"

            # 문서 검색 수행
            text, sql, citations = await call_cortex_agent(
                query,
                use_analyst=False,
                use_search=True,
                max_search_results=max_search_results
            )

            logging.debug(f"[cortex_unified_agent] search response: text: {text[:200]}, citations: {len(citations)}")
            return {
                "intent": intent,
                "response": text,
                "citations": citations,
                "found_documents": len(citations)
            }


        else:  # integrated_analysis
            # 통합 분석 (데이터 + 문서)
            text, sql, citations = await call_cortex_agent(
                query,
                use_analyst=True,
                use_search=True,
                max_search_results=max_search_results
            )

            # SQL 실행
            results = await execute_sql(sql) if sql else None

            response_data = {
                "intent": intent,
                "response": text,
                "sql": sql,
                "results": results,
                "citations": citations
            }

            # 스타일이 적용된 보고서 생성 (요청 시)
            if include_styled_report and citations:
                styled_report = await generate_styled_report(query, results or {}, citations)
                response_data["styled_report"] = styled_report

            return response_data

    except Exception as e:
        logging.error(f"Error in cortex_unified_agent: {e}")
        return {
            "intent": intent,
            "error": str(e),
            "response": f"처리 중 오류가 발생했습니다: {e}"
        }


@mcp.tool(description="Generate comprehensive analysis reports with document template styling.")
async def generate_comprehensive_report(
    query: str,
    template_documents: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    포괄적인 분석 보고서 생성 (문서 템플릿 스타일 적용)

    Args:
        query: 분석 질문
        template_documents: 템플릿으로 사용할 문서 경로 목록

    Returns:
        dict: 포괄적인 분석 보고서
    """

    # 1단계: 통합 분석 수행
    analysis_result = await cortex_unified_agent(
        query,
        force_mode="integrated_analysis",
        include_styled_report=False,
        max_search_results=20
    )

    # 2단계: 템플릿 문서 추가 검색 (지정된 경우)
    template_citations = []
    if template_documents:
        for doc_path in template_documents:
            try:
                text, sql, citations = await call_cortex_agent(
                    f"문서 {doc_path}의 내용을 찾아주세요",
                    use_analyst=False,
                    use_search=True,
                    max_search_results=10,
                    search_filters={
                        "@and": [
                            {"@eq": {"language": "Korean"}},
                            {"@eq": {"relative_path": doc_path}}
                        ]
                    }
                )
                template_citations.extend(citations)
            except Exception as e:
                logging.warning(f"템플릿 문서 {doc_path} 검색 실패: {e}")

    # 3단계: 스타일이 적용된 보고서 생성
    all_citations = analysis_result.get("citations", []) + template_citations
    styled_report = ""

    if all_citations:
        styled_report = await generate_styled_report(
            query,
            analysis_result.get("results", {}),
            all_citations
        )

    return {
        "original_analysis": analysis_result,
        "styled_report": styled_report,
        "template_documents_used": template_documents or [],
        "total_template_citations": len(template_citations),
        "has_styled_report": bool(styled_report)
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