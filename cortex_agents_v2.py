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
                                "content": s.get("content", ""),  # 검색된 내용도 포함
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


async def search_cortex_documents(query: str, max_results: int = 10) -> Tuple[str, List[Dict]]:
    """
    Cortex Search를 사용하여 문서 검색
    
    Args:
        query (str): 검색할 쿼리
        max_results (int): 최대 결과 수
        
    Returns:
        Tuple[str, List[Dict]]: (응답 텍스트, 검색 결과 리스트)
    """
    payload = {
        "model": "claude-4-sonnet",
        "response_instruction": (
            f"Search for documents related to '{query}'. "
            "Return the search results with document titles, content summaries, and relevance information. "
            "Focus on finding report templates, formats, and document structures."
        ),
        "experimental": {},
        "tools": [
            {"tool_spec": {"type": "cortex_search", "name": "Search1"}},
        ],
        "tool_resources": {
            "Search1": {
                "name": CORTEX_SEARCH_SERVICE,
                "max_results": max_results,
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

    return text, citations


@mcp.tool(description="Get available report format templates by searching the document database")
async def get_report_formats(search_query: str = "보고서 템플릿 형식") -> Dict[str, Any]:
    """
    Cortex Search를 사용하여 사용 가능한 보고서 형식 템플릿을 검색합니다.
    
    Args:
        search_query (str): 검색할 쿼리 (기본값: "보고서 템플릿 형식")
        
    Returns:
        dict: 검색된 보고서 형식들과 설명
    """
    
    try:
        # 보고서 템플릿 관련 문서들 검색
        search_text, search_results = await search_cortex_documents(search_query, max_results=3)
        
        # 검색 결과를 구조화
        available_formats = []
        
        for i, result in enumerate(search_results):
            doc_id = result.get("doc_id", f"doc_{i}")
            source_id = result.get("source_id", "")
            content = result.get("content", "")
            
            # 문서 제목에서 보고서 형식 추출 시도
            if source_id:
                format_name = source_id.split("/")[-1] if "/" in source_id else source_id
                format_name = format_name.replace(".pdf", "").replace(".docx", "").replace("_", " ")
            else:
                format_name = f"보고서 형식 {i+1}"
            
            # 내용에서 핵심 설명 추출 (처음 200자)
            description = content[:200] + "..." if len(content) > 200 else content
            
            # 실제 문서 항목에 따른 보고서 형식 분류
            format_type = "기본"
            content_lower = content.lower()
            source_lower = source_id.lower() if source_id else ""
            
            # 문서 경로/제목과 내용을 모두 고려하여 분류
            if any(keyword in content_lower or keyword in source_lower for keyword in ["용어사전", "glossary", "dictionary", "용어", "정의"]):
                format_type = "비즈니스 용어사전"
            elif any(keyword in content_lower or keyword in source_lower for keyword in ["내부보고", "내부 보고", "internal", "팀보고", "부서보고"]):
                format_type = "내부보고자료"
            elif any(keyword in content_lower or keyword in source_lower for keyword in ["트렌드", "trend", "시장동향", "업계동향", "전망"]):
                format_type = "트렌드 자료"
            elif any(keyword in content_lower or keyword in source_lower for keyword in ["경영자", "컨설팅", "consulting", "전략", "strategy", "임원"]):
                format_type = "경영자 컨설팅 자료"
            elif any(keyword in content_lower or keyword in source_lower for keyword in ["상부보고", "상부 보고", "경영진", "ceo", "임원보고", "executive"]):
                format_type = "상부보고자료"
            # 추가적인 세부 분류 (기존 로직 유지)
            elif any(keyword in content_lower for keyword in ["ebg", "피드백", "우선순위", "문제점"]):
                format_type = "EBG 피드백"
            elif any(keyword in content_lower for keyword in ["정량", "수치", "지표", "kpi", "성과측정"]):
                format_type = "정량분석"
            elif any(keyword in content_lower for keyword in ["종합", "현황", "전체", "overview"]):
                format_type = "종합현황"
            
            available_formats.append({
                "id": doc_id,
                "name": format_name,
                "type": format_type,
                "description": description,
                "source_document": source_id,
                "relevance_score": len([k for k in ["보고서", "템플릿", "형식"] if k in content.lower()])
            })
        
        # 관련성 점수로 정렬
        available_formats.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # 상위 8개만 선택 (너무 많으면 혼란)
        top_formats = available_formats[:8]
        
        return {
            "success": True,
            "search_query": search_query,
            "total_found": len(search_results),
            "available_formats": top_formats,
            "search_summary": search_text[:500] + "..." if len(search_text) > 500 else search_text,
            "raw_search_results": search_results  # 원본 검색 결과도 포함
        }
        
    except Exception as e:
        logging.error(f"보고서 형식 검색 중 오류: {e}")
        traceback.print_exc()
        
        # 오류 시 실제 문서 타입 기반 기본 형식 반환
        return {
            "success": False,
            "error": str(e),
            "available_formats": [
                {
                    "id": "default_internal",
                    "name": "내부보고자료",
                    "type": "내부보고자료",
                    "description": "팀/부서 내부용 보고서 - 실무진 대상의 상세 분석 자료",
                    "source_document": "기본 템플릿"
                },
                {
                    "id": "default_executive",
                    "name": "상부보고자료", 
                    "type": "상부보고자료",
                    "description": "경영진/임원진 대상 보고서 - 핵심 메트릭스와 전략적 시사점",
                    "source_document": "기본 템플릿"
                },
                {
                    "id": "default_trend",
                    "name": "트렌드 자료",
                    "type": "트렌드 자료", 
                    "description": "시장 동향 및 업계 트렌드 분석 자료",
                    "source_document": "기본 템플릿"
                },
                {
                    "id": "default_consulting",
                    "name": "경영자 컨설팅 자료",
                    "type": "경영자 컨설팅 자료",
                    "description": "전략적 관점의 비즈니스 인사이트 및 액션 플랜",
                    "source_document": "기본 템플릿"
                },
                {
                    "id": "default_glossary",
                    "name": "비즈니스 용어사전",
                    "type": "비즈니스 용어사전",
                    "description": "전문 용어 정의 및 비즈니스 컨텍스트 설명",
                    "source_document": "기본 템플릿"
                }
            ]
        }


@mcp.tool(description="Run Cortex analysis with specific report format guidance from searched templates")
async def run_formatted_analysis(query: str, report_format_id: str = "", format_type: str = "기본") -> Dict[str, Any]:
    """
    특정 보고서 형식에 맞춰 Cortex 분석을 실행합니다.
    
    Args:
        query (str): 분석할 사용자 질문
        report_format_id (str): get_report_formats에서 얻은 보고서 형식 ID
        format_type (str): 보고서 형식 타입 (EBG 피드백, 정량분석, 종합현황, 기본)
        
    Returns:
        dict: 지정된 형식에 맞춘 분석 결과
    """
    
    # 실제 문서 타입에 따른 분석 지시사항 + 환각 방지 규칙
    format_instructions = {
        "비즈니스 용어사전": (
            "임베딩된 문서들에서 비즈니스 용어를 추출하여 용어사전 형식으로 분석하세요. "
            "🔥 중요: 실제 용어사전 문서가 없으므로, Search1 도구로 패션/경영/트렌드 관련 문서들을 검색하여 "
            "그 안에서 전문 용어들을 찾아내고 정의하세요.\n"
            "1) **용어 검색**: '패션', '경영', '트렌드', '컨설팅' 등으로 기존 문서 검색\n"
            "2) **용어 추출**: 검색된 문서에서 전문 용어들 (예: MBWA, 직접체험관매, 정판율 등) 식별\n"
            "3) **맥락 정의**: 문서 내용을 바탕으로 해당 용어들의 정의와 사용 맥락 파악\n"
            "4) **데이터 연계**: 용어와 관련된 실제 데이터가 있으면 Analyst1으로 조회\n"
            "5) **용어사전 형식**: 추출한 용어들을 체계적인 용어사전 보고서 형식으로 정리\n"
            "기존 임베딩 문서들을 적극 활용하여 해당 조직/업계의 실제 용어들을 정의하세요.\n\n"
            "⚠️ 환각 방지 규칙: 임의로 용어를 만들지 말고, 반드시 검색된 문서에서 실제 사용된 용어들만 정의하세요."
        ),
        "내부보고자료": (
            "내부보고자료 형식으로 분석하세요. "
            "팀/부서 내부용 보고서 스타일로 작성하며, "
            "실무진이 이해하기 쉬운 구체적인 데이터와 실행 가능한 인사이트를 제공하세요. "
            "상세한 프로세스와 근거를 포함하세요.\n\n"
            "⚠️ 환각 방지 규칙: 보고일자, 작성부서, 보고대상 등은 실제 데이터나 검색된 문서에 있는 경우에만 사용하세요."
        ),
        "트렌드 자료": (
            "트렌드 분석 자료 형식으로 분석하세요. "
            "시장 동향, 업계 트렌드, 미래 전망에 중점을 두고, "
            "시계열 분석과 변화 패턴을 포함하여 작성하세요. "
            "경쟁사 비교와 시장 포지셔닝 관점을 추가하세요.\n\n"
            "⚠️ 환각 방지 규칙: 추측이나 가정이 아닌 실제 데이터와 검색된 문서 내용만 사용하세요."
        ),
        "경영자 컨설팅 자료": (
            "경영자 컨설팅 자료 형식으로 분석하세요. "
            "전략적 관점에서 비즈니스 인사이트를 제공하고, "
            "의사결정 지원을 위한 다각도 분석과 리스크 평가, "
            "구체적인 액션 플랜과 ROI 예측을 포함하세요.\n\n"
            "⚠️ 환각 방지 규칙: 가상의 전략이나 추정치가 아닌 실제 데이터 기반의 분석만 제공하세요."
        ),
        "상부보고자료": (
            "상부보고자료 형식으로 분석하세요. "
            "경영진/임원진 대상 보고서 스타일로 작성하며, "
            "핵심 메트릭스와 전략적 시사점에 집중하세요. "
            "간결하면서도 임팩트 있는 요약과 명확한 결론을 제시하세요.\n\n"
            "⚠️ 환각 방지 규칙: 실제 승인자나 보고 대상이 명시된 경우에만 해당 정보를 포함하세요."
        ),
        "EBG 피드백": (
            "EBG 피드백 보고서 형식으로 분석하세요. "
            "우선순위별 문제점을 식별하고, 각 우선순위마다 다음을 포함하세요: "
            "1) 현재 성과 vs 목표 (구체적 수치) "
            "2) 근본 원인 분석 (버그) "
            "3) 구체적 해결방안 (대안) "
            "문제해결 중심의 접근법을 사용하세요.\n\n"
            "⚠️ 환각 방지 규칙: 실제 데이터에 없는 목표치나 성과 수치를 임의로 생성하지 마세요."
        ),
        "정량분석": (
            "정량분석 보고서 형식으로 분석하세요. "
            "정확한 지표, 성장률, 통계적 인사이트에 집중하세요. "
            "카테고리별, 기간별 상세 분석과 비교 분석을 포함하세요. "
            "구체적인 백분율, 비율, 트렌드 계산을 제공하세요.\n\n"
            "⚠️ 환각 방지 규칙: 모든 수치와 계산은 실제 조회된 데이터에 기반해야 하며, 추정치는 명확히 표시하세요."
        ),
        "종합현황": (
            "종합현황 보고서 형식으로 분석하세요. "
            "모든 핵심 영역의 전체적인 성과 관점을 제공하세요. "
            "부서별 분석, 트렌드 분석, 예측 요소, "
            "비즈니스 계획에 대한 전략적 시사점을 포함하세요.\n\n"
            "⚠️ 환각 방지 규칙: 부서명, 조직 구조, 비즈니스 계획은 실제 검색된 문서에 있는 내용만 사용하세요."
        ),
        "기본": (
            "간결하고 전문적인 요약과 핵심 인사이트, 실행 가능한 결론을 제공하세요.\n\n"
            "⚠️ 환각 방지 규칙: 실제 데이터와 검색 결과에 기반한 내용만 제공하세요."
        )
    }
    
    # 🔧 개선된 템플릿 검색 - 다양한 검색 시도
    template_guidance = ""
    template_documents = []
    
    # 여러 검색 쿼리로 관련 문서 찾기
    search_queries = []
    
    if report_format_id:
        search_queries.append(f"보고서 템플릿 {report_format_id}")
        search_queries.append(f"{report_format_id} 형식")
    
    if format_type != "기본":
        # 형식별 맞춤 검색 키워드
        if format_type == "비즈니스 용어사전":
            # 🔍 질문에서 핵심 용어 추출하여 직접 검색
            import re
            # 질문에서 비즈니스 용어 가능성이 있는 단어들 추출
            potential_terms = re.findall(r'[가-힣]+(?:율|률|지표|KPI|성과|매출|수익|브랜드|정판|할인|마진)', query, re.IGNORECASE)
            
            for term in potential_terms:
                search_queries.append(term)  # 용어 직접 검색
                search_queries.append(f"{term} 정의")
                search_queries.append(f"{term} 의미")
            
            # 기본 용어사전 검색도 추가
            search_queries.extend([
                "용어사전",
                "비즈니스 용어",
                "용어 정의"
            ])
        else:
            search_queries.extend([
                f"{format_type} 템플릿",
                f"{format_type} 양식", 
                f"{format_type} 구조",
                f"{format_type} 예시",
                format_type
            ])
    
    # 각 검색 쿼리 시도
    for search_query in search_queries:
        try:
            template_text, template_results = await search_cortex_documents(search_query, max_results=5)
            if template_results:
                template_documents.extend(template_results)
                logging.info(f"템플릿 검색 성공: '{search_query}' -> {len(template_results)}개 문서")
                break  # 첫 번째 성공한 검색으로 충분하면 중단
        except Exception as e:
            logging.warning(f"템플릿 검색 '{search_query}' 실패: {e}")
            continue
    
    # 검색된 템플릿 문서들을 guidance로 구성
    if template_documents:
        # 중복 제거 및 관련성 높은 문서 우선
        unique_docs = {}
        for doc in template_documents:
            doc_id = doc.get("doc_id", "")
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc
        
        # 템플릿 내용 추출 및 구조화
        template_parts = []
        for doc in list(unique_docs.values())[:3]:  # 상위 3개 문서만 사용
            content = doc.get("content", "")
            source = doc.get("source_id", "")
            
            if content:
                # 내용을 적절한 길이로 요약
                content_summary = content[:500] + "..." if len(content) > 500 else content
                template_parts.append(f"📄 {source}: {content_summary}")
        
        if template_parts:
            template_guidance = "\n\n".join(template_parts)
            logging.info(f"템플릿 가이던스 생성 완료: {len(template_parts)}개 문서 참조")
    else:
        logging.warning(f"'{format_type}' 형식의 템플릿 문서를 찾을 수 없음")
    
    # 🔧 더 구체적인 지시사항 구성 (형식별 특화)
    enhanced_instruction = format_instructions.get(format_type, format_instructions["기본"])
    
    # 비즈니스 용어사전 형식에 대한 특별 처리
    if format_type == "비즈니스 용어사전":
        enhanced_instruction += """

🎯 비즈니스 용어사전 형식 작성 가이드:
- 🔍 STEP 1: 질문에서 핵심 용어를 추출하여 Search1으로 반드시 검색
- 📚 STEP 2: 검색된 용어사전 문서의 정의를 직접 인용하여 사용
- 🔗 STEP 3: 용어사전의 설명과 실제 데이터를 연결하여 분석
- 📊 STEP 4: 용어와 관련된 현황 데이터를 Analyst1으로 조회
- 📝 STEP 5: 용어사전 기반의 정확하고 권위 있는 해석 제공

⚠️ 주의사항: 
- 임의로 용어를 정의하지 말고 반드시 용어사전 검색 결과를 우선 사용
- 용어사전에서 찾은 내용은 "용어사전에 따르면..." 형태로 명시적 인용
- 검색된 용어사전 내용이 없으면 일반적 정의 후 "용어사전 검색 필요" 언급
"""
    
    # 템플릿 문서가 있으면 구체적인 참조 지시
    if template_guidance:
        enhanced_instruction = f"""
{enhanced_instruction}

🔥 중요: 아래 실제 문서들을 반드시 참조하여 형식을 맞춰주세요:

{template_guidance}

위 문서들의 구조, 용어, 스타일을 최대한 따라서 작성하세요. 단순히 일반적인 분석이 아니라, 실제 조직에서 사용하는 {format_type} 형식의 특징을 정확히 반영해주세요.
"""
    else:
        # 템플릿이 없으면 Search1 도구 적극 활용 지시
        enhanced_instruction = f"""
{enhanced_instruction}

🔍 중요: Search1 도구를 사용하여 '{format_type}' 관련 문서들을 적극적으로 검색하고 참조하세요. 
검색 키워드 예시: "{format_type} 양식", "{format_type} 템플릿", "{format_type} 구조"
검색된 문서의 실제 형식과 구조를 따라서 분석 보고서를 작성하세요.
"""
    
    payload = {
        "model": "claude-4-sonnet",
        "response_instruction": (
            f"{enhanced_instruction}\n\n"
            f"🎯 실행 지침:\n"
            f"1. {'🔍 CRITICAL: 먼저 Search1 도구로 질문의 핵심 용어들을 반드시 검색하세요' if format_type == '비즈니스 용어사전' else '먼저 Search1 도구로 관련 문서들을 검색하세요'}\n"
            f"2. {'📚 검색된 용어사전 내용을 직접 인용하고 참조하세요' if format_type == '비즈니스 용어사전' else '검색된 문서의 구조와 형식을 분석하세요'}\n" 
            f"3. Analyst1 도구로 관련 데이터를 조회하세요\n"
            f"4. {'용어사전 정의 + 실제 데이터를 결합하여 분석하세요' if format_type == '비즈니스 용어사전' else '검색된 문서의 실제 형식을 따라 결과를 구성하세요'}\n"
            f"5. {format_type}의 고유한 특징을 반영한 전문적 보고서로 작성하세요\n\n"
            f"🚨 중요한 환각 방지 규칙:\n"
            f"- 보고일자: 실제 데이터 조회 날짜가 있는 경우에만 사용하고, 없으면 '데이터 기준일: 조회일 기준' 등으로 명시\n"
            f"- 작성부서: 검색된 문서에 명시된 부서명만 사용하고, 없으면 생략하거나 '해당없음'으로 표시\n"
            f"- 보고대상: 실제 문서에서 확인된 승인자/수신자만 사용하고, 없으면 생략\n"
            f"- 조직정보: 검색된 문서에서 확인할 수 없는 조직 구조나 직책은 임의로 생성하지 말 것\n"
            f"- 수치데이터: Analyst1 도구로 조회된 실제 데이터만 사용하고, 추정치는 명확히 '추정' 표시\n"
            f"- 날짜정보: 실제 데이터의 기준일이나 업데이트일만 사용하고, 임의의 날짜 생성 금지\n\n"
            f"{'🔥 비즈니스 용어사전 특별 지시: 질문에 포함된 모든 비즈니스 용어들을 Search1으로 검색하고, 임베딩된 용어사전 데이터에서 정확한 정의를 찾아 인용하세요. 용어사전에 없는 내용은 추측하지 말고 명시하세요.' if format_type == '비즈니스 용어사전' else ''}\n"
            f"실제 데이터와 검색된 문서 내용만을 기반으로 하여 정확하고 신뢰할 수 있는 분석을 제공하세요."
        ),
        "experimental": {},
        "tools": [
            {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "Analyst1"}},
            {"tool_spec": {"type": "cortex_search", "name": "Search1"}},
            {"tool_spec": {"type": "sql_exec", "name": "sql_execution_tool"}},
        ],
        "tool_resources": {
            "Analyst1": {"semantic_model_file": SEMANTIC_MODEL_FILE},
            "Search1": {
                "name": CORTEX_SEARCH_SERVICE,
                "max_results": 8,  # 더 많은 문서 검색
                "title_column": "relative_path",
                "id_column": "doc_id",
                "filter": {"@eq": {"language": "Korean"}}
            }
        },
        "tool_choice": {"type": "auto"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": f"{query} (보고서 형식: {format_type})"}]}
        ],
    }

    request_id = str(uuid.uuid4())
    url = f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/cortex/agent:run"
    headers = {
        **API_HEADERS,
        "Accept": "text/event-stream",
    }

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

    results = await execute_sql(sql) if sql else None

    return {
        "text": text,
        "citations": citations,
        "sql": sql,
        "results": results,
        "applied_format": format_type,
        "format_id": report_format_id,
        "template_guidance_used": bool(template_guidance)
    }


@mcp.tool(description="Smart Cortex Agent that automatically detects optimal report format and provides formatted analysis")
async def run_cortex_agents(query: str) -> Dict[str, Any]:
    """
    지능형 Cortex Agent: 쿼리를 분석하여 최적의 보고서 형식을 자동 감지하고 해당 형식으로 분석을 수행합니다.

    This tool analyzes the user query to automatically detect the most appropriate report format
    and performs the analysis using that format. Falls back to basic analysis if no specific format is detected.

    Args:
        query (str): The user's natural language question to analyze.

    Returns:
        dict: A dictionary containing:
            - text (str): The natural language response from the agent
            - sql (str): The generated SQL query
            - citations (List[dict]): List of document sources cited
            - results (dict): Raw execution result from the Snowflake SQL API
            - detected_format (str): The automatically detected report format
            - confidence_score (int): Confidence level of format detection (0-5)
            - applied_formatted_analysis (bool): Whether formatted analysis was applied
    """
    
    # 🎯 쿼리 분석하여 최적 형식 자동 감지
    query_lower = query.lower()
    detected_format = "기본"
    confidence_score = 0
    
    # 형식별 키워드 매핑 (우선순위 포함)
    format_keywords = {
        "비즈니스 용어사전": {
            "keywords": ["용어", "정의", "의미", "뜻", "설명", "무엇", "개념", "정확한"],
            "priority": 5  # 가장 높은 우선순위
        },
        "내부보고자료": {
            "keywords": ["상세", "자세", "세부", "깊이", "구체적", "상세하게", "완전한", "전체적", "심층"],
            "priority": 4
        },
        "상부보고자료": {
            "keywords": ["요약", "간단", "핵심", "간략", "요점", "summary", "경영진", "임원", "상부"],
            "priority": 4
        },
        "트렌드 자료": {
            "keywords": ["트렌드", "동향", "변화", "추세", "전망", "예측", "흐름", "미래", "시장"],
            "priority": 3
        },
        "경영자 컨설팅 자료": {
            "keywords": ["전략", "개선", "방향", "제안", "솔루션", "방안", "컨설팅", "최적화"],
            "priority": 3
        }
    }
    
    # 각 형식별 매칭 점수 계산
    format_scores = {}
    for format_type, info in format_keywords.items():
        score = 0
        keywords = info["keywords"]
        priority = info["priority"]
        
        # 키워드 매칭 점수
        matched_keywords = [kw for kw in keywords if kw in query_lower]
        if matched_keywords:
            score = len(matched_keywords) * priority
            format_scores[format_type] = {
                "score": score,
                "matched_keywords": matched_keywords,
                "priority": priority
            }
    
    # 가장 높은 점수의 형식 선택
    if format_scores:
        best_format = max(format_scores.items(), key=lambda x: x[1]["score"])
        detected_format = best_format[0]
        confidence_score = best_format[1]["score"]
        matched_keywords = best_format[1]["matched_keywords"]
        
        logging.info(f"형식 자동 감지: {detected_format} (점수: {confidence_score}, 키워드: {matched_keywords})")
    
    # 🔥 자동 형식 적용 조건
    # 신뢰도가 3 이상이면 자동으로 해당 형식 적용
    if confidence_score >= 3:
        logging.info(f"자동 형식 적용: {detected_format} (신뢰도: {confidence_score})")
        
        try:
            # run_formatted_analysis 호출
            formatted_result = await run_formatted_analysis(query, "", detected_format)
            
            # 결과에 자동 감지 정보 추가
            formatted_result.update({
                "detected_format": detected_format,
                "confidence_score": confidence_score,
                "applied_formatted_analysis": True,
                "detection_method": "automatic_keyword_matching"
            })
            
            return formatted_result
            
        except Exception as e:
            logging.error(f"자동 형식 분석 실패: {e}, 기본 분석으로 폴백")
            # 형식 분석 실패 시 기본 분석 계속 진행
    
    # 🔧 기본 분석 수행 (신뢰도가 낮거나 형식 분석 실패 시)
    logging.info(f"기본 분석 수행 (감지된 형식: {detected_format}, 신뢰도: {confidence_score})")
    
    payload = {
        "model": "claude-4-sonnet",
        "response_instruction": (
            "You are a helpful data analytics agent. "
            "1. Always use the Analyst1 tool to convert user questions into SQL, referencing the provided semantic model. "
            "2. Use the Search1 tool to find relevant documentation that might help interpret the data context. "
            "3. Combine the SQL execution results and search results into a concise, structured answer. "
            "4. For SQL generation, ensure queries are safe and match the schema in the semantic model file. "
            "5. Respond in a structured format: natural language answer first, then SQL, then citations. "
            f"6. 💡 Note: This appears to be a request for {detected_format} style analysis based on keywords detected.\n\n"
            "🚨 CRITICAL ANTI-HALLUCINATION RULES:\n"
            "- Report dates: Only use actual data query dates, never fabricate dates like '2025년 1월 기준'\n"
            "- Department names: Only use department names found in searched documents, never invent them\n"
            "- Report recipients: Only mention actual recipients if found in documents, otherwise omit\n"
            "- Organization info: Never create fictional organizational structures or job titles\n"
            "- Data values: Only use data retrieved from Analyst1 tool, mark any estimates clearly\n"
            "- Use only factual information from actual data queries and document searches"
        ),
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
                "max_results": 3,
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

    results = await execute_sql(sql) if sql else None

    return {
        "text": text,
        "citations": citations,
        "sql": sql,
        "results": results,
        "detected_format": detected_format,
        "confidence_score": confidence_score,
        "applied_formatted_analysis": False,  # 기본 분석이므로 False
        "detection_method": "keyword_analysis_insufficient_confidence"
    }


if __name__ == "__main__":
    try:
        logging.info("Cortex Agent MCP 서버 시작...")
        mcp.run(transport="stdio")  # Claude는 stdio 기반 JSON-RPC 만 허용함
    except KeyboardInterrupt:
        logging.info("Cortex Agent MCP 서버가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logging.error(f"Cortex Agent MCP 서버 실행 중 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)