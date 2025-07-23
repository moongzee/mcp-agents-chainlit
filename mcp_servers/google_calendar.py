from datetime import datetime, timedelta
import sys
from mcp.server.fastmcp import FastMCP
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from pydantic import BaseModel, Field
import json
import os
import asyncio
import aiohttp
from aiohttp import web
import webbrowser
from typing import Dict, Any, List
import logging
from datetime import datetime
from dotenv import load_dotenv

# 로깅 설정
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     stream=sys.stdout,
#     force=True  # Python 3.8+
# )

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("google_calendar")

# 세션별 토큰 저장소 (메모리)
sessions: Dict[str, str] = {}

# 전역 변수로 인증 코드 저장
auth_code = None
auth_state = None
callback_server_task = None

# 전역 credentials 캐시
cached_credentials = None

SCOPES = ['https://www.googleapis.com/auth/calendar']
CLIENT_CONFIG = {
    "web": {
        "client_id": os.getenv('GOOGLE_CLIENT_ID'),
        "client_secret": os.getenv('GOOGLE_CLIENT_SECRET'),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:8080/callback"]
    }
}

# 환경변수 확인
if not CLIENT_CONFIG["web"]["client_id"] or not CLIENT_CONFIG["web"]["client_secret"]:
    logger.error("❌ GOOGLE_CLIENT_ID 또는 GOOGLE_CLIENT_SECRET 환경변수가 설정되지 않았습니다.")
    logger.error("Google Cloud Console에서 OAuth 2.0 클라이언트를 생성하고 환경변수를 설정해주세요.")

# credentials는 인증 후에 생성됩니다
def get_cached_credentials():
    """캐시된 credentials를 반환하거나, 토큰 파일에서 새로 생성"""
    global cached_credentials

    # 캐시된 credentials가 있고 유효하면 반환
    if cached_credentials and cached_credentials.valid:
        return cached_credentials

    # 토큰 파일에서 credentials 생성
    try:
        if os.path.exists('token.json'):
            with open('token.json', 'r') as token_file:
                token_data = json.load(token_file)

            credentials = Credentials.from_authorized_user_info(token_data, SCOPES)

            # 토큰이 만료되었고 refresh_token이 있으면 갱신
            if credentials.expired and credentials.refresh_token:
                from google.auth.transport.requests import Request
                credentials.refresh(Request())
                # 갱신된 토큰 저장
                with open('token.json', 'w') as token_file:
                    token_file.write(credentials.to_json())

            # 캐시에 저장
            cached_credentials = credentials
            return credentials
        else:
            return None
    except Exception as e:
        logger.error(f"Credentials 생성 실패: {e}")
        return None

def get_calendar_service():
    """Calendar API 서비스 객체 반환"""
    credentials = get_cached_credentials()
    if not credentials:
        raise Exception("인증이 필요합니다. 먼저 OAuth 인증을 완료해주세요.")

    return build("calendar", "v3", credentials=credentials)

def clear_credentials_cache():
    """credentials 캐시를 무효화"""
    global cached_credentials
    cached_credentials = None

@mcp.tool(description="Generate Google OAuth URL for calendar authentication. ")
async def get_auth_url() -> str:
    """Google 인증 URL 생성"""
    email = "elandinnopleai@gmail.com"

    flow = Flow.from_client_config(CLIENT_CONFIG, scopes=SCOPES)
    flow.redirect_uri = "http://localhost:8080/callback"

    auth_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent' # 매번 새로운 허용 및 리프레시 토큰
    )

    return auth_url

@mcp.tool(description="Start callback server to receive OAuth code. ")
async def start_callback_server() -> str:
    global callback_server_task

    if callback_server_task is not None and not callback_server_task.done():
        return "Callback 서버가 이미 실행 중입니다."

    # 새 Task로 백그라운드 실행
    loop = asyncio.get_running_loop()
    callback_server_task = loop.create_task(run_callback_server())

    return "✅ Callback 서버가 localhost:8080에서 실행되었습니다. 인증 URL을 열어주세요."

async def run_callback_server():
    global auth_code, auth_state

    auth_code = None
    auth_state = None

    app = web.Application()
    app.router.add_get('/callback', callback_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()

    logger.info("🚀 Callback 서버 실행 중...")
    # 여기는 무기한 살아있도록 유지
    while auth_code is None:
        await asyncio.sleep(0.5)

async def callback_handler(request):
    global auth_code, auth_state

    try:
        logger.info("📥 콜백 요청 수신됨")
        logger.debug("요청 URL: %s", str(request.url))
        logger.debug("쿼리 파라미터: %s", request.rel_url.query)

        query = request.rel_url.query
        auth_code = query.get('code')
        auth_state = query.get('state')

        if not auth_code:
            logger.warning("❌ 'code' 파라미터가 없습니다.")
            return web.Response(
                text="인증에 실패했습니다. (code 없음)",
                status=400,
                content_type='text/html'
            )

        logger.info("✅ 인증 성공! Code: %s | State: %s", auth_code, auth_state)

        return web.Response(
            text="✅ 인증이 완료되었습니다. 이 창은 닫아도 됩니다.",
            content_type='text/html'
        )

    except Exception as e:
        logger.error("🔥 콜백 처리 중 예외 발생", exc_info=True)
        return web.Response(
            text=f"❌ 서버 오류: {type(e).__name__}: {e}",
            status=500,
            content_type='text/html'
        )

@mcp.tool(description="Exchange OAuth code for access token. ")
async def exchange_code() -> str:
    """인증 코드를 토큰으로 교환하고 세션에 저장"""
    global auth_code, auth_state

    if not auth_code:
        return "❌ 인증 코드가 아직 없습니다. 먼저 인증을 완료해주세요."

    try:
        flow = Flow.from_client_config(CLIENT_CONFIG, scopes=SCOPES)
        flow.redirect_uri = "http://localhost:8080/callback"
        flow.fetch_token(code=auth_code)

        creds = flow.credentials
        with open(f"token.json", "w") as f:
            f.write(creds.to_json())

        # credentials 캐시 무효화 (다음 호출 시 새로 생성)
        clear_credentials_cache()
        logger.info("✅ 토큰이 저장되었습니다. credentials 캐시가 무효화되었습니다.")

        return "✅ 토큰이 저장되었습니다. 이제 캘린더 기능을 사용할 수 있습니다."
    except Exception as e:
        logger.exception("❌ 토큰 교환 중 예외 발생")
        return f"❌ 토큰 교환 실패: {e}"

@mcp.tool(description="Get the received OAuth code from callback. ")
async def get_received_code() -> Dict[str, Any]:
    """Callback에서 받은 인증 코드를 반환"""
    global auth_code, auth_state

    if auth_code:
        return {
            "code": auth_code,
            "state": auth_state,
            "received": True
        }
    else:
        return {
            "received": False,
            "message": "아직 인증 코드를 받지 못했습니다. 브라우저에서 인증을 완료해주세요."
        }

@mcp.tool(description="List calendar events. Input: {'maxResults': int, 'timeMin': str, 'timeMax': str}. timeMin and timeMax accept ISO format datetime strings.")
def list_events(args: Dict[str, Any]) -> str:
    """List calendar events. You can specify timeMin and timeMax parameters to fetch events from past dates or within a specific date range. Both parameters accept ISO format datetime strings."""
    try:
        maxResults = args.get('maxResults', 10)
        timeMin = args.get('timeMin')
        timeMax = args.get('timeMax')

        if not timeMin:
            timeMin = datetime.now().isoformat() + "Z"

        request_params = {
            "calendarId": "primary",
            "timeMin": timeMin,
            "maxResults": maxResults,
            "singleEvents": True,
            "orderBy": "startTime",
        }
        print(request_params, flush=True)
        if timeMax:
            request_params["timeMax"] = timeMax

        service = get_calendar_service()
        print(service, flush=True)
        response = service.events().list(**request_params).execute()

        events = []
        for event in response.get("items", []):
            events.append({
                "id": event["id"],
                "summary": event.get("summary"),
                "start": event.get("start"),
                "end": event.get("end"),
                "location": event.get("location"),
            })

        return json.dumps(events, indent=2, ensure_ascii=False)

    except Exception as e:
        return f"Error fetching calendar events: {str(e)}"

@mcp.tool(description="Create a new calendar event. Input: {'summary': str, 'start': str, 'end': str, 'location': str, 'description': str, 'attendees': List[str]}")
def create_event(args: Dict[str, Any]) -> str:
    """Create a new calendar event"""
    try:
        summary = args.get('summary')
        start = args.get('start')
        end = args.get('end')
        location = args.get('location')
        description = args.get('description')
        attendees = args.get('attendees')

        event = {
            "summary": summary,
            "start": {
                "dateTime": start,
                "timeZone": "Asia/Seoul",
            },
            "end": {
                "dateTime": end,
                "timeZone": "Asia/Seoul",
            },
        }

        if location:
            event["location"] = location
        if description:
            event["description"] = description
        if attendees:
            event["attendees"] = [{"email": email} for email in attendees]

        service = get_calendar_service()
        response = service.events().insert(
            calendarId="primary",
            body=event
        ).execute()

        return f"Event created successfully. Event ID: {response['id']}"

    except Exception as e:
        return f"Error creating event: {str(e)}"

@mcp.tool(description="Update an existing calendar event. Input: {'eventId': str, 'summary': str, 'location': str, 'description': str, 'start': str, 'end': str, 'attendees': List[str]}")
def update_event(args: Dict[str, Any]) -> str:
    """Update an existing calendar event"""
    try:
        eventId = args.get('eventId')
        summary = args.get('summary')
        location = args.get('location')
        description = args.get('description')
        start = args.get('start')
        end = args.get('end')
        attendees = args.get('attendees')

        event = {}

        if summary:
            event["summary"] = summary
        if location:
            event["location"] = location
        if description:
            event["description"] = description
        if start:
            event["start"] = {
                "dateTime": start,
                "timeZone": "Asia/Seoul",
            }
        if end:
            event["end"] = {
                "dateTime": end,
                "timeZone": "Asia/Seoul",
            }
        if attendees:
            event["attendees"] = [{"email": email} for email in attendees]

        service = get_calendar_service()
        response = service.events().patch(
            calendarId="primary",
            eventId=eventId,
            body=event
        ).execute()

        return f"Event updated successfully. Event ID: {response['id']}"

    except Exception as e:
        return f"Error updating event: {str(e)}"

@mcp.tool(description="Delete a calendar event. Input: {'eventId': str}")
def delete_event(args: Dict[str, Any]) -> str:
    """Delete a calendar event"""
    try:
        eventId = args.get('eventId')

        service = get_calendar_service()
        service.events().delete(
            calendarId="primary",
            eventId=eventId
        ).execute()

        return f"Event deleted successfully. Event ID: {eventId}"

    except Exception as e:
        return f"Error deleting event: {str(e)}"

# 테스트용 함수들
async def test_auth_flow():
    """테스트용 인증 플로우"""
    # 1. Callback 서버 시작
    await start_callback_server()

    # 2. 인증 URL 생성
    test_email = "elandinnopleai@gmail.com"
    auth_url = await get_auth_url()

    print(f"\n🔗 인증 URL: {auth_url}", flush=True)

    # 3. 브라우저에서 인증 URL 열기
    # webbrowser.open(auth_url)

    # 4. 인증 완료까지 대기
    print("인증을 완료해주세요...", flush=True)
    while auth_code is None:
        await asyncio.sleep(1)

    # 5. 토큰 교환
    if auth_code:
        print(f"\n🔄 토큰 교환 중...", flush=True)
        result = await exchange_code()
        print(f"결과: {result}", flush=True)

    # 6. 일정 가져오기 테스트
    print("\n📅 일정 가져오기 테스트...", flush=True)
    events = list_events({'maxResults': 5, 'timeMin': (datetime.now() - timedelta(days=1)).replace(microsecond=0).isoformat() + "Z", 'timeMax': (datetime.now() + timedelta(days=1)).replace(microsecond=0).isoformat() + "Z"})  # 최근 5개 일정만 가져오기
    print(f"일정 목록:\n{events}", flush=True)

async def test_with_saved_token():
    """저장된 토큰을 사용하여 테스트"""
    try:
        print("\n📅 저장된 토큰으로 일정 가져오기 테스트...", flush=True)
        events = list_events({
            'maxResults': 5,
            'timeMin': (datetime.now() - timedelta(days=1)).replace(microsecond=0).isoformat() + "Z",
            'timeMax': (datetime.now() + timedelta(days=1)).replace(microsecond=0).isoformat() + "Z"
        })
        if events:
            print(f"일정 목록:\n{events}", flush=True)
        else:
            print("일정이 없습니다.", flush=True)

        # 새 일정 생성 테스트
        print("\n📝 새 일정 생성 테스트...", flush=True)
        new_event = create_event({
            'summary': '테스트 일정',
            'start': (datetime.now() + timedelta(hours=1)).isoformat(),
            'end': (datetime.now() + timedelta(hours=2)).isoformat(),
            'description': '토큰 테스트로 생성된 일정입니다.'
        })
        print(f"생성된 일정:\n{new_event}", flush=True)

        # 생성된 일정 ID 추출
        event_data = json.loads(new_event)
        event_id = event_data.get('id')

        if event_id:
            # 일정 삭제 테스트
            print("\n🗑 생성된 일정 삭제 테스트...", flush=True)
            delete_result = delete_event({'eventId': event_id})
            print(f"삭제 결과: {delete_result}", flush=True)

    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {str(e)}", flush=True)

if __name__ == "__main__":
    # asyncio.run(test_auth_flow())
    mcp.run(transport="stdio")
