from typing import Any
import httpx
from datetime import datetime, timedelta
import json
import urllib.parse
import math
import asyncio
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os

# Initialize FastMCP server
mcp = FastMCP("korea_weather")

KOREA_WEATHER_API_KEY = "NJZf0IxXtTO8vlgpcZ8TbyYzgziNOLkFbn8dWmvTRbx4AYWTjdPnpNd2nbroAcineXLi971rvGbpoy23qSMPmQ%3D%3D"

# Lambert Conformal Conic Projection 파라미터
class LambertConformalConic:
    def __init__(self):
        self.Re = 6371.00877    # 지구 반경(km)
        self.grid = 5.0         # 격자 간격(km)
        self.slat1 = 30.0       # 표준 위도 1
        self.slat2 = 60.0       # 표준 위도 2
        self.olon = 126.0       # 기준점 경도
        self.olat = 38.0        # 기준점 위도
        self.xo = 210/self.grid # 기준점 X좌표
        self.yo = 675/self.grid # 기준점 Y좌표
        self.first = True
        
        # 미리 계산할 상수들
        if self.first:
            self.PI = math.pi
            self.DEGRAD = self.PI/180.0
            self.RADDEG = 180.0/self.PI
            
            self.re = self.Re/self.grid
            self.slat1_rad = self.slat1 * self.DEGRAD
            self.slat2_rad = self.slat2 * self.DEGRAD
            self.olon_rad = self.olon * self.DEGRAD
            self.olat_rad = self.olat * self.DEGRAD
            
            self.sn = math.tan(self.PI * 0.25 + self.slat2_rad * 0.5) / math.tan(self.PI * 0.25 + self.slat1_rad * 0.5)
            self.sn = math.log(math.cos(self.slat1_rad)/math.cos(self.slat2_rad)) / math.log(self.sn)
            self.sf = math.tan(self.PI * 0.25 + self.slat1_rad * 0.5)
            self.sf = math.pow(self.sf, self.sn) * math.cos(self.slat1_rad) / self.sn
            self.ro = math.tan(self.PI * 0.25 + self.olat_rad * 0.5)
            self.ro = self.re * self.sf / math.pow(self.ro, self.sn)
            self.first = False

def convert_grid_gps(lon, lat, map_instance):
    """
    위경도 좌표를 기상청 격자 좌표로 변환
    """
    ra = math.tan(map_instance.PI * 0.25 + lat * map_instance.DEGRAD * 0.5)
    ra = map_instance.re * map_instance.sf / math.pow(ra, map_instance.sn)
    theta = lon * map_instance.DEGRAD - map_instance.olon_rad
    
    if theta > map_instance.PI:
        theta -= 2.0 * map_instance.PI
    if theta < -map_instance.PI:
        theta += 2.0 * map_instance.PI
        
    theta *= map_instance.sn
    
    x = ra * math.sin(theta) + map_instance.xo
    y = map_instance.ro - ra * math.cos(theta) + map_instance.yo
    
    return int(x + 1.5), int(y + 1.5)

def get_grid_coordinate_from_lonlat(lon, lat):
    """
    위경도 좌표를 입력받아 기상청 격자 좌표(nx, ny)를 반환합니다.
    
    Args:
        lon (float): 경도 값
        lat (float): 위도 값
    
    Returns:
        tuple: (nx, ny) 격자 좌표
    """
    map_instance = LambertConformalConic()
    nx, ny = convert_grid_gps(lon, lat, map_instance)
    return nx, ny

async def get_nowcast_observation_from_api(lon, lat) -> str:
    """위경도 좌표의 현재 날씨 정보를 조회합니다.

    Args:
        lon (float): 경도 값
        lat (float): 위도 값
        
    Returns:
        str: 날씨 정보 문자열
    """
    
    try:
        api_result = []
        # 격자 좌표 조회
        nx, ny = get_grid_coordinate_from_lonlat(lon, lat)
        location_str = f"위도 {lat}, 경도 {lon}"
        
        # 기상청 API 키 (URL 디코딩)
        api_key = urllib.parse.unquote(KOREA_WEATHER_API_KEY)
        
        # 현재 시간 정보 (매시각 40분 이후 호출가능)
        now = datetime.now()
        if now.minute < 40:  # 40분 이전이면 한 시간 전 데이터 사용
            now = now - timedelta(hours=1)
        base_date = now.strftime("%Y%m%d")
        base_time = now.strftime("%H00")

        # API 요청 URL
        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
        params = {
            'serviceKey': api_key,
            'numOfRows': '10',
            'pageNo': '1',
            'dataType': 'JSON',
            'base_date': base_date,
            'base_time': base_time,
            'nx': nx,
            'ny': ny
        }

        # 비동기 HTTP 요청을 위해 httpx 사용
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            # api_result.append(f"API 응답 상태 코드: {response.status_code}\n")
            
            if response.status_code == 200:
                result = response.json()
                items = result['response']['body']['items']['item']
                weather_data = {}
                
                for item in items:
                    category = item['category']
                    value = item['obsrValue']
                    
                    if category == 'T1H':  # 기온
                        weather_data['temperature'] = f"{value}°C"
                    elif category == 'RN1':  # 1시간 강수량
                        weather_data['rainfall'] = f"{value}mm"
                    elif category == 'REH':  # 습도
                        weather_data['humidity'] = f"{value}%"
                    elif category == 'WSD':  # 풍속
                        weather_data['wind_speed'] = f"{value}m/s"

                api_result.append(f"\n{location_str} 현재 날씨:")
                api_result.append(f"기온: {weather_data.get('temperature', 'N/A')}\n")
                api_result.append(f"강수량: {weather_data.get('rainfall', 'N/A')}\n")
                api_result.append(f"습도: {weather_data.get('humidity', 'N/A')}\n")
                api_result.append(f"풍속: {weather_data.get('wind_speed', 'N/A')}\n")
                
            else:
                api_result.append(f"날씨 정보를 가져오는데 실패했습니다. 상태 코드: {response.status_code}\n")
                
    except ValueError as e:
        api_result.append(f"오류: {e}\n")
    except httpx.RequestError as e:
        api_result.append(f"API 요청 중 오류 발생: {e}\n")
    except json.JSONDecodeError as e:
        api_result.append(f"JSON 파싱 중 오류 발생: {e}\n")
    except Exception as e:
        api_result.append(f"예상치 못한 오류 발생: {e}\n")

    # 문자열 리스트를 하나의 문자열로 변환하여 반환
    return "".join(api_result)

async def get_nowcast_forecast_from_api(lon, lat) -> str:
    """위경도 좌표의 초단기 (6시간 이내) 예보 정보를 조회합니다.

    Args:
        lon (float): 경도 값
        lat (float): 위도 값
        
    Returns:
        str: 초단기 예보 정보 문자열
    """
    
    try:
        api_result = []
        # 격자 좌표 조회
        nx, ny = get_grid_coordinate_from_lonlat(lon, lat)
        location_str = f"위도 {lat}, 경도 {lon}"
        
        # 기상청 API 키 (URL 디코딩)
        api_key = urllib.parse.unquote(KOREA_WEATHER_API_KEY)
        
        # 현재 시간 정보 (매시각 30분 이후 호출가능)
        now = datetime.now()
        # 45분 이전이면 한 시간 전 데이터 사용 (API는 매시간 30분에 생성되고 45분 이후 호출 가능)
        if now.minute < 45:
            now = now - timedelta(hours=1)
            
        base_date = now.strftime("%Y%m%d")
        base_time = now.strftime("%H30")  # 매시간 30분 발표

        # API 요청 URL
        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
        params = {
            'serviceKey': api_key,
            'numOfRows': '60',  # 시간당 10개 요소 * 6시간 예보
            'pageNo': '1',
            'dataType': 'JSON',
            'base_date': base_date,
            'base_time': base_time,
            'nx': nx,
            'ny': ny
        }

        # 비동기 HTTP 요청을 위해 httpx 사용
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            # api_result.append(f"API 응답 상태 코드: {response.status_code}\n")
            
            if response.status_code == 200:
                result = response.json()
                # API 응답 체크
                if 'response' in result and 'body' in result['response'] and 'items' in result['response']['body']:
                    items = result['response']['body']['items']['item']
                    
                    # 예보 시간별로 데이터 그룹화
                    forecast_by_time = {}
                    for item in items:
                        fcst_date = item['fcstDate']
                        fcst_time = item['fcstTime']
                        category = item['category']
                        value = item['fcstValue']
                        
                        time_key = f"{fcst_date} {fcst_time}"
                        if time_key not in forecast_by_time:
                            forecast_by_time[time_key] = {}
                            
                        forecast_by_time[time_key][category] = value
                    
                    # 헤더 정보 추가
                    # 발표 시간 형식 변경 {base_date} {base_time} -> XXXX년 MM월 DD일 HH:MM
                    base_date_str = base_date[:4] + "년 " + base_date[4:6] + "월 " + base_date[6:] + "일 " + base_time[:2] + ":" + base_time[2:] + "시"
                    api_result.append(f"\n{location_str} 초단기 예보 (발표: {base_date_str})\n")
                    api_result.append("=" * 50 + "\n")
                    
                    # 시간순으로 정렬
                    sorted_times = sorted(forecast_by_time.keys())
                    
                    # 각 시간대별 예보 정보 출력
                    for time_key in sorted_times:
                        fcst_data = forecast_by_time[time_key]
                        # date_str 형식 변경 {base_date} {base_time} -> XXXX년 MM월 DD일
                        date_str = base_date[:4] + "년 " + base_date[4:6] + "월 " + base_date[6:] + "일"
                        time_str = time_key[9:]
                        formatted_time = f"{time_str[:2]}:{time_str[2:]}시"
                        
                        api_result.append(f"■ {date_str} {formatted_time} 예보\n")
                        
                        # 기온
                        if 'T1H' in fcst_data:
                            api_result.append(f"  기온: {fcst_data['T1H']}°C\n")
                        
                        # 강수형태
                        if 'PTY' in fcst_data:
                            pty_code = int(fcst_data['PTY'])
                            pty_str = "없음"
                            if pty_code == 1:
                                pty_str = "비"
                            elif pty_code == 2:
                                pty_str = "비/눈"
                            elif pty_code == 3:
                                pty_str = "눈"
                            elif pty_code == 5:
                                pty_str = "빗방울"
                            elif pty_code == 6:
                                pty_str = "빗방울눈날림"
                            elif pty_code == 7:
                                pty_str = "눈날림"
                            api_result.append(f"  강수형태: {pty_str}\n")
                        
                        # 강수량
                        if 'RN1' in fcst_data:
                            rain_value = fcst_data['RN1']
                            if rain_value == '강수없음':
                                api_result.append(f"  1시간 강수량: 없음\n")
                            else:
                                api_result.append(f"  1시간 강수량: {rain_value}\n")
                        
                        # 습도
                        if 'REH' in fcst_data:
                            api_result.append(f"  습도: {fcst_data['REH']}%\n")
                        
                        # 하늘상태
                        if 'SKY' in fcst_data:
                            sky_code = int(fcst_data['SKY'])
                            sky_str = "알 수 없음"
                            if sky_code == 1:
                                sky_str = "맑음"
                            elif sky_code == 3:
                                sky_str = "구름많음"
                            elif sky_code == 4:
                                sky_str = "흐림"
                            api_result.append(f"  하늘상태: {sky_str}\n")
                        
                        # 풍속
                        if 'WSD' in fcst_data:
                            api_result.append(f"  풍속: {fcst_data['WSD']}m/s\n")
                        
                        # 풍향
                        if 'VEC' in fcst_data:
                            vec_value = float(fcst_data['VEC'])
                            # 16방위 변환
                            vec_dir = int((vec_value + 22.5 * 0.5) / 22.5) % 16
                            dir_str = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                                       "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"][vec_dir]
                            api_result.append(f"  풍향: {dir_str} ({fcst_data['VEC']}°)\n")
                        
                        # 낙뢰
                        if 'LGT' in fcst_data:
                            lgt_value = int(fcst_data['LGT'])
                            lgt_str = "없음" if lgt_value == 0 else f"{lgt_value} kA/㎢"
                            api_result.append(f"  낙뢰: {lgt_str}\n")
                        
                        api_result.append("\n")
                else:
                    api_result.append(f"API 응답에서 데이터를 찾을 수 없습니다.\n")
                    if 'response' in result and 'header' in result['response']:
                        header = result['response']['header']
                        if 'resultCode' in header and 'resultMsg' in header:
                            api_result.append(f"결과 코드: {header['resultCode']}\n")
                            api_result.append(f"결과 메시지: {header['resultMsg']}\n")
            else:
                api_result.append(f"날씨 정보를 가져오는데 실패했습니다. 상태 코드: {response.status_code}\n")
                if response.status_code == 529:
                    api_result.append("서버가 과부하 상태입니다. 잠시 후 다시 시도해주세요.\n")
                
    except ValueError as e:
        api_result.append(f"오류: {e}\n")
    except httpx.RequestError as e:
        api_result.append(f"API 요청 중 오류 발생: {e}\n")
    except json.JSONDecodeError as e:
        api_result.append(f"JSON 파싱 중 오류 발생: {e}\n")
    except Exception as e:
        api_result.append(f"예상치 못한 오류 발생: {e}\n")

    # 문자열 리스트를 하나의 문자열로 변환하여 반환
    return "".join(api_result)

async def get_short_term_forecast_from_api(lon, lat) -> str:
    """위경도 좌표의 단기 예보 (3일~5일) 정보를 조회합니다.

    Args:
        lon (float): 경도 값
        lat (float): 위도 값
        
    Returns:
        str: 단기 예보 (3일~5일) 정보 문자열
    """
    
    try:
        api_result = []
        # 격자 좌표 조회
        nx, ny = get_grid_coordinate_from_lonlat(lon, lat)
        location_str = f"위도 {lat}, 경도 {lon}"
        
        # 기상청 API 키 (URL 디코딩)
        api_key = urllib.parse.unquote(KOREA_WEATHER_API_KEY)
        
        # 현재 시간 정보
        now = datetime.now()
        
        # 발표시각 결정 (0200, 0500, 0800, 1100, 1400, 1700, 2000, 2300)
        base_times = ['0200', '0500', '0800', '1100', '1400', '1700', '2000', '2300']
        current_hour = int(now.strftime('%H'))
        
        # 가장 최근 발표시각 찾기 (발표 후 10분 소요 감안)
        available_time = None
        for bt in base_times:
            bt_hour = int(bt[:2])
            if current_hour > bt_hour or (current_hour == bt_hour and now.minute >= 10):
                available_time = bt
        
        # 만약 당일 발표된 예보가 없으면 전날 마지막 예보 사용
        if available_time is None:
            available_time = '2300'
            now = now - timedelta(days=1)
            
        base_date = now.strftime("%Y%m%d")
        base_time = available_time

        # API 요청 URL
        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        params = {
            'serviceKey': api_key,
            'numOfRows': '1000',  # 충분한 데이터 요청 (최대 3일치 예보)
            'pageNo': '1',
            'dataType': 'JSON',
            'base_date': base_date,
            'base_time': base_time,
            'nx': nx,
            'ny': ny
        }

        # 비동기 HTTP 요청을 위해 httpx 사용
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            # api_result.append(f"API 응답 상태 코드: {response.status_code}\n")
            
            if response.status_code == 200:
                result = response.json()
                # API 응답 체크
                if 'response' in result and 'body' in result['response'] and 'items' in result['response']['body']:
                    items = result['response']['body']['items']['item']
                    total_count = result['response']['body']['totalCount']
                    
                    # 날짜 및 시간별로 데이터 그룹화
                    forecast_by_date_time = {}
                    for item in items:
                        fcst_date = item['fcstDate']
                        fcst_time = item['fcstTime']
                        category = item['category']
                        value = item['fcstValue']
                        
                        # 날짜별 그룹화
                        if fcst_date not in forecast_by_date_time:
                            forecast_by_date_time[fcst_date] = {}
                        
                        # 시간별 그룹화
                        time_key = fcst_time
                        if time_key not in forecast_by_date_time[fcst_date]:
                            forecast_by_date_time[fcst_date][time_key] = {}
                            
                        forecast_by_date_time[fcst_date][time_key][category] = value
                    
                    # 헤더 정보 추가
                    api_result.append(f"\n{location_str} 단기 예보 (발표: {base_date} {base_time})\n")
                    api_result.append(f"총 {total_count}개 데이터 조회\n")
                    api_result.append("=" * 50 + "\n")
                    
                    # 날짜별 처리
                    for fcst_date in sorted(forecast_by_date_time.keys()):
                        # 날짜 포맷팅 (YYYYMMDD -> YYYY년 MM월 DD일)
                        formatted_date = f"{fcst_date[:4]}년 {fcst_date[4:6]}월 {fcst_date[6:]}일"
                        api_result.append(f"\n【 {formatted_date} 예보 】\n")
                        
                        # 일 최고/최저기온 찾기
                        tmn, tmx = None, None
                        for time_data in forecast_by_date_time[fcst_date].values():
                            if 'TMN' in time_data and time_data['TMN'] != '':
                                tmn = time_data['TMN']
                            if 'TMX' in time_data and time_data['TMX'] != '':
                                tmx = time_data['TMX']
                        
                        # 일 최고/최저기온 출력
                        if tmn is not None:
                            api_result.append(f"  ▶ 최저기온: {tmn}°C\n")
                        if tmx is not None:
                            api_result.append(f"  ▶ 최고기온: {tmx}°C\n")
                        
                        api_result.append("\n  시간별 예보:\n")
                        
                        # 시간별 예보 처리
                        for fcst_time in sorted(forecast_by_date_time[fcst_date].keys()):
                            # 오전/오후 표시를 위한 시간 변환
                            hour = int(fcst_time[:2])
                            am_pm = "오전" if hour < 12 else "오후"
                            display_hour = hour if hour <= 12 else hour - 12
                            if hour == 0:
                                display_hour = 12
                                am_pm = "오전"
                            
                            formatted_time = f"{am_pm} {display_hour}시"
                            time_data = forecast_by_date_time[fcst_date][fcst_time]
                            
                            # 주요 날씨 정보 수집
                            weather_info = []
                            
                            # 기온
                            if 'TMP' in time_data:
                                weather_info.append(f"기온 {time_data['TMP']}°C")
                            
                            # 강수확률
                            if 'POP' in time_data:
                                weather_info.append(f"강수확률 {time_data['POP']}%")
                            
                            # 강수형태
                            if 'PTY' in time_data:
                                pty_code = int(time_data['PTY'])
                                pty_str = ""
                                if pty_code == 1:
                                    pty_str = "비"
                                elif pty_code == 2:
                                    pty_str = "비/눈"
                                elif pty_code == 3:
                                    pty_str = "눈"
                                elif pty_code == 4:
                                    pty_str = "소나기"
                                
                                if pty_str:
                                    weather_info.append(f"{pty_str}")
                            
                            # 강수량
                            if 'PCP' in time_data:
                                pcp_value = time_data['PCP']
                                if pcp_value != "강수없음" and pcp_value != "":
                                    weather_info.append(f"강수량 {pcp_value}")
                            
                            # 신적설
                            if 'SNO' in time_data:
                                sno_value = time_data['SNO']
                                if sno_value != "적설없음" and sno_value != "":
                                    weather_info.append(f"적설 {sno_value}")
                            
                            # 하늘상태
                            if 'SKY' in time_data:
                                sky_code = int(time_data['SKY'])
                                sky_str = ""
                                if sky_code == 1:
                                    sky_str = "맑음"
                                elif sky_code == 3:
                                    sky_str = "구름많음"
                                elif sky_code == 4:
                                    sky_str = "흐림"
                                
                                if sky_str:
                                    weather_info.append(f"{sky_str}")
                            
                            # 습도
                            if 'REH' in time_data:
                                weather_info.append(f"습도 {time_data['REH']}%")
                            
                            # 풍속과 풍향
                            wind_info = []
                            
                            if 'VEC' in time_data and 'WSD' in time_data:
                                vec_value = float(time_data['VEC'])
                                wsd_value = float(time_data['WSD'])
                                
                                # 풍향 변환 (16방위)
                                vec_dir = int((vec_value + 22.5 * 0.5) / 22.5) % 16
                                dir_names = ["북", "북북동", "북동", "동북동", "동", "동남동", "남동", "남남동", 
                                           "남", "남남서", "남서", "서남서", "서", "서북서", "북서", "북북서"]
                                dir_str = dir_names[vec_dir]
                                
                                # 풍속 해석
                                wind_desc = ""
                                if wsd_value < 4:
                                    wind_desc = "약한 바람"
                                elif wsd_value < 9:
                                    wind_desc = "약간 강한 바람"
                                else:
                                    wind_desc = "강한 바람"
                                
                                wind_info.append(f"{dir_str}풍 {wind_desc}({wsd_value}m/s)")
                            
                            # 출력
                            api_result.append(f"  ■ {formatted_time}: {', '.join(weather_info)}\n")
                            if wind_info:
                                api_result.append(f"    - {', '.join(wind_info)}\n")
                else:
                    api_result.append(f"API 응답에서 데이터를 찾을 수 없습니다.\n")
                    if 'response' in result and 'header' in result['response']:
                        header = result['response']['header']
                        if 'resultCode' in header and 'resultMsg' in header:
                            api_result.append(f"결과 코드: {header['resultCode']}\n")
                            api_result.append(f"결과 메시지: {header['resultMsg']}\n")
            else:
                api_result.append(f"날씨 정보를 가져오는데 실패했습니다. 상태 코드: {response.status_code}\n")
                if response.status_code == 529:
                    api_result.append("서버가 과부하 상태입니다. 잠시 후 다시 시도해주세요.\n")
                
    except ValueError as e:
        api_result.append(f"오류: {e}\n")
    except httpx.RequestError as e:
        api_result.append(f"API 요청 중 오류 발생: {e}\n")
    except json.JSONDecodeError as e:
        api_result.append(f"JSON 파싱 중 오류 발생: {e}\n")
    except Exception as e:
        api_result.append(f"예상치 못한 오류 발생: {e}\n")

    # 문자열 리스트를 하나의 문자열로 변환하여 반환
    return "".join(api_result)


@mcp.tool(description="Retrieves only the current weather information for a specific latitude and longitude. Historical or forecast data is not available.")    
async def get_nowcast_observation(lon: float, lat: float) -> str:
    """특정 위경도의 날씨 정보를 가져옵니다.
    
    Args:
        lon (float): 경도 값
        lat (float): 위도 값
    """
    response = await get_nowcast_observation_from_api(lon, lat)
    return response


@mcp.tool(description="Retrieves weather forecasts for the next 3 to 5 days at a specific latitude and longitude. Historical data is not available.")
async def get_short_term_forecast(lon: float, lat: float) -> str:
    """특정 위경도의 단기 예보 (3일~5일) 정보를 가져옵니다.
    
    Args:
        lon (float): 경도 값
        lat (float): 위도 값
        
    Returns:
        str: 단기 예보 정보 문자열 (최대 3일 예보)
    """
    response = await get_short_term_forecast_from_api(lon, lat)
    return response


@mcp.tool(description="Retrieves weather forecasts for a specific latitude and longitude within the next 6 hours. Historical data is not available.")
async def get_nowcast_forecast(lon: float, lat: float) -> str:
    """특정 위경도의 초단기 (6시간 이내) 예보 정보를 가져옵니다.
    
    Args:
        lon (float): 경도 값
        lat (float): 위도 값
        
    Returns:
        str: 초단기 (6시간 이내) 예보 정보 문자열
    """
    response = await get_nowcast_forecast_from_api(lon, lat)
    return response


if __name__ == "__main__":
    mcp.run(transport="stdio")
