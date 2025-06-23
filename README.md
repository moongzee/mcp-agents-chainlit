# MCP Agents Chainlit MVP Prototype

이 프로젝트는 MCP 플랫폼 기반의 Chainlit MVP 프로토타입입니다.  
아직 리팩토링이 진행 중이며, 코드 구조 및 기능이 변경될 수 있습니다.

## 주요 특징

- Chainlit, LangChain, LangGraph 등 최신 LLM 프레임워크 활용
- MCP 플랫폼 연동
- 데이터 처리 및 모델 설정을 위한 다양한 패키지 사용

## 시스템 구성 요소
```
+------------------+
|      User        |
+--------+---------+
         |
         v
+-------------------------+
| LangGraph ReAct Agent  |
|    (app_chainlit.py)   |
+-----------+-------------+
            |
            | Reads / Writes
            v
+---------------------------+
|         SQLite DB         |
|  - User Info              |
|  - Memory / History       |
+---------------------------+

            |
            | Loads system prompts
            v
+---------------------------+
|     system_prompt.txt     |
+---------------------------+

            |
            | LLM API Calls
            v
+-----------------------------+
| Snowflake Cortex MCP Agent |
+-----------------------------+

            |
            v
+----------------------------+
|     Docker Container       |
|  - Runs all above modules  |
|  - API exposed via port    |
+----------------------------+
```



## 설치 방법

1. 저장소 클론
2. **Python 3.11** 환경 준비
3. 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

## 실행 방법

1. 프로젝트 디렉토리로 이동
2. 아래 명령어로 Chainlit 서버 실행
   ```bash
   chainlit run [실행할_파이썬_파일].py
   ```
   (예시: chainlit run main.py --port 8000)
3. 웹 브라우저에서 [http://localhost:8000](http://localhost:8000) 접속

## 사용자 추가 방법

- 사용자를 추가하려면 반드시 add_user 파일(스크립트)을 사용해야 합니다.
- add_user 파일의 사용법은 해당 파일 내 주석 또는 도움말을 참고하세요.

## 리팩토링 안내

- 현재 코드베이스는 리팩토링이 예정되어 있습니다.
- 구조, 함수명, 사용법 등이 변경될 수 있으니 참고 바랍니다.
- 최신 사용법 및 변경사항은 README와 코드 주석을 확인해 주세요.

## 문의

- 개선사항, 버그 제보 등은 고객의 생소리함 접수 plz
