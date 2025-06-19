import sqlite3
import json
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
import os

app = Flask(__name__)
DB_PATH = "chat_history.db"

# HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NOA 데이터베이스 뷰어</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .nav {
            background: #f8f9fa;
            padding: 15px 30px;
            border-bottom: 1px solid #dee2e6;
        }
        .nav select, .nav button {
            padding: 8px 15px;
            margin-right: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: white;
        }
        .nav button {
            background: #007bff;
            color: white;
            cursor: pointer;
        }
        .nav button:hover {
            background: #0056b3;
        }
        .content {
            padding: 30px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007bff;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            color: #6c757d;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .query-section {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            resize: vertical;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success {
            color: #155724;
            background: #d4edda;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 NOA 데이터베이스 뷰어</h1>
            <p>채팅 데이터베이스를 쉽게 조회하고 분석하세요</p>
        </div>
        
        <div class="nav">
            <select id="tableSelect" onchange="loadTable()">
                <option value="">테이블 선택...</option>
            </select>
            <button onclick="loadStats()">📈 통계 새로고침</button>
            <button onclick="downloadCSV()">📥 CSV 다운로드</button>
        </div>
        
        <div class="content">
            <!-- 통계 섹션 -->
            <div class="stats" id="statsSection">
                <div class="stat-card">
                    <div class="stat-number" id="userCount">-</div>
                    <div class="stat-label">총 사용자 수</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="sessionCount">-</div>
                    <div class="stat-label">채팅 세션 수</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="messageCount">-</div>
                    <div class="stat-label">총 메시지 수</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="todayCount">-</div>
                    <div class="stat-label">오늘 메시지</div>
                </div>
            </div>
            
            <!-- 데이터 테이블 -->
            <div id="tableSection">
                <div class="loading">테이블을 선택해주세요</div>
            </div>
            
            <!-- 쿼리 섹션 -->
            <div class="query-section">
                <h3>🔧 사용자 정의 쿼리</h3>
                <p style="color: #dc3545;">⚠️ 보안상 SELECT 쿼리만 실행됩니다</p>
                <select id="querySelect" onchange="setQuery()">
                    <option value="">미리 정의된 쿼리 선택...</option>
                    <option value="SELECT * FROM users">전체 사용자 목록</option>
                    <option value="SELECT * FROM chat_sessions ORDER BY created_at DESC LIMIT 10">최근 10개 채팅 세션</option>
                    <option value="SELECT * FROM chat_messages ORDER BY timestamp DESC LIMIT 20">최근 20개 메시지</option>
                    <option value="SELECT u.display_name, COUNT(cm.id) as message_count FROM users u LEFT JOIN chat_sessions cs ON u.id = cs.user_id LEFT JOIN chat_messages cm ON cs.id = cm.session_id GROUP BY u.id, u.display_name">사용자별 메시지 수</option>
                </select>
                <textarea id="queryInput" rows="4" placeholder="SELECT * FROM table_name WHERE condition"></textarea>
                <button onclick="executeQuery()" style="margin-top: 10px;">🚀 쿼리 실행</button>
                <div id="queryResult"></div>
            </div>
        </div>
    </div>

    <script>
        let currentData = [];
        
        // 페이지 로드 시 초기화
        window.onload = function() {
            loadTables();
            loadStats();
        }
        
        // 테이블 목록 로드
        async function loadTables() {
            try {
                const response = await fetch('/api/tables');
                const tables = await response.json();
                const select = document.getElementById('tableSelect');
                select.innerHTML = '<option value="">테이블 선택...</option>';
                tables.forEach(table => {
                    select.innerHTML += `<option value="${table}">${table}</option>`;
                });
            } catch (error) {
                console.error('테이블 로드 실패:', error);
            }
        }
        
        // 선택된 테이블 데이터 로드
        async function loadTable() {
            const tableName = document.getElementById('tableSelect').value;
            if (!tableName) return;
            
            document.getElementById('tableSection').innerHTML = '<div class="loading">데이터 로딩 중...</div>';
            
            try {
                const response = await fetch(`/api/table/${tableName}`);
                const data = await response.json();
                currentData = data;
                displayTable(data, tableName);
            } catch (error) {
                document.getElementById('tableSection').innerHTML = '<div class="error">데이터 로드 실패</div>';
            }
        }
        
        // 테이블 표시
        function displayTable(data, tableName) {
            if (!data || data.length === 0) {
                document.getElementById('tableSection').innerHTML = '<div class="loading">데이터가 없습니다</div>';
                return;
            }
            
            const columns = Object.keys(data[0]);
            let html = `<h3>📋 ${tableName} (${data.length}개 행)</h3>`;
            html += '<table><thead><tr>';
            columns.forEach(col => {
                html += `<th>${col}</th>`;
            });
            html += '</tr></thead><tbody>';
            
            data.slice(0, 100).forEach(row => {  // 최대 100개 행만 표시
                html += '<tr>';
                columns.forEach(col => {
                    let value = row[col];
                    if (typeof value === 'string' && value.length > 100) {
                        value = value.substring(0, 100) + '...';
                    }
                    html += `<td>${value || ''}</td>`;
                });
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            if (data.length > 100) {
                html += `<p style="text-align: center; color: #6c757d;">처음 100개 행만 표시됩니다. 전체 ${data.length}개 행</p>`;
            }
            
            document.getElementById('tableSection').innerHTML = html;
        }
        
        // 통계 로드
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                document.getElementById('userCount').textContent = stats.users || 0;
                document.getElementById('sessionCount').textContent = stats.sessions || 0;
                document.getElementById('messageCount').textContent = stats.messages || 0;
                document.getElementById('todayCount').textContent = stats.today || 0;
            } catch (error) {
                console.error('통계 로드 실패:', error);
            }
        }
        
        // 미리 정의된 쿼리 설정
        function setQuery() {
            const query = document.getElementById('querySelect').value;
            document.getElementById('queryInput').value = query;
        }
        
        // 쿼리 실행
        async function executeQuery() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) {
                alert('쿼리를 입력해주세요');
                return;
            }
            
            document.getElementById('queryResult').innerHTML = '<div class="loading">쿼리 실행 중...</div>';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({query: query})
                });
                
                const result = await response.json();
                
                if (result.error) {
                    document.getElementById('queryResult').innerHTML = `<div class="error">${result.error}</div>`;
                } else {
                    displayQueryResult(result.data);
                }
            } catch (error) {
                document.getElementById('queryResult').innerHTML = '<div class="error">쿼리 실행 실패</div>';
            }
        }
        
        // 쿼리 결과 표시
        function displayQueryResult(data) {
            if (!data || data.length === 0) {
                document.getElementById('queryResult').innerHTML = '<div class="success">쿼리가 실행되었지만 결과가 없습니다</div>';
                return;
            }
            
            const columns = Object.keys(data[0]);
            let html = '<div class="success">쿼리 실행 완료!</div>';
            html += `<h4>결과 (${data.length}개 행)</h4>`;
            html += '<table><thead><tr>';
            columns.forEach(col => {
                html += `<th>${col}</th>`;
            });
            html += '</tr></thead><tbody>';
            
            data.forEach(row => {
                html += '<tr>';
                columns.forEach(col => {
                    let value = row[col];
                    if (typeof value === 'string' && value.length > 100) {
                        value = value.substring(0, 100) + '...';
                    }
                    html += `<td>${value || ''}</td>`;
                });
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            document.getElementById('queryResult').innerHTML = html;
        }
        
        // CSV 다운로드
        function downloadCSV() {
            if (!currentData || currentData.length === 0) {
                alert('다운로드할 데이터가 없습니다');
                return;
            }
            
            const tableName = document.getElementById('tableSelect').value;
            const csvContent = convertToCSV(currentData);
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', `${tableName}_${new Date().toISOString().slice(0,10)}.csv`);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
        
        // CSV 변환
        function convertToCSV(data) {
            if (!data || data.length === 0) return '';
            
            const columns = Object.keys(data[0]);
            let csv = columns.join(',') + '\\n';
            
            data.forEach(row => {
                const values = columns.map(col => {
                    let value = row[col] || '';
                    // CSV 이스케이프 처리
                    if (typeof value === 'string') {
                        value = value.replace(/"/g, '""');
                        if (value.includes(',') || value.includes('\\n') || value.includes('"')) {
                            value = `"${value}"`;
                        }
                    }
                    return value;
                });
                csv += values.join(',') + '\\n';
            });
            
            return csv;
        }
    </script>
</body>
</html>
"""

def get_db_connection():
    """데이터베이스 연결"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
        return conn
    except Exception as e:
        print(f"데이터베이스 연결 실패: {e}")
        return None

@app.route('/')
def index():
    """메인 페이지"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/tables')
def get_tables():
    """테이블 목록 조회"""
    conn = get_db_connection()
    if not conn:
        return jsonify([])
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return jsonify(tables)
    except Exception as e:
        return jsonify([])

@app.route('/api/table/<table_name>')
def get_table_data(table_name):
    """특정 테이블 데이터 조회"""
    conn = get_db_connection()
    if not conn:
        return jsonify([])
    
    try:
        cursor = conn.cursor()
        # 보안: 테이블명 검증
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            return jsonify({"error": "존재하지 않는 테이블입니다"})
        
        # 데이터 조회 (최대 1000개 행)
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 1000")
        rows = cursor.fetchall()
        
        # Row 객체를 딕셔너리로 변환
        data = [dict(row) for row in rows]
        conn.close()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/stats')
def get_stats():
    """데이터베이스 통계"""
    conn = get_db_connection()
    if not conn:
        return jsonify({})
    
    try:
        cursor = conn.cursor()
        stats = {}
        
        # 사용자 수
        try:
            cursor.execute("SELECT COUNT(*) FROM users")
            stats['users'] = cursor.fetchone()[0]
        except:
            stats['users'] = 0
        
        # 세션 수
        try:
            cursor.execute("SELECT COUNT(*) FROM chat_sessions")
            stats['sessions'] = cursor.fetchone()[0]
        except:
            stats['sessions'] = 0
        
        # 메시지 수
        try:
            cursor.execute("SELECT COUNT(*) FROM chat_messages")
            stats['messages'] = cursor.fetchone()[0]
        except:
            stats['messages'] = 0
        
        # 오늘 메시지 수
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("SELECT COUNT(*) FROM chat_messages WHERE timestamp LIKE ?", (f'{today}%',))
            stats['today'] = cursor.fetchone()[0]
        except:
            stats['today'] = 0
        
        conn.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({})

@app.route('/api/query', methods=['POST'])
def execute_query():
    """사용자 정의 쿼리 실행"""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"error": "쿼리가 비어있습니다"})
    
    # 보안: SELECT 쿼리만 허용
    if not query.upper().startswith('SELECT'):
        return jsonify({"error": "보안상 SELECT 쿼리만 허용됩니다"})
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "데이터베이스 연결 실패"})
    
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Row 객체를 딕셔너리로 변환
        result_data = [dict(row) for row in rows]
        conn.close()
        return jsonify({"data": result_data})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    print("🚀 NOA 데이터베이스 뷰어 시작")
    print(f"📊 데이터베이스: {DB_PATH}")
    print("🌐 웹브라우저에서 http://localhost:5000 접속")
    print("⏹️  종료하려면 Ctrl+C 를 누르세요")
    app.run(debug=True, host='0.0.0.0', port=5000) 