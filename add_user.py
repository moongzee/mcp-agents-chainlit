#!/usr/bin/env python3
# add_user.py - 새 사용자 추가 스크립트

import sqlite3
import hashlib
from datetime import datetime

DB_PATH = "chat_history.db"

def init_database():
    """데이터베이스 및 테이블 초기화"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 사용자 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 채팅 세션 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # 메시지 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ 데이터베이스가 초기화되었습니다.")

def hash_password(password: str) -> str:
    """비밀번호 해시 생성"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email: str, password: str, display_name: str = None) -> bool:
    """새 사용자 생성"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        user_id = hashlib.md5(email.encode()).hexdigest()
        password_hash = hash_password(password)
        display_name = display_name or email.split('@')[0]
        
        cursor.execute('''
            INSERT INTO users (id, username, password_hash, display_name)
            VALUES (?, ?, ?, ?)
        ''', (user_id, email, password_hash, display_name))
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def list_users():
    """모든 사용자 목록 표시"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT username, display_name, created_at
        FROM users
        ORDER BY created_at DESC
    ''')
    
    users = cursor.fetchall()
    conn.close()
    
    if users:
        print("\n=== 등록된 사용자 목록 ===")
        for username, display_name, created_at in users:
            created_date = datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M")
            print(f"• {username} ({display_name}) - {created_date}")
    else:
        print("\n등록된 사용자가 없습니다.")

def main():
    print("🔐 NOA 채팅봇 사용자 관리")
    print("=" * 30)
    
    # 먼저 데이터베이스 초기화
    init_database()
    
    while True:
        print("\n1. 새 사용자 추가")
        print("2. 사용자 목록 보기")
        print("3. 종료")
        
        choice = input("\n선택하세요 (1-3): ").strip()
        
        if choice == "1":
            print("\n새 사용자 정보를 입력하세요:")
            email = input("이메일: ").strip()
            password = input("비밀번호: ").strip()
            display_name = input("표시 이름 (선택사항): ").strip()
            
            if not email or not password:
                print("❌ 이메일과 비밀번호는 필수입니다.")
                continue
            
            if '@' not in email:
                print("❌ 올바른 이메일 형식이 아닙니다.")
                continue
            
            display_name = display_name if display_name else email.split('@')[0]
            
            if create_user(email, password, display_name):
                print(f"✅ 사용자 '{email}' ({display_name})이(가) 성공적으로 추가되었습니다!")
                print(f"💡 로그인 시 이메일: {email}, 비밀번호: {password}")
            else:
                print(f"❌ 이메일 '{email}'이(가) 이미 존재합니다.")
        
        elif choice == "2":
            list_users()
        
        elif choice == "3":
            print("프로그램을 종료합니다.")
            break
        
        else:
            print("❌ 잘못된 선택입니다. 1-3 중에서 선택해주세요.")

if __name__ == "__main__":
    main()