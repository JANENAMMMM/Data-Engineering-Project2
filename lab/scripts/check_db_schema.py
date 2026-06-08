import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "pipeline"))
from dotenv import load_dotenv; load_dotenv(_PROJECT_ROOT / ".env")
from libsql import connect
from src.config import DB_URL, DB_ACCESS_TOKEN

conn = connect(DB_URL, auth_token=DB_ACCESS_TOKEN, _uri=True)

print('=== 실제 CREATE TABLE DDL ===')
rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='fashion_items'").fetchall()
print(rows[0][0] if rows else '테이블 없음')

print('\n=== 인덱스 ===')
for r in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='fashion_items'").fetchall():
    print(r)

print('\n=== 같은 file_id에 여러 row 존재 예시 ===')
dup = conn.execute("SELECT file_id, COUNT(*) as cnt FROM fashion_items GROUP BY file_id HAVING cnt > 1 LIMIT 5").fetchall()
print(dup if dup else '없음 (현재 데이터 기준)')

print('\n=== 전체 row 수 ===')
print(conn.execute('SELECT COUNT(*) FROM fashion_items').fetchone()[0])

print('\n=== 컬럼 목록 (PRAGMA) ===')
for r in conn.execute("PRAGMA table_info('fashion_items')").fetchall():
    print(f"  {r[0]:2d} {r[1]:<25} {r[2]:<10} PK={r[5]}")

conn.close()
