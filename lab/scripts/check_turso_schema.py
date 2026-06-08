import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "pipeline"))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from libsql import connect
from src.config import DB_URL, DB_ACCESS_TOKEN

conn = connect(DB_URL, auth_token=DB_ACCESS_TOKEN, _uri=True)

# 모든 테이블 목록
tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()

print(f"=== 테이블 목록 ({len(tables)}개) ===")
for (name,) in tables:
    print(f"  {name}")

# 각 테이블의 DDL + row 수
print("\n=== 테이블별 DDL 및 row 수 ===")
for (name,) in tables:
    ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    count = conn.execute(f"SELECT COUNT(*) FROM \"{name}\"").fetchone()[0]
    print(f"\n-- {name} ({count} rows) --")
    print(ddl[0] if ddl else "(DDL 없음)")

# 인덱스 목록
print("\n=== 인덱스 목록 ===")
indexes = conn.execute(
    "SELECT tbl_name, name, sql FROM sqlite_master WHERE type='index' ORDER BY tbl_name"
).fetchall()
for tbl, idx_name, sql in indexes:
    print(f"  [{tbl}] {idx_name}")
    if sql:
        print(f"         {sql}")

conn.close()
