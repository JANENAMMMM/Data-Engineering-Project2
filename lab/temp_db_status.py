from dotenv import load_dotenv
import os
from pathlib import Path
from libsql import connect
load_dotenv(Path(r'c:\Users\janen\Documents\26-1 ewha\데이터엔지니어링\project 2\Label Studio Project\.env'))
url = os.environ['DB_URL']
token = os.environ['DB_ACCESS_TOKEN']
conn = connect(url, auth_token=token, _uri=True)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [row[0] for row in cur.fetchall()]
print('tables=')
for t in tables:
    print('  ', t)
print('\ncounts=')
for t in tables:
    cur.execute(f'SELECT COUNT(*) FROM {t}')
    print(f'  {t}:', cur.fetchone()[0])
conn.close()
