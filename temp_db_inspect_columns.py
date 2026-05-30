from dotenv import load_dotenv
import os
from pathlib import Path
from libsql import connect
load_dotenv(Path(r'c:\Users\janen\Documents\26-1 ewha\데이터엔지니어링\project 2\Label Studio Project\.env'))
url=os.environ['DB_URL']
token=os.environ['DB_ACCESS_TOKEN']
conn=connect(url, auth_token=token, _uri=True)
cur=conn.cursor()
cur.execute("PRAGMA table_info(clothing_items)")
for r in cur.fetchall():
    print(r)
conn.close()
