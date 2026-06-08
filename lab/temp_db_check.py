from dotenv import load_dotenv
import os
from pathlib import Path
from libsql import connect
load_dotenv(Path(r'c:\Users\janen\Documents\26-1 ewha\데이터엔지니어링\project 2\Label Studio Project\.env'))
url=os.environ.get('DB_URL')
token=os.environ.get('DB_ACCESS_TOKEN')
print('url=', url)
print('token=', 'set' if token else 'missing')
conn=connect(url, auth_token=token, _uri=True)
cur=conn.cursor()
cur.execute('SELECT 1')
print(cur.fetchone())
conn.close()
