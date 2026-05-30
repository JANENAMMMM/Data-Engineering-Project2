from dotenv import load_dotenv
import os
from pathlib import Path
from libsql import connect
load_dotenv(Path(r'c:\Users\janen\Documents\26-1 ewha\데이터엔지니어링\project 2\Label Studio Project\.env'))
url=os.environ['DB_URL']
token=os.environ['DB_ACCESS_TOKEN']
conn=connect(url, auth_token=token, _uri=True)
cur=conn.cursor()
cur.execute('SELECT COUNT(*) FROM clothing_items WHERE vlm_annotatioin IS NOT NULL AND vlm_annotatioin != ""')
print('non-empty vlm_annotatioin:', cur.fetchone()[0])
cur.execute('SELECT vlm_annotatioin FROM clothing_items WHERE vlm_annotatioin IS NOT NULL AND vlm_annotatioin != "" LIMIT 5')
for row in cur.fetchall():
    print(row[0])
conn.close()
