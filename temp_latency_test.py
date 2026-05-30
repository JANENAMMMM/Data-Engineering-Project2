import time
from dotenv import load_dotenv
import os
from pathlib import Path
from libsql import connect
load_dotenv(Path(r'c:/Users/janen/Documents/26-1 ewha/데이터엔지니어링/project 2/Label Studio Project/.env'))
url=os.environ['DB_URL']
token=os.environ['DB_ACCESS_TOKEN']
conn=connect(url, auth_token=token, _uri=True)
cur=conn.cursor()
for i in range(3):
    t0=time.time()
    cur.execute('SELECT COUNT(*) FROM clothing_items WHERE image_id = ? AND clothing_type = ?', (1000012,'상의'))
    res=cur.fetchone()
    print('select latency', i, time.time()-t0, res)
for i in range(3):
    t0=time.time()
    cur.execute('UPDATE clothing_items SET caption_category = ? WHERE image_id = ? AND clothing_type = ?', ('X',1000012,'상의'))
    conn.commit()
    print('update latency', i, time.time()-t0)
cur.execute('UPDATE clothing_items SET caption_category = NULL WHERE image_id = ? AND clothing_type = ?', (1000012,'상의'))
conn.commit()
conn.close()
