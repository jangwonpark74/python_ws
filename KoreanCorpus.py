# -*- coding: utf-8 -*-
import urllib.request
import urllib.parse
import html
import re
import os.path
import threading
import queue
 
def deleteTag(x):
    if x == None: return ""
    else: return re.sub("<[^>]*>", "", x)
 
def getPage(url):
    try:
        f = urllib.request.urlopen(url)
    except:
        print("% error" % url)
        return None
    data = f.read().decode('utf-8')
    match = re.search("""<div id="mw-content-text" lang="ko" dir="ltr" class="mw-content-ltr">(.*?)<div class="printfooter">""", data, re.DOTALL | re.UNICODE)
    if not match: return None
    s = re.sub("""(\s| )+""", " ", match.group(1))
    s = re.sub("""<span class="mw-editsection-bracket">(.+?)</span>""", "", s)
    s = re.sub("""<a href="[^"]*edit[^"]*"[^>]*>편집</a>""", "", s)
    s = re.sub("""<(h[1-6]|div|li|p|td|th)(\s[^>]*)?>""", "\n", s)
    s = html.unescape(deleteTag(s))
    s = re.sub("""(\n[\r\t ]*)+""", " .\n", s)
    return s
 
lock = threading.Lock()
# 출력을 위해 lock을 만듭니다.
 
def do_work(item):
    fpath = 'doc\\%s.txt'% item[1]
    if os.path.isfile(fpath): return
    try:
        with lock:
            print(item[0], item[1])
# 출력시는 항상 락을 걸고 출력
    except:
        None
    try:
        page = getPage('https://ko.wikipedia.org' + item[0])
        file = open(fpath, 'wt', encoding='euc-kr', errors='replace')
        file.write(page)
        file.close()
    except:
        with lock:
            print("Error", item[0], item[1])
 
def worker():
    while True:
        item = q.get()
# 큐에 작업할 녀석이 있다면 가져옵니다.
        do_work(item)
# 가져온 작업목록을 수행
        q.task_done()
# 작업이 끝났다고 큐에다가 알려줌
 
q = queue.Queue()
for i in range(16):
     t = threading.Thread(target=worker)
     t.daemon = True
     t.start()
# 스레드 16개를 만들어서 시작시킵니다.
 
doclist = open('list.txt', 'rt', encoding='utf-8')
 
for l in doclist:
    t = l.replace('\n', '').split('\t')
    t[1] = re.sub("""[<>:"/\\|?*]""", '', t[1])
    q.put(t)
# 가져올 사이트 목록을 차례대로 큐에 넣어줍니다.
 
q.join()
# 큐의 모든 작업이 끝날때까지 대기
 
doclist.close()
