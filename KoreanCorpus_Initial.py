# -*- coding: utf-8 -*-
import urllib.request
import urllib.parse
import html
import re
 
def getPageList(url, output):
# getPageList 함수
# url주소에 있는 위키백과 문서 목록 페이지를 파싱해 목록을 추려냅니다.
# 그 결과를 output 파일에 씁니다.
    while True:
        print("Processing: %s" % url)
        try:
            f = urllib.request.urlopen(url)
        except:
# 웹페이지를 가져올 수 없을땐 에러를 띄웁니다
# 인터넷 연결을 확인해보거나, URL 주소를 확인해보세요.
            raise("% error" % url)
            return None
        data = f.read().decode('utf-8')
# 읽은 HTML소스 코드를 utf-8로 해석합니다.
# 위키백과 사이트는 utf-8을 쓰기때문에!
        match = re.search("""<ul class="mw-allpages-chunk">(.+?)</ul>""", data, re.DOTALL | re.UNICODE)
        ws = re.findall("""<a\s+href="([^"]+)"([^>]*)>([^<]*)</a>""", match.group(1))
        for w in ws:
            if re.search("""class="mw-redirect""", w[1]): continue
# class="mw-redirect"가 있다면 리다이렉트 문서이므로 패스.
            output.write("%s\t%s\n" % (w[0], html.unescape(w[2])))
# 아닐 경우 해당 페이지의 URL과 페이지 제목을 output에 씁니다.
# 이때 페이지 제목의 HTML Entity가 이스케이프되어있을 수으므로
# html.unescape로 언이스케이프해줍니다.
        nextPage = re.search("""<a\s+href="([^"]+)"[^>]*>다음 문서""", data, re.DOTALL | re.UNICODE)
# 다음 페이지가 있는지 추출합니다.
        if nextPage: url = urllib.parse.urljoin(url, html.unescape(nextPage.group(1)))
# 있다면 url을 업데이트하고 반복.
        else: return
# 없다면 끝.
         
 
file = open('list.txt', 'wt', encoding='utf-8')
getPageList("https://ko.wikipedia.org/wiki/%ED%8A%B9%EC%88%98:%EB%AA%A8%EB%93%A0%EB%AC%B8%EC%84%9C", file)
file.close()
