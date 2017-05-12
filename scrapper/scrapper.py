import sys
import re
import urlparse
from webscraping import download, xpath

resource = sys.argv[1]
outputfile = sys.argv[2]
archive = 'http://web.archive.org'
f = open(resource, 'r')
lines = f.readlines()
f.close()

# name of URL to be crawled
domain = lines[0].strip()

# timestamp of the snapshots
snapshots = lines[1:]
D = download.Download()
#seen_urls = set()

# get the content of the URL given
def download_content(outputfile, seen_urls):
	f = open(outputfile, 'a')
	for url in seen_urls:
		page_html = D.get(url)
		#print page_html
		#ba_cntebt_text introFirst
		contents = xpath.search(page_html, '//div[@id="ba_content"]//div/text()')
		#contents = xpath.search(page_html, '//div[@class="ba_cntebt_text_introFirst"]/div/text() | //div[@class="mainText"]/div/text()')
		f.write(url+'\n')
		#for content in contents:
		#	f.write(content)
		if contents == None:	
			contents = xpath.search(page_html, '//div[@class="mainpopup"]//div/text()')
                #contents = xpath.search(page_html, '//div[@class="ba_cntebt_text_introFirst"]/div/text() | //div[@class="mainText"]/div/text()')
                #f.write(url+'\n')
                for content in contents:
                        f.write(content)
		#break
	f.close()	

# get the external URLs to be crawled
def get_external_URL(page_html):
	seen_urls = set()
	urls =  xpath.search(page_html, '//section[@class="maincontainer"]//a/@href')
       	for url in urls:
            #print link
	    url = url.replace(archive,'')
            if url not in seen_urls:
               #num_new_articles += 1
               seen_urls.add(url)
	return seen_urls


for snapshot in snapshots:
	#seen_urls = set()
	print snapshot
	url_to_be_crawled = snapshot.strip()+'/'+domain
	#print urlparse.urljoin(snapshot.strip(),domain)
	page_html = D.archive_get(url_to_be_crawled)
	#print page_html
	#print xpath.get(html, '//title')
	seen_urls = get_external_URL(page_html)
	outputfile = outputfile + snapshot + '.txt'
	download_content(outputfile, seen_urls)
		
