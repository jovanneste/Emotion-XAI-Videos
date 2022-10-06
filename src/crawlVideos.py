import requests
from bs4 import BeautifulSoup

url = "https://archive.org/details/classic_tv_commercials?query=tv+commercials&and%5B%5D=mediatype%3A%22movies%22&sort=&page="


class_list = []
count = 0

for pagenumber in range(1,4):
	print("Crawling page " + str(pagenumber) + "...")
	url = url + str(pagenumber)
	r = requests.get(url)
	soup = BeautifulSoup(r.content, 'html5lib')
	divs = soup.find('div', class_ = "results")

	for divtag in divs.find_all('div', class_="item-ia"):
		count +=1

print(count)
#tags = {tag.name for tag in soup.find_all()}
