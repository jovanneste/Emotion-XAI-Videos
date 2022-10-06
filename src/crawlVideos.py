import requests
from bs4 import BeautifulSoup

for pagenumber in range(1,2):
	url = "https://archive.org/details/classic_tv_commercials?query=tv+commercials&and%5B%5D=mediatype%3A%22movies%22&sort=&page="+pagenumber

c=0
r = requests.get(url)
 

soup = BeautifulSoup(r.content, 'html5lib')

class_list = []

divs = soup.find('div', class_ = "results")


for divtag in divs.find_all('div', class_="item-ia"):
	c+=1
	print(divtag["data-id"])

print(c)



#tags = {tag.name for tag in soup.find_all()}
