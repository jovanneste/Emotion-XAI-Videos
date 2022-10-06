import requests
from bs4 import BeautifulSoup
import csv


url = "https://archive.org/details/classic_tv_commercials?query=tv+commercials&and%5B%5D=mediatype%3A%22movies%22&sort=&page="

#no duplicates
video_links = set()
count = 0

for pagenumber in range(1,100):
	print("Crawling page " + str(pagenumber) + "...")
	url = url + str(pagenumber)
	r = requests.get(url)
	soup = BeautifulSoup(r.content, 'html5lib')
	divs = soup.find('div', class_ = "results")

	for itemai in divs.find_all('div', class_="item-ia"):
		for C234 in itemai.find_all('div', class_="C234"):
			for item in C234.find_all('div', class_="item-ttl"):
				for atag in item.find_all('a'):
					video_links.add(atag["href"])
					count +=1
					print(count)

print("Total videos: " + str(count))

print("Writing links to csv...")
with open('videolinks.csv','w') as f:
    for line in video_links:
        f.write(line)
        f.write('\n')