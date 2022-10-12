import requests
from bs4 import BeautifulSoup
import csv
import os 
import urllib.request

def main(a,b):
	#used curl initially 
	url = "https://archive.org/details/classic_tv_commercials?query=tv+commercials&and%5B%5D=mediatype%3A%22movies%22&sort=&page="

	#no duplicates
	video_links = set()
	count = 0

	for pagenumber in range(a,b):
		print("Crawling page " + str(pagenumber) + "...")
		url = url + str(pagenumber)
		try:
			r = requests.get(url, timeout = 10)
			soup = BeautifulSoup(r.content, 'html.parser')
			
		except requests.RequestException as e:
			print(str(e))

		divs = soup.find('div', class_ = "results")
		print(len(divs))
		for itemai in divs.find_all('div', class_="item-ia"):
			for C234 in itemai.find_all('div', class_="C234"):
				for item in C234.find_all('div', class_="item-ttl"):
					for atag in item.find_all('a'):
						video_links.add(atag["href"])
						count +=1
		print(count)

	print("Total videos: " + str(count))

	print("Writing links to csv...")
	with open('../data/videolinks.csv','a') as f:
	    for line in video_links:
	        f.write('https://archive.org'+str(line))
	        f.write('\n')



if __name__ == '__main__':
	for i in range(1,2,2):
		main(i,i+2)
	#v = 'https://archive.org'+str(v)
	#v = 'https://ia903108.us.archive.org/1/items/commercials_clothing/commercials_clothing.mp4'
	#urllib.request.urlretrieve(v, 'video_name.mp4')


