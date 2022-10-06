import requests
from bs4 import BeautifulSoup


url = "https://archive.org/details/classic_tv_commercials?query=tv+commercials&sort=&and[]=mediatype%3A%22movies%22"
 

r = requests.get(url)
 

soup = BeautifulSoup(r.content, 'html5lib')


video_tags = soup.find(class_="item-ia hov")

print(video_tags)
