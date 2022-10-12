import requests
from bs4 import BeautifulSoup


url = "https://archive.org/details/abcearly1970scommercials"

	
r = requests.get(url, timeout = 10)
soup = BeautifulSoup(r.content, 'html.parser')
			
		
links = soup.find_all('a', class_='stealth')

for link in links:
	if(str(link['href']).endswith('mp4')):
		if float(str(link['title'])[:-1]) < 150:
			print(link['href'])
	
	

