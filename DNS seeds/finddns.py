import requests
from bs4 import BeautifulSoup
import csv
import time

url = "https://wiz.biz/bitcoin/seed/1"
ipv4_counts = {}

while True:
    response = requests.get(url)
    html_content = response.content

    soup = BeautifulSoup(html_content, 'html.parser')

    ipv4_tags = soup.find_all('pre')[0].find_all('br')

    with open('ipv4_addresses.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['IPv4 Address', 'Information', 'Count'])
        for i in range(0, len(ipv4_tags), 2):
            ipv4_address = ipv4_tags[i].previous_sibling.strip()
            ipv4_info = ipv4_tags[i+1].previous_sibling.strip()
            count = ipv4_counts.get(ipv4_address, 0) + 1
            ipv4_counts[ipv4_address] = count
            writer.writerow([ipv4_address, ipv4_info, count])
    
    time.sleep(60)  
