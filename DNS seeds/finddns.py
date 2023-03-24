import requests
from bs4 import BeautifulSoup
import csv
import time
from datetime import datetime

url = "https://wiz.biz/bitcoin/seed/1"
ipv4_counts = {}

while True:
    response = requests.get(url)
    html_content = response.content

    soup = BeautifulSoup(html_content, 'html.parser')

    # <div>As of Thu Mar 23 11:54:01 UTC 2023</div>
    timestamp_str = soup.find_all('div')[0].text.strip()
    timestamp = datetime.strptime(timestamp_str, 'As of %a %b %d %H:%M:%S %Z %Y')

    ipv4_tags = soup.find_all('pre')[0].find_all('br')

    with open('ipv4_addresses.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['IPv4 Address', 'Information', 'Count', 'Timestamp',])
        for i in range(0, len(ipv4_tags), 2):
            ipv4_address = ipv4_tags[i].previous_sibling.strip()
            ipv4_info = ipv4_tags[i+1].previous_sibling.strip()
            count = ipv4_counts.get(ipv4_address, 0) + 1
            ipv4_counts[ipv4_address] = count
            writer.writerow([ipv4_address, ipv4_info, count, timestamp.strftime('%d:%H:%M:%S')])

    time.sleep(60)  
