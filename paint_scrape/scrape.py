from selenium import  webdriver
import time
from bs4 import BeautifulSoup as Soup
import json
from tqdm import tqdm

driver = webdriver.Chrome("C:\chromedriver.exe")
url = "https://zbiory.mnk.pl/pl/katalog?filter=JTdCJTIyb25seUltYWdlJTIyOiU3QiUyMm5hbWUlMjI6JTIyb25seUltYWdlJTIyLCUyMnZhbHVlJTIyOnRydWUsJTIyZGVzYyUyMjpudWxsLCUyMmFjdGl2ZSUyMjp0cnVlJTdELCUyMnR5cGVzJTIyOiU3QiUyMm5hbWUlMjI6JTIydHlwZXMlMjIsJTIydmFsdWUlMjI6JTVCJTdCJTIyaWQlMjI6MTAwNDg5LCUyMm5hbWUlMjI6JTIyb2JyYXolMjIlN0QlNUQsJTIyZGVzYyUyMjpudWxsLCUyMmFjdGl2ZSUyMjp0cnVlJTdEJTdE&view=viewList"
base_url = "https://www.zbiory.mnk.pl"

# changes link name to gain access to databses
def image_link_trick(image_link):
    return image_link.replace('/cache/multimedia_detail/','/multimedia/', 1)

def find_urls(url, first=False):
    # connecting with url + waiting for loading
    if first == True:
        driver.get(url)
    time.sleep(4)

    # scrolling so if not loaded or connection is slow
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)

    page = driver.page_source
    soup = Soup(page, "lxml")

    # returns links to images
    links = [image_link_trick(base_url+link['src']) for link in soup.select('img', {'class':'img-list'}) if "multimedia" in str(link) ]
    print(links)
    if len(links) < 20:
        print("Links", len(links))

    # returns titles
    titles = [title_div.text for title_div in soup.select('.addClipboardShow-Desktop .item-title')]
    if len(titles) < 20:
        print("Titles", len(titles))

    return dict(zip(titles, links))

def next_page():
    element = driver.find_element_by_xpath('//*[@id="app-body"]/app-root/div/footer/app-footer/div[1]/app-footer-up/div/app-footer-pagination-box/app-pagination-box/div/div/div[1]/div/div[3]/div/div[3]/div[1]')
    driver.execute_script("arguments[0].click();", element)
    time.sleep(2)
    return driver.current_url


find_urls(url, True)

data = {}
for _ in tqdm(range(0, 107)):
    data.update(find_urls(next_page()))



print("Scraping completed!")
with open('paintings.json', 'w', encoding="utf-8" ) as file:
    json.dump(data, file, ensure_ascii=False)
