import requests
from bs4 import BeautifulSoup
import time
import os


# Seed URL and settings
seed_url = "https://www.geeksforgeeks.org/ai-ml-ds/?ref=home-articlecards"
max_pages = 300 # Amount of Documents
delay_between_requests = 1  # seconds
domain = "https://www.geeksforgeeks.org" # my main domain

# Dir to save html files content as txt file
if not os.path.exists("Documents"):
    os.makedirs("Documents")

# Tracking visited pages = to calculate the length / count of docs, we use set instead of list beacuse of unqiueness
visited_urls = set()

# Getting the content from url and checking the fetching status
def crawl_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
    except requests.exceptions.RequestException as e: 
        print(f"Request failed: {e}")
        return [] # return empty list to contniue to next url

    # Parse page content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract main content and save it
    save_page_content(url, soup)

    # Find all links within the same domain:(https://www.zenrows.com/blog/web-crawler-python#initial-crawling-script)
    link_elements = soup.select("a[href]")
    links = []

    for link_element in link_elements: # should handle absolute and relative lin ks
        url = link_element['href']
        # chck relative url and make it as a unique url
        if url.startswith("/"):
            url = domain + url
        # checking my url list 
        if url.startswith(domain) and url not in visited_urls:
            links.append(url)
    return links 


def save_page_content(url, soup):
    # Collect content from <h1>, <h2>, <p> tags as a sample structure for main content
    content = []
    for tag in soup.find_all(['h1', 'h2', 'p']):
        content.append(tag.get_text(strip=True)) #Ref=https://www.tutorialspoint.com/beautiful_soup/beautiful_soup_get_text_method.htm

    # Join all content into a single text and save as a .txt file
    text_content = "\n".join(content)  #Ref:https://stackoverflow.com/questions/14560863/python-join-with-newline
    filename = os.path.join("Documents", f"page{len(visited_urls)}.txt")
    with open(filename, 'w', encoding="utf-8") as file: #Ref: https://stackoverflow.com/questions/46882521/python-scrape-a-part-of-source-code-and-save-it-as-html
        file.write(f"URL: {url}\n")  # Include URL for reference
        file.write(text_content)
    print(f"Saved main content of page: {url}")

def start_crawling(seed_url): 
    urls_to_crawl = [seed_url]
    while urls_to_crawl and len(visited_urls) < max_pages:
        current_url = urls_to_crawl.pop(0)
        if current_url in visited_urls:
            continue

        print(f"Crawling: {current_url}")
        visited_urls.add(current_url)
        
        # Crawl the page and get new links
        links = crawl_page(current_url)
        urls_to_crawl.extend(links) # here we cant use append even its list, bcz if we use append it will be list of list.
        
        # Delay to prevent overloading the server
        time.sleep(delay_between_requests)

# Start the crawling process
start_crawling(seed_url)