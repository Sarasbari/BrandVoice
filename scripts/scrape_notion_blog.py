import os
import time
import urllib.parse
import datetime
import requests
from bs4 import BeautifulSoup

# Configuration
BASE_URL = "https://www.notion.so/blog"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
DATA_DIR = os.path.join("data", "raw", "blog")
ERROR_LOG_PATH = os.path.join("data", "raw", "scrape_errors.log")
RATE_LIMIT_DELAY = 1.5  # seconds

def log_error(url, message):
    os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] URL: {url} | Error: {message}\n")

def collect_article_urls():
    print("Collecting article URLs from Notion blog index...")
    article_urls = set()
    page_num = 1
    
    while True:
        # Construct URL: page 1 is BASE_URL, subsequent pages are BASE_URL/page/N
        if page_num == 1:
            url = BASE_URL
        else:
            url = f"{BASE_URL}/page/{page_num}"
            
        print(f"Fetching index page {page_num}: {url}")
        try:
            # We wait 1.5 seconds between page index requests to respect rate limiting
            if page_num > 1:
                time.sleep(RATE_LIMIT_DELAY)
                
            response = requests.get(url, headers=HEADERS, timeout=15)
            
            if response.status_code == 404:
                print(f"Reached end of pagination (Page {page_num} returned 404).")
                break
                
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.find_all("a")
            page_articles = set()
            
            for link in links:
                href = link.get("href")
                if not href:
                    continue
                
                # Normalize URL
                full_url = urllib.parse.urljoin(BASE_URL, href)
                parsed = urllib.parse.urlparse(full_url)
                
                # Check if it is a blog article
                # Path must start with /blog/ and not be a category/topic/author page
                path = parsed.path.rstrip('/')
                
                if (path.startswith("/blog") and 
                    len(path.split("/")) == 3 and  # e.g., /blog/slug (split: '', 'blog', 'slug')
                    not path.endswith("/blog") and
                    not path.startswith("/blog/topic") and
                    not path.startswith("/blog/page") and
                    not path.startswith("/blog/author")):
                    
                    page_articles.add(full_url)
            
            if not page_articles:
                print(f"No articles found on Page {page_num}. Ending index collection.")
                break
                
            # If all articles on this page are already in our list, we might have reached the end
            new_articles = page_articles - article_urls
            if not new_articles:
                print(f"No new articles found on Page {page_num}. Ending index collection.")
                break
                
            article_urls.update(page_articles)
            print(f"Found {len(page_articles)} articles on Page {page_num} ({len(new_articles)} new).")
            page_num += 1
            
        except Exception as e:
            log_error(url, f"Failed to retrieve index page: {e}")
            print(f"Error retrieving page {page_num}. Checked error log. Ending collection.")
            break
            
    return sorted(list(article_urls))

def scrape_article(url):
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # 1. Title Extraction
    title = ""
    # Try meta tags first
    title_meta = soup.find("meta", property="og:title")
    if title_meta:
        title = title_meta.get("content", "")
    if not title:
        title = soup.title.string if soup.title else ""
    # Clean up common title suffixes
    title = title.replace(" – Notion Blog", "").replace(" | Notion", "").strip()
    
    # 2. Date Extraction
    published_date = ""
    date_meta = soup.find("meta", property="article:published_time")
    if date_meta:
        published_date = date_meta.get("content", "")
    if not published_date:
        date_itemprop = soup.find("meta", itemprop="datePublished")
        if date_itemprop:
            published_date = date_itemprop.get("content", "")
            
    # 3. Body Text Extraction
    # Strategy:
    # Look for divs with classes containing 'contentfulRichText_bodyLimit'
    body_limit_divs = soup.find_all(class_=lambda x: x and "contentfulRichText_bodyLimit" in x)
    
    paragraphs = []
    for div in body_limit_divs:
        # Skip divs that contain images, SVGs, buttons, or videos
        if div.find(["img", "svg", "button", "video"]):
            continue
        text = div.get_text().strip()
        if text:
            paragraphs.append(text)
            
    # Fallback to general paragraph/heading tags under <article> if no bodyLimit matches
    if not paragraphs:
        article_tag = soup.find("article")
        if article_tag:
            # Find all relevant text elements inside the main article
            for element in article_tag.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
                # Exclude elements containing interactive widgets
                if element.find_parent(class_=lambda x: x and ("video" in x.lower() or "widget" in x.lower() or "button" in x.lower())):
                    continue
                text = element.get_text().strip()
                # Exclude typical video placeholder warning messages
                if "ad blocker" in text.lower() or "watch it on youtube" in text.lower():
                    continue
                if text and text not in paragraphs:
                    paragraphs.append(text)
                    
    body_text = "\n\n".join(paragraphs)
    
    return {
        "title": title,
        "published_date": published_date,
        "url": url,
        "body_text": body_text
    }

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Stage 1: Collect URLs
    article_urls = collect_article_urls()
    total_articles = len(article_urls)
    print(f"Total articles found to scrape: {total_articles}")
    
    if total_articles == 0:
        print("No articles to scrape. Exiting.")
        return
        
    saved_count = 0
    newly_saved_count = 0
    
    # Stage 2: Scrape articles
    for index, url in enumerate(article_urls):
        slug = url.rstrip('/').split('/')[-1]
        file_path = os.path.join(DATA_DIR, f"{slug}.txt")
        display_index = index + 1
        
        # Check if already saved (resumable)
        if os.path.exists(file_path):
            saved_count += 1
            print(f"Scraped {display_index}/{total_articles}: {slug}")
            continue
            
        # Rate limiting delay
        time.sleep(RATE_LIMIT_DELAY)
        
        try:
            data = scrape_article(url)
            
            # Write to plain text file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Title: {data['title']}\n")
                f.write(f"Published Date: {data['published_date']}\n")
                f.write(f"URL: {data['url']}\n\n")
                f.write(data['body_text'])
                
            saved_count += 1
            newly_saved_count += 1
            print(f"Scraped {display_index}/{total_articles}: {slug}")
            
        except Exception as e:
            log_error(url, str(e))
            # Even on failure, we print the progress step to keep it readable, showing error
            print(f"Failed {display_index}/{total_articles}: {slug} (Logged error)")
            
    print("\n" + "="*40)
    print(f"Scraping Completed.")
    print(f"Total articles in blog: {total_articles}")
    print(f"Total articles saved: {saved_count}")
    print(f"Articles scraped in this run: {newly_saved_count}")
    print("="*40)

if __name__ == "__main__":
    main()
