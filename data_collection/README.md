# Firecrawl Crawler for Pittsburgh/CMU Knowledge Base

A web scraping tool that collects data about Pittsburgh and CMU using the Firecrawl API.

## Prerequisites

- Python 3.7+
- Firecrawl API key (get from [firecrawl.dev](https://firecrawl.dev))

## Setup

1. Install required packages:
```bash
pip install requests
```

2. Set your API key as environment variable:

**Windows CMD:**
```cmd
set FIRECRAWL_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export FIRECRAWL_API_KEY="your_api_key_here"
```

## Usage

Run the crawler:
```bash
python firecrawl_crawler.py
```

The script will automatically:
- Crawl all configured websites
- Extract content from HTML pages and PDFs
- Save progress to resume if interrupted
- Clean and process the data

## Output

- `data/raw/*.jsonl` - Individual files for each source
- `data/pittsburgh_cmu_knowledge_base.jsonl` - Combined knowledge base
- `data/failed_urls.txt` - List of URLs that failed to scrape
- `data/raw/scraping_log.txt` - Detailed execution log
- `progress.json` - Progress tracker for resuming

## Configuration

### Crawler/Scraper Parameters

Edit `WEBSITE_CONFIGS` for multi-page crawling:

```python
{
    "url": "https://example.com",           # Starting URL to crawl
    "max_depth": 2,                         # How many levels deep to crawl (1-3 recommended)
    "limit": 50,                            # Maximum number of pages
}
```

**Parameter Details:**
- `max_depth`: Controls crawl depth. `1` = only the starting URL, `2` = starting URL + direct links, `3` = two levels deep
- `limit`: Prevents excessive scraping. Set higher for large sites (100+), lower for small sites (10-20)

Edit `SINGLE_PAGE_URLS` for individual pages (no crawling):

```python
"category_name": [
    "https://example.com/page1",
    "https://example.com/page2.pdf"
]
```