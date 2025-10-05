"""
Pittsburgh/CMU Knowledge Base Collection - FINAL VERSION
"""

import os
import json
import time
import sys
from datetime import datetime
from typing import List, Dict, Optional, Set
import requests
import re
from urllib.parse import urlparse


class Logger:
    """Simple logger that outputs to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()


class FirecrawlCollector:
    """Handles data collection using Firecrawl API"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.firecrawl.dev/v2"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def crawl_website(self, url: str, max_depth: int = 2, limit: int = 100,
                      include_paths: Optional[List[str]] = None,
                      exclude_paths: Optional[List[str]] = None,
                      max_retries: int = 5) -> Dict:
        endpoint = f"{self.base_url}/crawl"
        
        payload = {
            "url": url,
            "maxDiscoveryDepth": max_depth,
            "limit": limit,
            "scrapeOptions": {
                "formats": ["markdown"],
                "parsers": ["pdf"],
                "includeTags": ["article", "main", "section"],
                "excludeTags": ["nav", "footer", "header", "aside", "script", "style"],
                "waitFor": 1000
            }
        }
        
        if include_paths:
            payload["includePaths"] = include_paths
        if exclude_paths:
            payload["excludePaths"] = exclude_paths
        
        print(f"Starting crawl for: {url}")
        
        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, headers=self.headers, json=payload)
                response.raise_for_status()
                
                job_data = response.json()
                job_id = job_data.get("id")
                
                if not job_id:
                    print(f"Failed to start crawl: {job_data}")
                    return {"success": False, "data": []}
                
                print(f"Crawl job started with ID: {job_id}")
                return self._wait_for_crawl_completion(job_id)
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = (2 ** attempt) * 10
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"HTTP Error: {e}")
                    return {"success": False, "data": []}
            except Exception as e:
                print(f"Error during crawl: {e}")
                return {"success": False, "data": []}
        
        print(f"Failed after {max_retries} retries")
        return {"success": False, "data": []}
    
    def _wait_for_crawl_completion(self, job_id: str, max_wait_time: int = 600) -> Dict:
        endpoint = f"{self.base_url}/crawl/{job_id}"
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            
            status_data = response.json()
            status = status_data.get("status")
            
            print(f"Job {job_id} status: {status}")
            
            if status == "completed":
                print(f"Crawl completed! Found {len(status_data.get('data', []))} pages")
                return status_data
            elif status == "failed":
                print(f"Crawl failed: {status_data}")
                return {"success": False, "data": []}
            
            time.sleep(5)
        
        print(f"Crawl timed out after {max_wait_time} seconds")
        return {"success": False, "data": []}
    
    def scrape_single_page(self, url: str, max_retries: int = 5) -> Dict:
        endpoint = f"{self.base_url}/scrape"
        
        payload = {
            "url": url,
            "formats": ["markdown"],
            "parsers": ["pdf"]
        }
        
        print(f"Scraping single page: {url}")
        
        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = (2 ** attempt) * 5
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"HTTP Error: {e}")
                    return {"success": False, "data": []}
            except Exception as e:
                print(f"Error scraping page: {e}")
                return {"success": False, "data": []}
        
        print(f"Failed after {max_retries} retries")
        return {"success": False, "data": []}


class DataProcessor:
    """Processes and saves scraped data"""
    
    @staticmethod
    def smart_clean_markdown(markdown: str, is_pdf: bool = False) -> str:
        """
        Smart cleaning that preserves useful information while removing navigation/boilerplate.
        """
        if not markdown:
            return ""
        
        # For PDFs, minimal cleaning
        if is_pdf:
            return markdown.strip()
        
        lines = markdown.split('\n')
        cleaned = []
        
        # Patterns for lines to completely skip
        skip_line_patterns = [
            r'^Search$',
            r'^Skip to content$',
            r'^Generic selectors$',
            r'^Exact matches only$',
            r'^Search in title$',
            r'^Search in content$',
            r'^Post Type Selectors$',
            r'^Main Navigation$',
            r'^\[Skip to',
            r'Sign Up$',
            r'^Receive Our Newsletters$',
            r'^BESbswy$',
            r'^Ã‚Â©\s*\d{4}',
            r'^Â©\s*Carnegie',
            r'Follow.*:$'
        ]
        
        # Patterns for navigation links (but keep event/article links)
        nav_patterns = [
            r'^\[(Join|Donate|Visit|Support|Membership|About)\]',
            r'^\[(Privacy Policy|Terms of Use|Non-Discrimination)\]',
            r'^\[(LinkedIn|Instagram|Facebook|X)\]',
            r'^\[(Accessibility|Press|Opportunities|Contact Us|Shop)\]'
        ]
        
        in_table = False
        prev_was_empty = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip truly empty lines but track them
            if not stripped:
                if not prev_was_empty:  # Keep one blank line for structure
                    cleaned.append('')
                    prev_was_empty = True
                continue
            
            prev_was_empty = False
            
            # Table detection
            if '|' in stripped and ('---' in stripped or stripped.count('|') > 2):
                in_table = True
                continue
            elif in_table and '|' not in stripped:
                in_table = False
            
            if in_table:
                continue
            
            # Skip navigation patterns
            if any(re.search(pattern, stripped, re.I) for pattern in skip_line_patterns):
                continue
            
            # Skip pure navigation but keep content links
            if any(re.search(pattern, stripped) for pattern in nav_patterns):
                continue
            
            # Skip image-only lines (but keep lines with text and images)
            if re.match(r'^!\[.*\]\(.*\)$', stripped):
                continue
            
            # Keep everything else
            cleaned.append(stripped)
        
        # Join and clean up excessive whitespace
        result = '\n'.join(cleaned)
        result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 consecutive newlines
        
        return result.strip()
    
    @staticmethod
    def extract_event_info(content: str, title: str) -> Dict:
        """Extract event metadata from content."""
        metadata = {}
        
        # Search in first portion of content
        lines = content.split('\n')
        search_content = '\n'.join(lines[:max(15, len(lines)//3)])
        
        # Date patterns
        date_patterns = [
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, title + ' ' + search_content)
            if match:
                metadata['event_date'] = match.group(0)
                break
        
        # Time patterns
        time_patterns = [
            r'\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.|AM|PM)\s*[-—]\s*\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.|AM|PM)',
            r'\d{1,2}\s*(?:a\.m\.|p\.m\.|AM|PM)\s*[-—]\s*\d{1,2}\s*(?:a\.m\.|p\.m\.|AM|PM)',
            r'\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.|AM|PM)',
            r'\d{1,2}\s*(?:a\.m\.|p\.m\.|AM|PM)',
            r'All Day'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, search_content, re.I)
            if match:
                metadata['event_time'] = match.group(0)
                break
        
        return metadata
    
    @staticmethod
    def detect_format(url: str, metadata: Dict) -> str:
        """Detect the format of the document"""
        url_lower = url.lower()
        
        # Check URL extension first
        if url_lower.endswith('.pdf'):
            return 'PDF'
        elif url_lower.endswith(('.html', '.htm')):
            return 'HTML'
        elif url_lower.endswith('.txt'):
            return 'TXT'
        
        # Check content type from metadata if available
        content_type = metadata.get('content-type', '').lower()
        if 'pdf' in content_type:
            return 'PDF'
        elif 'html' in content_type:
            return 'HTML'
        elif 'text' in content_type:
            return 'TXT'
        
        # Default to HTML for web pages
        return 'HTML'
    
    @staticmethod
    def process_crawl_results(results: Dict, category: str, source_url: str) -> tuple[List[Dict], Set[str]]:
        """
        Process and clean results.
        Returns: (processed_docs, failed_urls)
        """
        processed_docs = []
        failed_urls = set()
        
        for item in results.get("data", []):
            raw_markdown = item.get("markdown", "")
            title = item.get("metadata", {}).get("title", "")
            item_url = item.get("metadata", {}).get("url", "")
            item_metadata = item.get("metadata", {})
            
            # Check if scrape failed (empty content)
            if not raw_markdown or len(raw_markdown.strip()) < 10:
                print(f"⚠️ FAILED SCRAPE (empty content): {item_url}")
                failed_urls.add(item_url)
                continue
            
            # Detect format
            doc_format = DataProcessor.detect_format(item_url, item_metadata)
            is_pdf = doc_format == 'PDF'
            
            # Record original content length
            original_content_length = len(raw_markdown)
            
            # Clean content
            if is_pdf:
                cleaned_content = raw_markdown.strip()
                print(f"📄 Processing PDF: {title or item_url}")
            else:
                cleaned_content = DataProcessor.smart_clean_markdown(raw_markdown, is_pdf)
            
            # Extract metadata
            event_metadata = DataProcessor.extract_event_info(cleaned_content, title)
            
            doc = {
                "url": item_url,
                "title": title,
                "content": cleaned_content,
                "source_category": category,
                "source_root_url": source_url,
                "format": doc_format,
                "is_pdf": is_pdf,
                "scraped_at": datetime.now().isoformat(),
                "metadata": {
                    "original_content_length": original_content_length,
                    "cleaned_content_length": len(cleaned_content),
                    "word_count": len(cleaned_content.split()),
                    "char_count": len(cleaned_content),
                    "description": item_metadata.get("description", ""),
                    **event_metadata
                }
            }
            
            # Keep all documents, even low content ones
            processed_docs.append(doc)
            print(f"✓ Processed [{doc_format}]: {title[:60]}... (orig: {original_content_length} chars, cleaned: {len(cleaned_content)} chars)")
        
        return processed_docs, failed_urls
    
    @staticmethod
    def save_to_jsonl(data: List[Dict], filepath: str):
        """Save documents to JSONL format"""
        if not data:
            print(f"No data to save for {filepath}")
            return
            
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in data:
                json.dump(doc, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"💾 Saved {len(data)} documents to {filepath}")
    
    @staticmethod
    def save_failed_urls(failed_urls: Set[str], filepath: str):
        """Save failed URLs to a text file"""
        if not failed_urls:
            return
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Failed URLs (empty content) - {datetime.now().isoformat()}\n")
            f.write(f"# Total: {len(failed_urls)}\n\n")
            for url in sorted(failed_urls):
                f.write(f"{url}\n")
        
        print(f"⚠️ Saved {len(failed_urls)} failed URLs to {filepath}")
    
    @staticmethod
    def get_safe_filename(url: str) -> str:
        """Convert URL to safe filename"""
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        path = parsed.path.strip('/').replace('/', '_')
        
        if path.endswith('.pdf'):
            path = path[:-4]
        
        if path:
            filename = f"{domain}_{path}"
        else:
            filename = domain
        
        filename = re.sub(r'[^\w\-_]', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        filename = filename[:100]
        
        return filename


class ProgressTracker:
    """Track progress and allow resuming"""
    
    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.completed = self._load_progress()
    
    def _load_progress(self) -> set:
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(list(self.completed), f, indent=2)
    
    def is_completed(self, url: str) -> bool:
        return url in self.completed
    
    def mark_completed(self, url: str):
        self.completed.add(url)
        self.save_progress()


def main(api_key: str, output_dir: str, combined_output_file: str,
         failed_urls_file: str, website_configs: Dict, single_page_urls: Dict,
         delay_between_requests: int = 10):
    
    # Setup logging to file
    log_file = os.path.join(output_dir, "scraping_log.txt")
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    
    print(f"📝 Logging to: {log_file}")
    print(f"⏰ Started at: {datetime.now().isoformat()}\n")
    
    collector = FirecrawlCollector(api_key)
    processor = DataProcessor()
    progress = ProgressTracker("./progress.json")
    all_documents = []
    all_failed_urls = set()
    
    total_urls = sum(len(config["urls"]) for config in website_configs.values())
    total_urls += sum(len(urls) for urls in single_page_urls.values())
    processed_count = 0
    
    # Process website crawls
    for category, config in website_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing category: {category.upper()}")
        print(f"{'='*60}\n")
        
        for site_config in config["urls"]:
            url = site_config["url"]
            
            if progress.is_completed(url):
                print(f"⏭️ SKIPPING (already completed): {url}")
                processed_count += 1
                continue
            
            safe_name = processor.get_safe_filename(url)
            output_file = os.path.join(output_dir, f"{safe_name}.jsonl")
            
            processed_count += 1
            print(f"\n[{processed_count}/{total_urls}] Processing: {url}")
            print(f"Output file: {output_file}")
            
            results = collector.crawl_website(
                url=url,
                max_depth=site_config.get("max_depth", 2),
                limit=site_config.get("limit", 50),
                include_paths=site_config.get("include_paths"),
                exclude_paths=site_config.get("exclude_paths")
            )
            
            docs, failed_urls = processor.process_crawl_results(results, category, url)
            all_failed_urls.update(failed_urls)
            
            processor.save_to_jsonl(docs, output_file)
            all_documents.extend(docs)
            
            progress.mark_completed(url)
            print(f"⏱️ Waiting {delay_between_requests} seconds before next request...")
            time.sleep(delay_between_requests)
    
    # Process single pages
    for category, urls in single_page_urls.items():
        print(f"\n{'='*60}")
        print(f"Processing single pages: {category.upper()}")
        print(f"{'='*60}\n")
        
        for url in urls:
            if progress.is_completed(url):
                print(f"⏭️ SKIPPING (already completed): {url}")
                processed_count += 1
                continue
            
            safe_name = processor.get_safe_filename(url)
            output_file = os.path.join(output_dir, f"{safe_name}.jsonl")
            
            processed_count += 1
            print(f"\n[{processed_count}/{total_urls}] Processing: {url}")
            print(f"Output file: {output_file}")
            
            result = collector.scrape_single_page(url)

            if result.get("success") and isinstance(result.get("data"), dict):
                item = result["data"]
                # ensure url is present in metadata
                item.setdefault("metadata", {})
                item["metadata"].setdefault("url", url)

                docs, failed_urls = processor.process_crawl_results({"data": [item]}, category, url)
                all_failed_urls.update(failed_urls)
                processor.save_to_jsonl(docs, output_file)
                all_documents.extend(docs)
            else:
                all_failed_urls.add(url)

            progress.mark_completed(url)
            print(f"⏱️ Waiting {delay_between_requests} seconds before next request...")
            time.sleep(delay_between_requests)
    
    # Save combined file
    if all_documents:
        processor.save_to_jsonl(all_documents, combined_output_file)
    
    # Save failed URLs
    if all_failed_urls:
        processor.save_failed_urls(all_failed_urls, failed_urls_file)
    
    # Print summary
    pdf_count = sum(1 for doc in all_documents if doc.get("format") == "PDF")
    html_count = sum(1 for doc in all_documents if doc.get("format") == "HTML")
    other_count = len(all_documents) - pdf_count - html_count
    
    print(f"\n{'='*60}")
    print(f"✅ COLLECTION COMPLETE!")
    print(f"⏰ Finished at: {datetime.now().isoformat()}")
    print(f"{'='*60}")
    print(f"Total documents collected: {len(all_documents)}")
    print(f"PDFs processed: {pdf_count}")
    print(f"HTML pages processed: {html_count}")
    print(f"Other formats: {other_count}")
    print(f"Failed scrapes (empty content): {len(all_failed_urls)}")
    print(f"Individual files saved to: {output_dir}/")
    print(f"Combined knowledge base: {combined_output_file}")
    print(f"Log file: {log_file}")
    if all_failed_urls:
        print(f"Failed URLs list: {failed_urls_file}")


if __name__ == "__main__":
    API_KEY = os.getenv("FIRECRAWL_API_KEY")
    if not API_KEY:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")
    
    OUTPUT_DIR = "./data/raw"
    COMBINED_OUTPUT_FILE = "./data/pittsburgh_cmu_knowledge_base.jsonl"
    FAILED_URLS_FILE = "./data/failed_urls.txt"
    DELAY_BETWEEN_REQUESTS = 15
    
    print("\n" + "="*60)
    print("🚀 RUNNING FULL SCALE DATA COLLECTION")
    print("📄 PDF parsing enabled")
    print("✓ Keeping all content (no filtering)")
    print("⚠️ Tracking failed scrapes")
    print("="*60 + "\n")
    
    WEBSITE_CONFIGS = {
        "general_info": {
            "urls": [
                {
                    "url": "https://www.cmu.edu/about/",
                    "max_depth": 3,
                    "limit": 40,
                },
                {
                    "url": "https://www.pittsburghpa.gov/",
                    "max_depth": 2,
                    "limit": 30,
                },
                {
                    "url": "https://www.visitpittsburgh.com/",
                    "max_depth": 2,
                    "limit": 50,
                },
                {
                    "url": "https://www.britannica.com/place/Pittsburgh",
                    "max_depth": 1,
                    "limit": 5
                }
            ]
        },
        "city_regulations": {
            "urls": [
                {
                    "url": "https://pittsburghpa.gov/finance/tax-forms",
                    "max_depth": 2,
                    "limit": 30,
                }
            ]
        },
        "events_pittsburgh": {
            "urls": [
                {
                    "url": "https://pittsburgh.events/",
                    "max_depth": 3,
                    "limit": 100,
                },
                {
                    "url": "https://downtownpittsburgh.com/events/",
                    "max_depth": 3,
                    "limit": 80,
                },
                {
                    "url": "https://www.pghcitypaper.com/pittsburgh/EventSearch",
                    "max_depth": 2,
                    "limit": 50
                }
            ]
        },
        "events_cmu": {
            "urls": [
                {
                    "url": "https://events.cmu.edu/",
                    "max_depth": 3,
                    "limit": 100,
                },
                {
                    "url": "https://www.cmu.edu/engage/alumni/events/campus/",
                    "max_depth": 2,
                    "limit": 40
                }
            ]
        },
        "music_culture": {
            "urls": [
                {
                    "url": "https://www.pittsburghsymphony.org/",
                    "max_depth": 2,
                    "limit": 40,
                },
                {
                    "url": "https://pittsburghopera.org/",
                    "max_depth": 2,
                    "limit": 30,
                },
                {
                    "url": "https://trustarts.org/",
                    "max_depth": 2,
                    "limit": 40,
                },
                {
                    "url": "https://carnegiemuseums.org/",
                    "max_depth": 3,
                    "limit": 80,
                },
                {
                    "url": "https://www.heinzhistorycenter.org/",
                    "max_depth": 2,
                    "limit": 50,
                },
                {
                    "url": "https://www.thefrickpittsburgh.org/",
                    "max_depth": 2,
                    "limit": 30,
                }
            ]
        },
        "food_events": {
            "urls": [
                {
                    "url": "https://www.visitpittsburgh.com/events-festivals/food-festivals/",
                    "max_depth": 2,
                    "limit": 30
                },
                {
                    "url": "https://www.picklesburgh.com/",
                    "max_depth": 2,
                    "limit": 20
                },
                {
                    "url": "https://www.pghtacofest.com/",
                    "max_depth": 2,
                    "limit": 15
                },
                {
                    "url": "https://pittsburghrestaurantweek.com/",
                    "max_depth": 2,
                    "limit": 20
                },
                {
                    "url": "https://littleitalydays.com/",
                    "max_depth": 2,
                    "limit": 15
                },
                {
                    "url": "https://bananasplitfest.com/",
                    "max_depth": 2,
                    "limit": 15
                }
            ]
        },
        "sports": {
            "urls": [
                {
                    "url": "https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/",
                    "max_depth": 1,
                    "limit": 10
                },
                {
                    "url": "https://www.mlb.com/pirates",
                    "max_depth": 2,
                    "limit": 40,
                },
                {
                    "url": "https://www.steelers.com/",
                    "max_depth": 2,
                    "limit": 40,
                },
                {
                    "url": "https://www.nhl.com/penguins/",
                    "max_depth": 2,
                    "limit": 40,
                }
            ]
        }
    }
    
    SINGLE_PAGE_URLS = {
        "wikipedia": [
            "https://en.wikipedia.org/wiki/Pittsburgh",
            "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
            "https://en.wikipedia.org/wiki/Carnegie_Mellon_University",
            "https://en.wikipedia.org/wiki/Pittsburgh_Symphony_Orchestra",
            "https://en.wikipedia.org/wiki/Pittsburgh_Opera",
            "https://en.wikipedia.org/wiki/Carnegie_Museums_of_Pittsburgh",
            "https://en.wikipedia.org/wiki/Heinz_History_Center",
            "https://en.wikipedia.org/wiki/The_Frick_Pittsburgh",
            "https://en.wikipedia.org/wiki/Pittsburgh_Pirates",
            "https://en.wikipedia.org/wiki/Pittsburgh_Steelers",
            "https://en.wikipedia.org/wiki/Pittsburgh_Penguins",
            "https://en.wikipedia.org/wiki/PNC_Park",
            "https://en.wikipedia.org/wiki/Acrisure_Stadium",
            "https://en.wikipedia.org/wiki/PPG_Paints_Arena"
        ],
        "city_documents": [
            "https://www.pittsburghpa.gov/files/assets/city/v/4/omb/documents/operating-budgets/2025-operating-budget.pdf"
        ]
    }
    
    main(
        api_key=API_KEY,
        output_dir=OUTPUT_DIR,
        combined_output_file=COMBINED_OUTPUT_FILE,
        failed_urls_file=FAILED_URLS_FILE,
        website_configs=WEBSITE_CONFIGS,
        single_page_urls=SINGLE_PAGE_URLS,
        delay_between_requests=DELAY_BETWEEN_REQUESTS
    )