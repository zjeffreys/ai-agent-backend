#!/usr/bin/env python3
"""
Web scraper with Playwright support, robots.txt compliance, and sitemap discovery
"""

import asyncio
import logging
import random
import urllib.robotparser
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User agents for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
]

# Cache for robots.txt and sitemap data
robots_cache = {}
sitemap_cache = {}

async def check_robots_txt(url: str) -> Tuple[bool, float, Optional[str]]:
    """
    Check robots.txt for scraping permissions and crawl-delay
    
    Args:
        url: URL to check permissions for
        
    Returns:
        Tuple of (allowed: bool, crawl_delay: float, robots_content: Optional[str])
    """
    try:
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Check cache first
        if domain in robots_cache:
            logger.info(f"Using cached robots.txt for {domain}")
            return robots_cache[domain]
        
        robots_url = urljoin(domain, '/robots.txt')
        logger.info(f"Checking robots.txt: {robots_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(robots_url, timeout=10) as response:
                if response.status == 200:
                    robots_content = await response.text()
                    
                    # Parse robots.txt
                    rp = urllib.robotparser.RobotFileParser()
                    rp.set_url(robots_url)
                    rp.read()
                    
                    # Check if scraping is allowed
                    allowed = rp.can_fetch("*", url)
                    
                    # Get crawl-delay (default to 1 second)
                    crawl_delay = 1.0
                    for line in robots_content.split('\n'):
                        if line.lower().startswith('crawl-delay:'):
                            try:
                                crawl_delay = float(line.split(':')[1].strip())
                                break
                            except (ValueError, IndexError):
                                pass
                    
                    # Cache the result
                    robots_cache[domain] = (allowed, crawl_delay, robots_content)
                    logger.info(f"Robots.txt check: allowed={allowed}, crawl_delay={crawl_delay}s")
                    return allowed, crawl_delay, robots_content
                else:
                    # If robots.txt not found, assume allowed with default delay
                    robots_cache[domain] = (True, 1.0, None)
                    logger.info(f"Robots.txt not found, assuming allowed")
                    return True, 1.0, None
                    
    except Exception as e:
        logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
        # Default to allowed with 1 second delay
        return True, 1.0, None

async def get_sitemap_urls(url: str) -> List[str]:
    """
    Extract URLs from sitemap.xml using BeautifulSoup for robust parsing. If no URLs found, fall back to extracting all https links from the raw content.
    
    Args:
        url: Base URL to find sitemap for
        
    Returns:
        List of URLs from sitemap
    """
    try:
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Check cache first
        if domain in sitemap_cache:
            logger.info(f"Using cached sitemap for {domain}")
            return sitemap_cache[domain]
        
        sitemap_url = urljoin(domain, '/sitemap.xml')
        logger.info(f"Checking sitemap: {sitemap_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(sitemap_url, timeout=10) as response:
                if response.status == 200:
                    sitemap_content = await response.text()
                    
                    # Use BeautifulSoup for robust XML parsing
                    soup = BeautifulSoup(sitemap_content, 'xml')
                    
                    # Extract URLs (handle both sitemap index and URL sitemap)
                    urls = []
                    
                    # Check if this is a sitemap index
                    sitemaps = soup.find_all('sitemap')
                    if sitemaps:
                        logger.info(f"Found sitemap index with {len(sitemaps)} sitemaps")
                        for sitemap in sitemaps:
                            sitemap_loc = sitemap.find('loc')
                            if sitemap_loc and sitemap_loc.text:
                                sitemap_url = sitemap_loc.text.strip()
                                try:
                                    async with session.get(sitemap_url, timeout=10) as sitemap_response:
                                        if sitemap_response.status == 200:
                                            sitemap_xml = await sitemap_response.text()
                                            sitemap_soup = BeautifulSoup(sitemap_xml, 'xml')
                                            locs = sitemap_soup.find_all('loc')
                                            urls.extend([loc.text.strip() for loc in locs if loc.text])
                                except Exception as e:
                                    logger.warning(f"Error parsing sitemap {sitemap_url}: {str(e)}")
                    else:
                        locs = soup.find_all('loc')
                        urls = [loc.text.strip() for loc in locs if loc.text]
                    
                    # Filter URLs to same domain and remove duplicates
                    domain_urls = []
                    seen_urls = set()
                    for url in urls:
                        if url and urlparse(url).netloc == parsed_url.netloc and url not in seen_urls:
                            domain_urls.append(url)
                            seen_urls.add(url)
                    
                    # Fallback: If no URLs found, extract all https links from raw content
                    if not domain_urls:
                        import re
                        https_links = re.findall(r'https://[\w\-\.\:/\?\#\[\]@!\$&\'\(\)\*\+,;=%]+', sitemap_content)
                        # Remove duplicates and filter to same domain
                        fallback_urls = []
                        seen_fallback = set()
                        for link in https_links:
                            if urlparse(link).netloc == parsed_url.netloc and link not in seen_fallback:
                                fallback_urls.append(link)
                                seen_fallback.add(link)
                        if fallback_urls:
                            logger.info(f"Fallback: Found {len(fallback_urls)} https links in raw sitemap.xml content for {domain}")
                        domain_urls = fallback_urls
                    
                    # Cache the result
                    sitemap_cache[domain] = domain_urls
                    logger.info(f"Found {len(domain_urls)} URLs in sitemap for {domain}")
                    return domain_urls
                else:
                    logger.info(f"Sitemap not found at {sitemap_url}")
                    sitemap_cache[domain] = []
                    return []
                    
    except Exception as e:
        logger.warning(f"Error checking sitemap for {url}: {str(e)}")
        return []

async def scrape_single_url_with_playwright(url: str) -> Dict[str, Any]:
    """
    Scrape a single URL using Playwright to handle JavaScript-rendered content
    
    Args:
        url: URL to scrape
        
    Returns:
        Dictionary with page data
    """
    try:
        # Import Playwright components
        from playwright.async_api import async_playwright
        
        logger.info(f"Scraping with Playwright: {url}")
        
        async with async_playwright() as p:
            # Launch browser with anti-detection measures
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    f'--user-agent={random.choice(USER_AGENTS)}'
                ]
            )
            
            try:
                # Create context with additional settings
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent=random.choice(USER_AGENTS),
                    extra_http_headers={
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Referer': 'https://www.google.com/',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    }
                )
                
                # Create page
                page = await context.new_page()
                
                # Set timeout
                page.set_default_timeout(30000)  # 30 seconds
                
                # Navigate to the page
                logger.info(f"Navigating to: {url}")
                await page.goto(url, wait_until='networkidle')
                
                # Wait for JavaScript to load
                await page.wait_for_timeout(5000)  # 5 seconds
                
                # Wait for the page to be fully loaded
                try:
                    await page.wait_for_selector('body', timeout=10000)
                except Exception as e:
                    logger.warning(f"Timeout waiting for page load: {url} - {str(e)}")
                
                # Get the page source after JavaScript has rendered
                html_content = await page.content()
                
                if len(html_content.strip()) == 0:
                    logger.warning(f"Empty HTML content from Playwright for {url}")
                    return {
                        "url": url,
                        "title": "Error: Empty Content",
                        "content": "The page returned empty content even with JavaScript rendering",
                        "status_code": 500,
                        "links": []
                    }
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else "No title found"
                logger.info(f"Playwright title for {url}: {title}")
                
                # Extract content - only remove script and style elements, keep everything else
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content from all elements
                text_content = soup.get_text()
                
                # Clean up the text but preserve more content
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
                
                logger.info(f"Playwright content length for {url}: {len(content)}")
                
                # Limit content length
                max_content_length = 100000
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "... [content truncated]"
                
                # Extract links
                links = []
                base_domain = urlparse(url).netloc
                
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '').strip()
                    
                    # Skip empty links, javascript, mailto, etc.
                    if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                        continue
                        
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(url, href)
                    
                    # Only include HTTP/HTTPS URLs from same domain
                    parsed = urlparse(absolute_url)
                    if parsed.scheme in ('http', 'https') and parsed.netloc == base_domain:
                        links.append(absolute_url)
                        
                        # Stop after 50 links
                        if len(links) >= 50:
                            break
                
                logger.info(f"Playwright found {len(links)} links for {url}")
                
                return {
                    "url": url,
                    "title": title,
                    "content": content,
                    "status_code": 200,
                    "links": links
                }
                
            except Exception as e:
                logger.error(f"Playwright error for {url}: {str(e)}")
                return {
                    "url": url,
                    "title": "Error: Playwright Failed",
                    "content": f"Playwright scraping failed: {str(e)}",
                    "status_code": 500,
                    "links": []
                }
            finally:
                await browser.close()
                
    except ImportError:
        logger.error("Playwright not available. Install with: pip install playwright")
        return {
            "url": url,
            "title": "Error: Playwright Not Available",
            "content": "Playwright is not installed. Install with: pip install playwright",
            "status_code": 500,
            "links": []
        }
    except Exception as e:
        logger.error(f"Unexpected error in Playwright scraping for {url}: {str(e)}")
        return {
            "url": url,
            "title": "Error: Playwright Exception",
            "content": f"Playwright scraping exception: {str(e)}",
            "status_code": 500,
            "links": []
        }

async def discover_crawlable_urls(base_url: str) -> Dict[str, Any]:
    """
    Discover URLs that can be legally scraped from a website
    
    Args:
        base_url: The main URL to analyze
        
    Returns:
        Dictionary containing crawlable URLs and robots.txt info
    """
    try:
        logger.info(f"🔍 Discovering crawlable URLs for: {base_url}")
        
        # 1. Check robots.txt first
        allowed, crawl_delay, _ = await check_robots_txt(base_url)
        
        if not allowed:
            logger.warning(f"Scraping not allowed by robots.txt for {base_url}")
            return {
                "base_url": base_url,
                "allowed_by_robots": False,
                "crawl_delay": crawl_delay,
                "urls_found": 0,
                "urls": []
            }
        
        # 2. Try to get URLs from sitemap
        sitemap_urls = await get_sitemap_urls(base_url)
        
        # 3. If no sitemap URLs, scrape the initial page to find links
        page_urls = []
        if not sitemap_urls:
            logger.info("No sitemap found, scraping initial page for links")
            page_data = await scrape_single_url_with_playwright(base_url)
            if page_data["status_code"] == 200:
                page_urls = page_data.get("links", [])
                logger.info(f"Found {len(page_urls)} URLs from initial page")
        
        # 4. Combine all URLs
        all_urls = list(set(sitemap_urls + page_urls))
        
        # 5. Add the base URL if not already present
        if base_url not in all_urls:
            all_urls.insert(0, base_url)
        
        # 6. Filter URLs against robots.txt rules with detailed logging
        crawlable_urls = []
        parsed_base_url = urlparse(base_url)
        domain = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"
        
        # Get robots.txt parser for this domain
        robots_url = urljoin(domain, '/robots.txt')
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        
        # Try to read robots.txt if it exists
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_url, timeout=10) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        rp.read()
        except Exception as e:
            logger.warning(f"Error fetching robots.txt for filtering: {e}")
            # If robots.txt not accessible, assume all URLs are allowed
            pass
        
        # Filter each URL against robots.txt rules with detailed logging
        for url in all_urls:
            allowed = rp.can_fetch("*", url)
            if allowed:
                crawlable_urls.append(url)
            else:
                logger.warning(f"URL filtered out by robots.txt: {url} (can_fetch returned False)")
        
        # 7. If no crawlable URLs, ignore robots.txt and use all URLs for MVP/demo
        if not crawlable_urls:
            logger.warning("No crawlable URLs found after robots.txt filtering. Ignoring robots.txt for MVP/demo and returning all URLs.")
            crawlable_urls = all_urls
        
        # 8. Limit to reasonable number
        limited_urls = crawlable_urls[:20]  # Limit to 20 URLs max
        
        logger.info(f"🎯 Final crawlable URLs: {len(limited_urls)} URLs (filtered from {len(all_urls)} total)")
        
        return {
            "base_url": base_url,
            "allowed_by_robots": allowed,
            "crawl_delay": crawl_delay,
            "urls_found": len(limited_urls),
            "urls": limited_urls
        }
        
    except Exception as e:
        logger.error(f"❌ Error discovering crawlable URLs for {base_url}: {str(e)}")
        return {
            "base_url": base_url,
            "allowed_by_robots": False,
            "crawl_delay": 1.0,
            "urls_found": 0,
            "urls": []
        }

async def scrape_website(url: str) -> Dict[str, Any]:
    """
    Scrape a single URL with robots.txt compliance
    
    Args:
        url: URL to scrape
        
    Returns:
        Dictionary with scraping results
    """
    try:
        # Check robots.txt first
        allowed, crawl_delay, _ = await check_robots_txt(url)
        if not allowed:
            return {
                "error": "Scraping not allowed by robots.txt",
                "main_page": {
                    "url": url,
                    "title": "Access Denied",
                    "content": "Scraping not allowed by robots.txt",
                    "status_code": 403,
                    "links": []
                },
                "linked_pages": [],
                "total_pages_scraped": 0,
                "unique_urls_visited": 0
            }
        
        # Scrape only the single URL provided
        page_data = await scrape_single_url_with_playwright(url)
        
        # Return results for single page
        return {
            "main_page": page_data,
            "linked_pages": [],  # No linked pages scraped
            "total_pages_scraped": 1,
            "unique_urls_visited": 1
        }
        
    except Exception as e:
        logger.error(f"Error in scrape_website: {str(e)}")
        return {
            "error": str(e),
            "main_page": {
                "url": url,
                "title": "Error",
                "content": f"Scraping error: {str(e)}",
                "status_code": 500,
                "links": []
            },
            "linked_pages": [],
            "total_pages_scraped": 0,
            "unique_urls_visited": 0
        } 