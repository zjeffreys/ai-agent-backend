# AI Agent Backend - Enhanced Web Scraping API

A FastAPI application that scrapes websites with **robots.txt compliance**, **sitemap discovery**, and **detailed logging** for frontend integration.

## 🚀 New Features

### ✅ Robots.txt Compliance
- Automatically checks robots.txt before scraping
- Respects crawl-delay directives
- Caches robots.txt results for performance

### ✅ Sitemap Discovery
- Automatically discovers and parses sitemap.xml
- Handles both sitemap index and URL sitemaps
- Filters URLs to same domain

### ✅ Single-Page Scraping
- Scrape one URL at a time for better frontend logging
- Detailed progress tracking
- Real-time status updates

### ✅ Enhanced Logging
- Emoji-rich logging for better visibility
- Progress tracking with percentages
- Detailed error reporting

## 📋 API Endpoints

### `/scrape-single` (POST)
Scrape a single URL with robots.txt compliance

```json
{
  "url": "https://example.com/page"
}
```

**Response:**
```json
{
  "success": true,
  "url": "https://example.com/page",
  "title": "Page Title",
  "content": "Page content...",
  "content_length": 1500,
  "status_code": 200,
  "links": ["https://example.com/link1", "https://example.com/link2"],
  "scraped_at": 1703123456.789,
  "error": null
}
```

### `/scrape-website` (POST)
Scrape a complete website with robots.txt compliance and detailed logging

```json
{
  "url": "https://example.com"
}
```

**Response:**
```json
{
  "base_url": "https://example.com",
  "total_pages": 5,
  "successful_pages": 4,
  "failed_pages": 1,
  "crawl_delay_used": 1.0,
  "results": [
    {
      "url": "https://example.com",
      "title": "Homepage",
      "content": "Content...",
      "success": true,
      "content_length": 1500,
      "progress": {"current": 1, "total": 5, "percentage": 20.0}
    }
  ],
  "scraping_completed_at": 1703123456.789
}
```

### `/discover-urls` (GET)
Discover URLs to scrape from a website

```
GET /discover-urls?url=https://example.com
```

**Response:**
```json
{
  "base_url": "https://example.com",
  "urls_found": 10,
  "urls": [
    "https://example.com",
    "https://example.com/about",
    "https://example.com/contact"
  ]
}
```

### `/check-robots` (GET)
Check robots.txt for a URL

```
GET /check-robots?url=https://example.com
```

**Response:**
```json
{
  "url": "https://example.com",
  "allowed": true,
  "crawl_delay": 1.0
}
```

### `/contact` (POST)
Submit a contact form to Google Sheets via Google Apps Script

**Request Body:**
```json
{
  "firstName": "string (required)",
  "lastName": "string (required)",
  "email": "string (required, valid email format)",
  "company": "string (optional)",
  "phone": "string (optional)",
  "subject": "string (required, one of: general, pilot, partnership, technical, pricing, other)",
  "message": "string (required, max 1000 characters)"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Contact form submitted successfully",
  "submission_id": "uuid-string"
}
```

**Validation Rules:**
- `firstName`, `lastName`, `email`, `subject`, and `message` are required
- `email` must be in valid email format
- `subject` must be one of: general, pilot, partnership, technical, pricing, other
- `message` must be 1000 characters or less
- `company` and `phone` are optional

## 🔧 Configuration

### ScrapingConfig Settings

```python
class ScrapingConfig:
    MAX_PAGES = 1  # Scrape up to 1 page at a time
    MAX_LINKS_PER_PAGE = 50
    REQUEST_DELAY = 0.5  # seconds
    TIMEOUT = 30  # seconds
    STAY_ON_DOMAIN = True  # Only scrape links from same domain
    CHECK_ROBOTS_TXT = True  # Enable robots.txt checking
    MAX_RETRIES = 3  # Add retry logic
    RETRY_DELAY = 1.0  # seconds between retries
    PLAYWRIGHT_TIMEOUT = 20000  # milliseconds to wait for page load
    PLAYWRIGHT_WAIT_FOR_JS = 5000  # 5 seconds for JS loading
    MAX_CONTENT_LENGTH = 100000  # 100k characters content limit
```

## 🏃‍♂️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Playwright Browsers
```bash
playwright install
```

### 3. Run the API
```bash
python main.py
```

### 4. Test the New Functionality
```bash
python test_new_scraper.py
```

## 📊 Frontend Integration

### Real-time Progress Tracking

The new scraper provides detailed progress information perfect for frontend integration:

```javascript
// Frontend can track progress like this:
const response = await fetch('/scrape-website', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ url: 'https://example.com' })
});

const result = await response.json();

// Show progress for each page
result.results.forEach(page => {
  console.log(`Page ${page.progress.current}/${page.progress.total}: ${page.url}`);
  console.log(`Progress: ${page.progress.percentage}%`);
  console.log(`Success: ${page.success}`);
});
```

### Logging Output Example

```
🚀 Starting website scraping for: https://example.com
🔍 Checking robots.txt: https://example.com/robots.txt
✅ Robots.txt check: allowed=True, crawl_delay=1.0s
🗺️  Checking sitemap: https://example.com/sitemap.xml
✅ Found 5 URLs in sitemap for example.com
📄 [1/5] Scraping: https://example.com
🔄 Starting scrape of: https://example.com
✅ Successfully scraped: https://example.com (1500 chars)
⏳ Waiting 1.0s (robots.txt crawl-delay)
📄 [2/5] Scraping: https://example.com/about
🔄 Starting scrape of: https://example.com/about
✅ Successfully scraped: https://example.com/about (1200 chars)
⏳ Waiting 1.0s (robots.txt crawl-delay)
🎉 Scraping completed! 5/5 pages successful
```

## 🔒 Robots.txt Compliance

The scraper automatically:

1. **Checks robots.txt** before scraping any URL
2. **Respects crawl-delay** directives (waits between requests)
3. **Caches results** to avoid repeated robots.txt requests
4. **Falls back gracefully** if robots.txt is not found

## 🗺️ Sitemap Discovery

The scraper intelligently discovers URLs by:

1. **Checking sitemap.xml** first (fastest method)
2. **Parsing sitemap index** files if present
3. **Extracting links** from initial page if no sitemap
4. **Filtering to same domain** for focused scraping
5. **Limiting to 10 URLs** maximum for performance

## ⚡ Performance Optimizations

- **Caching**: Robots.txt and sitemap results are cached per domain
- **Single-page scraping**: Process one URL at a time for better memory usage
- **Configurable delays**: Respect robots.txt crawl-delay while maintaining speed
- **Error handling**: Graceful fallbacks for missing robots.txt or sitemaps

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_new_scraper.py
```

This will test:
- Robots.txt checking
- Sitemap discovery
- URL discovery process
- Single page scraping
- Complete website scraping

## 📝 Logging

The scraper provides detailed logging with emojis for easy identification:

- 🚀 Starting processes
- 🔍 Checking robots.txt/sitemap
- 📄 Scraping pages
- ✅ Successful operations
- ❌ Errors
- ⏳ Waiting/delays
- 🎉 Completion

## 🔄 Migration from Old Scraper

The old scraping functionality is still available at `/scrape` endpoint. The new functionality is available at:

- `/scrape-single` - Single page scraping
- `/scrape-website` - Complete website scraping with logging
- `/discover-urls` - URL discovery
- `/check-robots` - Robots.txt checking