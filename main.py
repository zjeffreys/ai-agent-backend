import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import uvicorn
from scraper import scrape_website as scrape_website_old, discover_crawlable_urls
import time
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.concurrency import run_in_threadpool
import uuid
from datetime import datetime
import json
import logging
from supabase import create_client, Client

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Remove: openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Enhanced Web Scraping API",
    description="A FastAPI application that scrapes individual URLs with robots.txt compliance and discovers crawlable URLs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React default port
        "http://localhost:5173",      # Vite default port
        "http://localhost:3001",      # Alternative React port
        "http://127.0.0.1:3000",     # Alternative localhost
        "http://127.0.0.1:5173",     # Alternative localhost
        "http://127.0.0.1:3001",     # Alternative localhost
        "*"  # Allow all origins in development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

def remove_quotes(text: str) -> str:
    """Remove all double and single quotes from text, replacing them with spaces."""
    return text.replace('"', ' ').replace("'", ' ')

class DiscoverUrlsRequest(BaseModel):
    url: str

class DiscoverUrlsResponse(BaseModel):
    base_url: str
    allowed_by_robots: bool
    crawl_delay: float
    urls_found: int
    urls: List[str]

class AnalyzeBusinessResponse(BaseModel):
    company_overview: str
    key_offerings_or_products: list[str]
    target_customer_segments: list[str]
    unique_selling_points: list[str]
    industry_and_market_trends: list[str]
    potential_business_challenges: list[str]
    opportunities_for_using_ai: list[str]
    recommended_ai_use_cases: dict[str, list[str]]  # short_term, medium_term, long_term
    data_requirements_and_risks: list[str]
    suggested_next_steps_for_ai_adoption: list[str]
    customer_journey_mapping: str
    digital_maturity_assessment: str
    technology_stack_overview: list[str]
    partnerships_and_alliances: list[str]
    sustainability_and_social_responsibility: str
    financial_overview: str
    actionable_recommendations: list[str]
    competitive_landscape: list[str]
    customer_testimonials: list[str]
    quantitative_opportunity_metrics: list[str]
    content_inventory: list[str]
    ai_maturity_level: str
    data_sources_reviewed: list[str]
    business_stage: str
    branding_tone: str
    visual_opportunities: list[str]
    team_ai_readiness: str

class ChatRequest(BaseModel):
    chatbot: Dict[str, Any]
    messages: list  # Conversation history
    businessInfo: str = ""

# New models for refactored endpoints
class ScrapeUrlsRequest(BaseModel):
    urls: List[str]
    base_url: str

class ScrapeUrlsResponse(BaseModel):
    crawl_id: str
    status: str
    message: str

class ScrapeStatusResponse(BaseModel):
    crawl_id: str
    status: str
    progress: int
    total_urls: int
    scraped_count: int
    failed_count: int
    remaining_count: int
    scraped_urls: List[str]  # Add list of successfully scraped URLs
    failed_urls: List[str]  # Add list of failed URLs
    created_at: datetime
    updated_at: datetime

class AnalyzeScrapedDataRequest(BaseModel):
    crawl_id: str

class AnalyzeScrapedDataResponse(BaseModel):
    analysis_id: str
    status: str
    message: str

class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: str
    progress: int
    result: Optional[AnalyzeBusinessResponse] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class BusinessResponse(BaseModel):
    id: str
    business_name: str
    base_url: str
    industry: Optional[str] = None
    description: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

@app.get("/")
async def root():
    """
    Testing endpoint that returns a simple message
    """
    return {
        "message": "Enhanced Web Scraping API is running!",
        "description": "This API scrapes individual URLs with robots.txt compliance and discovers crawlable URLs",
        "features": [
            "Scrapes individual URLs with robots.txt compliance",
            "Discovers crawlable URLs from websites",
            "Uses Playwright for JavaScript support",
            "Respects crawl-delay and robots.txt rules"
        ],
        "endpoints": {
            "test": "/",
            "discover_crawlable_urls": "/discover-crawlable-urls",
            "scrape_urls": "/scrape-urls",
            "check_scrape_status": "/check-scrape-status/{crawl_id}",
            "analyze_scraped_data": "/analyze-scraped-data",
            "check_analysis": "/check-analysis/{analysis_id}",
            "businesses": "/businesses",
            "business_by_id": "/businesses/{business_id}",
            "chat": "/api/chat"
        }
    }

@app.post("/discover-crawlable-urls", response_model=DiscoverUrlsResponse)
async def discover_crawlable_urls_endpoint(request: DiscoverUrlsRequest):
    """
    Discover URLs that can be legally scraped from a website
    
    Args:
        request: DiscoverUrlsRequest containing the URL to analyze
        
    Returns:
        DiscoverUrlsResponse with crawlable URLs and robots.txt info
    """
    try:
        result = await discover_crawlable_urls(request.url)
        return DiscoverUrlsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to discover crawlable URLs: {str(e)}")

client = OpenAI()

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    chatbot = req.chatbot
    messages = req.messages
    business_info = req.businessInfo

    system_prompt = f"""
You are a helpful assistant for the business \"{chatbot.get('name', 'Business')}\".
Tone: {chatbot.get('tone', 'Friendly')}
Response style: {chatbot.get('response_style', 'Short')}
Knowledge sources:
{chr(10).join('- ' + entry for entry in (chatbot.get('manual_entries') or []))}
{f'Business Vault Info: {business_info}' if chatbot.get('knowledge_sources') and 'Business Vault' in chatbot['knowledge_sources'] else ''}
Prompts: {'; '.join(chatbot.get('prompts') or [])}
Language: {', '.join(chatbot.get('languages') or ['English'])}
"""

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        response = await run_in_threadpool(
            client.chat.completions.create,
            model="gpt-4",
            messages=full_messages
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot failed: {str(e)}")

# New refactored endpoints
@app.post("/scrape-urls", response_model=ScrapeUrlsResponse)
async def scrape_urls(request: ScrapeUrlsRequest):
    """Start scraping multiple URLs and create a crawl record"""
    crawl_id = str(uuid.uuid4())
    
    # Extract business name from base_url for deduplication
    from urllib.parse import urlparse
    parsed_url = urlparse(request.base_url)
    domain = parsed_url.netloc
    
    # Check if business already exists
    business_response = supabase.table("businesses").select("*").eq("base_url", request.base_url).execute()
    
    business_id = None
    if business_response.data:
        # Business exists, use existing record
        business_id = business_response.data[0]["id"]
    else:
        # Create new business record
        business_data = {
            "business_name": domain,  # Use domain as business name initially
            "base_url": request.base_url,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        business_insert = supabase.table("businesses").insert(business_data).execute()
        business_id = business_insert.data[0]["id"]
    
    # Create crawl record
    crawl_data = {
        "id": crawl_id,
        "business_id": business_id,
        "base_url": request.base_url,
        "urls_to_scrape": request.urls,
        "status": "pending",
        "progress": 0,
        "total_urls": len(request.urls),
        "scraped_count": 0,
        "failed_count": 0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    try:
        supabase.table("crawls").insert(crawl_data).execute()
        
        # Start background scraping task
        asyncio.create_task(process_scraping_task(crawl_id))
        
        return ScrapeUrlsResponse(
            crawl_id=crawl_id,
            status="pending",
            message="Scraping task created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create scraping task: {str(e)}")

@app.get("/check-scrape-status/{crawl_id}", response_model=ScrapeStatusResponse)
async def check_scrape_status(crawl_id: str):
    """Check the status of a scraping task"""
    try:
        response = supabase.table("crawls").select("*").eq("id", crawl_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Crawl not found")
        
        crawl = response.data[0]
        remaining = crawl["total_urls"] - crawl["scraped_count"] - crawl["failed_count"]
        
        # Get list of scraped URLs from scraped_pages table
        scraped_urls = []
        if crawl["scraped_count"] > 0:
            pages_response = supabase.table("scraped_pages").select("url").eq("crawl_id", crawl_id).execute()
            scraped_urls = [page["url"] for page in pages_response.data]
        
        # Get failed URLs from crawl record
        failed_urls = crawl.get("failed_urls", [])
        
        return ScrapeStatusResponse(
            crawl_id=crawl_id,
            status=crawl["status"],
            progress=crawl["progress"],
            total_urls=crawl["total_urls"],
            scraped_count=crawl["scraped_count"],
            failed_count=crawl["failed_count"],
            remaining_count=remaining,
            scraped_urls=scraped_urls,
            failed_urls=failed_urls,
            created_at=datetime.fromisoformat(crawl["created_at"]),
            updated_at=datetime.fromisoformat(crawl["updated_at"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get crawl status: {str(e)}")

@app.post("/analyze-scraped-data", response_model=AnalyzeScrapedDataResponse)
async def analyze_scraped_data(request: AnalyzeScrapedDataRequest):
    """Start analysis of scraped data"""
    analysis_id = str(uuid.uuid4())
    
    try:
        # Check if crawl exists and is completed
        crawl_response = supabase.table("crawls").select("*").eq("id", request.crawl_id).execute()
        if not crawl_response.data:
            raise HTTPException(status_code=404, detail="Crawl not found")
        
        crawl = crawl_response.data[0]
        if crawl["status"] != "completed":
            raise HTTPException(status_code=400, detail="Crawl must be completed before analysis")
        
        # Get all scraped pages for this crawl
        pages_response = supabase.table("scraped_pages").select("*").eq("crawl_id", request.crawl_id).execute()
        scraped_pages = pages_response.data
        
        if not scraped_pages:
            raise HTTPException(status_code=400, detail="No scraped pages found for this crawl")
        
        # Combine all content
        combined_content = "\n\n".join([page["content"] for page in scraped_pages if page["content"]])
        
        # Create analysis record
        analysis_data = {
            "id": analysis_id,
            "business_id": crawl["business_id"],
            "crawl_id": request.crawl_id,
            "base_url": crawl["base_url"],
            "status": "pending",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        supabase.table("business_analyses").insert(analysis_data).execute()
        
        # Start background analysis task
        asyncio.create_task(process_analysis_task_new(analysis_id, combined_content))
        
        return AnalyzeScrapedDataResponse(
            analysis_id=analysis_id,
            status="pending",
            message="Analysis task created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.get("/check-analysis/{analysis_id}", response_model=AnalysisStatusResponse)
async def check_analysis(analysis_id: str):
    """Check the status of an analysis task"""
    try:
        response = supabase.table("business_analyses").select("*").eq("id", analysis_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        analysis = response.data[0]
        
        # If analysis is completed, get the result
        result = None
        if analysis["status"] == "completed" and analysis.get("analysis_data"):
            try:
                result = AnalyzeBusinessResponse(**analysis["analysis_data"])
            except Exception as e:
                logger.warning(f"Error parsing analysis result for {analysis_id}: {e}")
        
        return AnalysisStatusResponse(
            analysis_id=analysis_id,
            status=analysis["status"],
            progress=analysis["progress"],
            result=result,
            error=analysis.get("error"),
            created_at=datetime.fromisoformat(analysis["created_at"]),
            updated_at=datetime.fromisoformat(analysis["updated_at"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis status: {str(e)}")

@app.get("/businesses")
async def get_all_businesses():
    """Get all businesses"""
    try:
        response = supabase.table("businesses").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch businesses: {str(e)}")

@app.get("/businesses/{business_id}")
async def get_business_by_id(business_id: str):
    """Get a specific business"""
    try:
        response = supabase.table("businesses").select("*").eq("id", business_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Business not found")
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch business: {str(e)}")

# Background task functions
async def process_scraping_task(crawl_id: str):
    """Background task to scrape URLs"""
    try:
        # Get crawl data
        crawl_response = supabase.table("crawls").select("*").eq("id", crawl_id).execute()
        crawl = crawl_response.data[0]
        
        # Update status to scraping
        supabase.table("crawls").update({
            "status": "scraping",
            "updated_at": datetime.now().isoformat()
        }).eq("id", crawl_id).execute()
        
        urls_to_scrape = crawl["urls_to_scrape"]
        total_urls = len(urls_to_scrape)
        scraped_count = 0
        failed_count = 0
        failed_urls = []
        
        # Scrape each URL
        for i, url in enumerate(urls_to_scrape):
            try:
                # Scrape the URL
                result = await scrape_website_old(url)
                content = remove_quotes(result["main_page"]["content"])
                
                # Store scraped page
                page_data = {
                    "crawl_id": crawl_id,
                    "business_id": crawl["business_id"],
                    "base_url": crawl["base_url"],
                    "url": url,
                    "title": remove_quotes(result["main_page"]["title"]),
                    "content": content,
                    "status_code": result["main_page"]["status_code"],
                    "scraped_at": datetime.now().isoformat()
                }
                
                supabase.table("scraped_pages").insert(page_data).execute()
                scraped_count += 1
                
                # Update progress
                progress = int((i + 1) / total_urls * 100)
                supabase.table("crawls").update({
                    "progress": progress,
                    "scraped_count": scraped_count,
                    "updated_at": datetime.now().isoformat()
                }).eq("id", crawl_id).execute()
                
                # Add delay to be respectful
                await asyncio.sleep(1)
                
            except Exception as e:
                failed_count += 1
                failed_urls.append(url)
                logger.error(f"Failed to scrape {url}: {str(e)}")
                
                # Update failed count
                supabase.table("crawls").update({
                    "failed_count": failed_count,
                    "failed_urls": failed_urls,  # Store failed URLs in crawl record
                    "updated_at": datetime.now().isoformat()
                }).eq("id", crawl_id).execute()
        
        # Update crawl as completed
        supabase.table("crawls").update({
            "status": "completed",
            "progress": 100,
            "updated_at": datetime.now().isoformat()
        }).eq("id", crawl_id).execute()
        
    except Exception as e:
        # Update crawl as failed
        supabase.table("crawls").update({
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.now().isoformat()
        }).eq("id", crawl_id).execute()
        logger.error(f"Crawl {crawl_id} failed: {str(e)}")

async def process_analysis_task_new(analysis_id: str, combined_content: str):
    """Background task to analyze scraped data"""
    try:
        # Update status to analyzing
        supabase.table("business_analyses").update({
            "status": "analyzing",
            "progress": 25,
            "updated_at": datetime.now().isoformat()
        }).eq("id", analysis_id).execute()
        
        # Your existing analysis logic
        system_prompt = (
            "You are a senior business analyst and AI strategy consultant. Your role is to review unstructured content from a business's website "
            "(such as product pages, about section, services, blogs, and customer testimonials), extract important information, and generate a comprehensive, structured JSON report.\n\n"
            "This report must include:\n"
            "1. Company Overview (mission, vision, values, history, leadership)\n"
            "2. Key Offerings or Products (detailed descriptions, pricing, testimonials)\n"
            "3. Target Customer Segments (demographics, pain points, personas)\n"
            "4. Unique Selling Points (competitive advantages)\n"
            "5. Industry & Market Trends (size, competitors, regulations, technologies)\n"
            "6. Potential Business Challenges (SWOT, barriers, bottlenecks)\n"
            "7. Opportunities for Using AI (by department, with examples)\n"
            "8. Recommended AI Use Cases (short/medium/long-term, roadmap) — ensure to include specific, actionable ways the business can use AI agents or automation agents in their operations, customer service, marketing, or product delivery.\n"
            "9. Data Requirements & Risks (sources, privacy, risks)\n"
            "10. Suggested Next Steps for AI Adoption (quick wins, training, budget)\n"
            "11. Customer Journey Mapping (touchpoints, pain points, automation opportunities)\n"
            "12. Digital Maturity Assessment (current tools, readiness)\n"
            "13. Technology Stack Overview (software, hardware, integrations)\n"
            "14. Partnerships & Alliances (key partners)\n"
            "15. Sustainability & Social Responsibility (ESG initiatives)\n"
            "16. Financial Overview (revenue, costs, profitability)\n"
            "17. Actionable Recommendations (prioritized actions, KPIs) — always include at least 3 recommendations for how the business can use AI agents or automation agents.\n"
            "18. Competitive Landscape (main competitors, market position)\n"
            "19. Customer Testimonials / Case Studies (quotes, stories)\n"
            "20. Quantitative Opportunity Metrics (ROI, time/cost savings, KPIs)\n"
            "21. Content Inventory (list of pages/URLs analyzed)\n"
            "22. AI Maturity Level (0-5 scale or descriptive)\n"
            "23. Data Sources Reviewed (list of URLs/pages)\n"
            "24. Business Stage (Startup, Growth, Mature, etc.)\n"
            "25. Branding Tone (e.g., Bold, Professional, Friendly)\n"
            "26. Visual Opportunities (UI/UX moments for AI)\n"
            "27. Team AI Readiness (skills gap analysis)\n\n"
            "If content is not directly available, infer from context or use reasonable industry-specific defaults. Never leave fields blank unless absolutely no information is possible.\n"
            "Format the output strictly as valid, minified JSON with these keys and types:\n"
            '{"company_overview": "...", "key_offerings_or_products": ["..."], "target_customer_segments": ["..."], "unique_selling_points": ["..."], "industry_and_market_trends": ["..."], "potential_business_challenges": ["..."], "opportunities_for_using_ai": ["..."], "recommended_ai_use_cases": {"short_term": ["..."], "medium_term": ["..."], "long_term": ["..."]}, "data_requirements_and_risks": ["..."], "suggested_next_steps_for_ai_adoption": ["..."], "customer_journey_mapping": "...", "digital_maturity_assessment": "...", "technology_stack_overview": ["..."], "partnerships_and_alliances": ["..."], "sustainability_and_social_responsibility": "...", "financial_overview": "...", "actionable_recommendations": ["..."], "competitive_landscape": ["..."], "customer_testimonials": ["..."], "quantitative_opportunity_metrics": ["..."], "content_inventory": ["..."], "ai_maturity_level": "...", "data_sources_reviewed": ["..."], "business_stage": "...", "branding_tone": "...", "visual_opportunities": ["..."], "team_ai_readiness": "..."}'
            "\nIf any section is missing, use an empty array, empty object, or 'N/A'. No explanations or extra text."
        )
        
        # Update progress
        supabase.table("business_analyses").update({
            "progress": 50,
            "updated_at": datetime.now().isoformat()
        }).eq("id", analysis_id).execute()
        
        response = await run_in_threadpool(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the scraped website content:\n\n{combined_content}"}
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0.3
        )
        
        # Update progress
        supabase.table("business_analyses").update({
            "progress": 75,
            "updated_at": datetime.now().isoformat()
        }).eq("id", analysis_id).execute()
        
        # Parse result
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Try to extract business name from the analysis result
        business_name = None
        if result.get("company_overview"):
            overview = result["company_overview"]
            if " is " in overview:
                business_name = overview.split(" is ")[0].strip()
            elif " - " in overview:
                business_name = overview.split(" - ")[0].strip()
        
        # Update business name if found
        if business_name:
            analysis_response = supabase.table("business_analyses").select("business_id").eq("id", analysis_id).execute()
            if analysis_response.data:
                business_id = analysis_response.data[0]["business_id"]
                supabase.table("businesses").update({
                    "business_name": business_name,
                    "updated_at": datetime.now().isoformat()
                }).eq("id", business_id).execute()
        
        # Update analysis as completed
        supabase.table("business_analyses").update({
            "status": "completed",
            "progress": 100,
            "analysis_data": result,
            "updated_at": datetime.now().isoformat()
        }).eq("id", analysis_id).execute()
        
    except Exception as e:
        # Update analysis as failed
        supabase.table("business_analyses").update({
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.now().isoformat()
        }).eq("id", analysis_id).execute()
        logger.error(f"Analysis {analysis_id} failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 