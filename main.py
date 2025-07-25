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

# Load environment variables
load_dotenv()
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

class ScrapeRequest(BaseModel):
    url: HttpUrl

class PageData(BaseModel):
    url: str
    title: str
    content: str
    status_code: int
    content_length: int

class ScrapeResponse(BaseModel):
    main_page: PageData
    scrape_time: float

class DiscoverUrlsRequest(BaseModel):
    url: str

class DiscoverUrlsResponse(BaseModel):
    base_url: str
    allowed_by_robots: bool
    crawl_delay: float
    urls_found: int
    urls: List[str]

class AnalyzeBusinessRequest(BaseModel):
    combined_content: str

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
            "scrape": "/scrape",
            "discover_crawlable_urls": "/discover-crawlable-urls"
        }
    }

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_url(request: ScrapeRequest):
    """
    Scrape a single URL with robots.txt compliance
    
    Args:
        request: ScrapeRequest containing the URL to scrape
        
    Returns:
        ScrapeResponse with the scraped page data
    """
    start_time = time.time()
    try:
        result = await scrape_website_old(str(request.url))
        scrape_time = time.time() - start_time
        
        main_page = PageData(
            url=result["main_page"]["url"],
            title=remove_quotes(result["main_page"]["title"]),
            content=remove_quotes(result["main_page"]["content"]),
            status_code=result["main_page"]["status_code"],
            content_length=len(result["main_page"]["content"])
        )
        
        return ScrapeResponse(
            main_page=main_page,
            scrape_time=scrape_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape website: {str(e)}")

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

@app.post("/analyze-business", response_model=AnalyzeBusinessResponse)
async def analyze_business(request: AnalyzeBusinessRequest):
    """
    Analyze business content and return a market/AI analysis report using ChatGPT
    """
    system_prompt = (
        "You are a senior business analyst and AI strategy consultant. Your role is to review unstructured content from a business’s website "
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

    prompt = request.combined_content

    try:
        response = await run_in_threadpool(
            client.chat.completions.create,
            model="gpt-4o",  # or "gpt-4-turbo", "gpt-4", etc.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the scraped website content:\n\n{prompt}"}
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0.3
        )
        import json
        content = response.choices[0].message.content
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Fallback schema if it isn't perfectly formatted JSON
            result = {
                "company_overview": content,
                "key_offerings": [],
                "target_customers": [],
                "unique_selling_points": [],
                "industry_trends": [],
                "business_challenges": [],
                "ai_opportunities": {},
                "recommended_ai_use_cases": {},
                "data_requirements_and_risks": "",
                "next_steps": []
            }
        return AnalyzeBusinessResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze business: {str(e)}")

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 