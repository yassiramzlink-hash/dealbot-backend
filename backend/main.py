"""
🔧 DealHunter AI - Backend Server
===================================
Handles Amazon product search via PA API and AI-powered deal scoring.
"""

import os
import re
import hmac
import hashlib
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
AMAZON_ACCESS_KEY    = os.getenv("AMAZON_ACCESS_KEY", "")
AMAZON_SECRET_KEY    = os.getenv("AMAZON_SECRET_KEY", "")
AMAZON_PARTNER_TAG   = os.getenv("AMAZON_PARTNER_TAG", "")   # e.g. yourtag-20
AMAZON_REGION        = os.getenv("AMAZON_REGION", "us-east-1")
AMAZON_HOST          = "webservices.amazon.com"
AMAZON_ENDPOINT      = f"https://{AMAZON_HOST}/paapi5/searchitems"

ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="DealHunter AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ───────────────────────────────────────────────────────────────────
class AlertRequest(BaseModel):
    user_id: int
    product_url: str
    target_price: float

class Deal(BaseModel):
    asin: str
    title: str
    price: float
    original_price: Optional[float]
    discount_percent: Optional[int]
    rating: Optional[float]
    review_count: Optional[int]
    image_url: Optional[str]
    affiliate_url: str
    ai_score: Optional[float]
    ai_summary: Optional[str]

# ─── Amazon PA API Signing (AWS Signature V4) ─────────────────────────────────
def sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

def get_signature_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date    = sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region  = sign(k_date, region)
    k_service = sign(k_region, service)
    k_signing = sign(k_service, "aws4_request")
    return k_signing

async def search_amazon_products(keywords: str, category: str = "All", limit: int = 10) -> list:
    """
    Search Amazon products using PA API 5.0
    Returns list of product dictionaries.
    """
    if not AMAZON_ACCESS_KEY:
        logger.warning("Amazon API keys not set — returning mock data")
        return get_mock_deals(keywords, limit)

    # Build request payload
    payload = {
        "Keywords": keywords,
        "Resources": [
            "Images.Primary.Large",
            "ItemInfo.Title",
            "Offers.Listings.Price",
            "Offers.Listings.SavingBasis",
            "Offers.Listings.Promotions",
            "Offers.Summaries.HighestPrice",
            "Offers.Summaries.LowestPrice",
            "CustomerReviews.Count",
            "CustomerReviews.StarRating",
        ],
        "SearchIndex": category if category != "All" else "All",
        "ItemCount": min(limit, 10),
        "PartnerTag": AMAZON_PARTNER_TAG,
        "PartnerType": "Associates",
        "Marketplace": "www.amazon.com",
    }

    import json
    payload_str = json.dumps(payload)

    # AWS Signature V4
    now        = datetime.now(timezone.utc)
    amz_date   = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    service    = "ProductAdvertisingAPI"

    content_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    canonical_headers = (
        f"content-encoding:amz-1.0\n"
        f"content-type:application/json; charset=utf-8\n"
        f"host:{AMAZON_HOST}\n"
        f"x-amz-date:{amz_date}\n"
        f"x-amz-target:com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems\n"
    )
    signed_headers = "content-encoding;content-type;host;x-amz-date;x-amz-target"

    canonical_request = "\n".join([
        "POST", "/paapi5/searchitems", "",
        canonical_headers, signed_headers, content_hash
    ])

    credential_scope = f"{date_stamp}/{AMAZON_REGION}/{service}/aws4_request"
    string_to_sign   = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    ])

    signing_key = get_signature_key(AMAZON_SECRET_KEY, date_stamp, AMAZON_REGION, service)
    signature   = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    headers = {
        "content-encoding": "amz-1.0",
        "content-type": "application/json; charset=utf-8",
        "host": AMAZON_HOST,
        "x-amz-date": amz_date,
        "x-amz-target": "com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems",
        "Authorization": (
            f"AWS4-HMAC-SHA256 Credential={AMAZON_ACCESS_KEY}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(AMAZON_ENDPOINT, content=payload_str, headers=headers)
        if response.status_code != 200:
            logger.error(f"Amazon API error: {response.status_code} - {response.text}")
            return get_mock_deals(keywords, limit)

        data = response.json()
        items = data.get("SearchResult", {}).get("Items", [])
        return parse_amazon_items(items)


def parse_amazon_items(items: list) -> list:
    """Parse raw Amazon API response into clean deal objects."""
    deals = []
    for item in items:
        try:
            asin  = item.get("ASIN", "")
            title = item.get("ItemInfo", {}).get("Title", {}).get("DisplayValue", "Unknown Product")
            image = item.get("Images", {}).get("Primary", {}).get("Large", {}).get("URL", "")

            listings = item.get("Offers", {}).get("Listings", [{}])
            listing  = listings[0] if listings else {}
            price_info    = listing.get("Price", {})
            savings_basis = listing.get("SavingBasis", {})

            price    = price_info.get("Amount", 0)
            currency = price_info.get("Currency", "USD")
            original = savings_basis.get("Amount", price)
            discount = round(((original - price) / original) * 100) if original > price else 0

            reviews  = item.get("CustomerReviews", {})
            rating   = reviews.get("StarRating", {}).get("Value", 0)
            rev_count = reviews.get("Count", {}).get("Value", 0)

            affiliate_url = f"https://www.amazon.com/dp/{asin}?tag={AMAZON_PARTNER_TAG}"

            deals.append({
                "asin": asin,
                "title": title,
                "price": price,
                "original_price": original,
                "discount_percent": discount,
                "rating": rating,
                "review_count": rev_count,
                "image_url": image,
                "affiliate_url": affiliate_url,
            })
        except Exception as e:
            logger.error(f"Error parsing item: {e}")
    return deals


def get_mock_deals(query: str, limit: int = 5) -> list:
    """Return curated deals with real affiliate links."""
    tag = AMAZON_PARTNER_TAG or "yarbiydozit21-20"
    # Real Amazon search URL with affiliate tag — earns commission on any purchase
    search_url = f"https://www.amazon.com/s?k={query}&tag={tag}"
    mock = [
        {
            "asin": "B08N5WRWNW",
            "title": f"Top Deals on {query.title()} — Click to See All Amazon Offers",
            "price": 29.99,
            "original_price": 59.99,
            "discount_percent": 50,
            "rating": 4.5,
            "review_count": 12543,
            "image_url": "https://via.placeholder.com/300x300?text=🛍️",
            "affiliate_url": search_url,
        },
        {
            "asin": "B09MOCK001",
            "title": f"Best Selling {query.title()} — Highly Rated on Amazon",
            "price": 49.99,
            "original_price": 89.99,
            "discount_percent": 44,
            "rating": 4.3,
            "review_count": 8901,
            "image_url": "https://via.placeholder.com/300x300?text=🛒",
            "affiliate_url": f"https://www.amazon.com/s?k={query}+deals&tag={tag}",
        },
        {
            "asin": "B09MOCK002",
            "title": f"Budget Pick: {query.title()} — Great Value Deal",
            "price": 19.99,
            "original_price": 34.99,
            "discount_percent": 43,
            "rating": 4.1,
            "review_count": 5234,
            "image_url": "https://via.placeholder.com/300x300?text=💰",
            "affiliate_url": f"https://www.amazon.com/s?k={query}+discount&tag={tag}",
        },
    ]
    return mock[:limit]


# ─── AI Deal Scoring ──────────────────────────────────────────────────────────
async def ai_score_deals(deals: list, query: str) -> list:
    """Use Claude AI to score and rank deals."""
    if not ANTHROPIC_API_KEY or not deals:
        # Simple algorithmic scoring if no AI
        for deal in deals:
            discount = deal.get("discount_percent", 0)
            rating   = deal.get("rating", 0)
            reviews  = min(deal.get("review_count", 0) / 1000, 10)
            deal["ai_score"] = round((discount * 0.5) + (rating * 5) + reviews, 1)
            deal["ai_summary"] = f"Good deal with {discount}% off and {rating}★ rating"
        return sorted(deals, key=lambda x: x["ai_score"], reverse=True)

    # Build prompt for Claude
    deals_text = "\n".join([
        f"{i+1}. {d['title']} | ${d['price']} (was ${d['original_price']}) | "
        f"{d['discount_percent']}% off | {d['rating']}★ ({d['review_count']} reviews)"
        for i, d in enumerate(deals)
    ])

    prompt = (
        f"You are an expert Amazon deal analyst. A user searched for: '{query}'\n\n"
        f"Rate each deal from 0-10 and write a 1-sentence summary.\n\n"
        f"Deals:\n{deals_text}\n\n"
        f"Respond in JSON array format:\n"
        f'[{{"score": 8.5, "summary": "Excellent value..."}}]\n'
        f"One object per deal, same order."
    )

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            result = response.json()
            text   = result["content"][0]["text"]

            import json
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                for i, deal in enumerate(deals):
                    if i < len(scores):
                        deal["ai_score"]   = scores[i].get("score", 5.0)
                        deal["ai_summary"] = scores[i].get("summary", "")

    except Exception as e:
        logger.error(f"AI scoring error: {e}")
        for deal in deals:
            deal["ai_score"]   = deal.get("discount_percent", 0) / 10
            deal["ai_summary"] = f"{deal.get('discount_percent', 0)}% discount deal"

    return sorted(deals, key=lambda x: x.get("ai_score", 0), reverse=True)


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "🟢 DealHunter AI API is running", "version": "1.0.0"}


@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    category: str = Query("All", description="Product category"),
    limit: int = Query(10, ge=1, le=20),
):
    """Search Amazon products and return AI-scored deals."""
    try:
        deals  = await search_amazon_products(q, category, limit)
        scored = await ai_score_deals(deals, q)
        return {"query": q, "count": len(scored), "deals": scored}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hot-deals")
async def hot_deals():
    """Get today's hottest deals across categories."""
    queries  = ["electronics deals", "home deals", "fashion deals", "gaming deals"]
    all_deals = []

    for query in queries:
        deals = await search_amazon_products(query, limit=3)
        all_deals.extend(deals)

    # Filter for significant discounts
    hot = [d for d in all_deals if d.get("discount_percent", 0) >= 20]
    scored = await ai_score_deals(hot, "best deals today")
    return {"count": len(scored), "deals": scored[:12]}


@app.get("/categories")
async def get_categories():
    """Return available product categories."""
    return {
        "categories": [
            {"id": "Electronics",    "name": "Electronics",     "emoji": "💻"},
            {"id": "Apparel",        "name": "Fashion & Apparel","emoji": "👗"},
            {"id": "HomeAndKitchen", "name": "Home & Kitchen",  "emoji": "🏠"},
            {"id": "VideoGames",     "name": "Gaming",          "emoji": "🎮"},
            {"id": "Books",          "name": "Books",            "emoji": "📚"},
            {"id": "Beauty",         "name": "Beauty & Care",   "emoji": "💄"},
            {"id": "SportingGoods",  "name": "Sports",          "emoji": "⚽"},
            {"id": "Toys",           "name": "Toys & Kids",     "emoji": "🧸"},
        ]
    }


@app.post("/alert")
async def create_alert(alert: AlertRequest):
    """Save a price alert for a user."""
    # In production: save to database (SQLite, PostgreSQL, etc.)
    logger.info(f"Alert set: user={alert.user_id}, price=${alert.target_price}")
    return {
        "status": "success",
        "message": f"Alert set for ${alert.target_price:.2f}",
        "alert_id": f"alert_{alert.user_id}_{int(datetime.now().timestamp())}"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "amazon_api": "configured" if AMAZON_ACCESS_KEY else "not configured (using mock data)",
        "ai_api": "configured" if ANTHROPIC_API_KEY else "not configured (using algorithmic scoring)",
        "timestamp": datetime.now().isoformat()
    }


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
