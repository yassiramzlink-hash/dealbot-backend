"""
🔧 DealHunter AI - Backend Server (Creators API v2.1)
======================================================
Uses Amazon Creators API with OAuth 2.0 authentication.
"""

import os
import re
import base64
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
CREATORS_CLIENT_ID     = os.getenv("CREATORS_CLIENT_ID", "")
CREATORS_CLIENT_SECRET = os.getenv("CREATORS_CLIENT_SECRET", "")
AMAZON_PARTNER_TAG     = os.getenv("AMAZON_PARTNER_TAG", "tarek-us-20")

# OAuth2 Token endpoint (Creators API)
TOKEN_URL    = "https://creatorsapi.auth.us-west-2.amazoncognito.com/oauth2/token"
CREATORS_API = "https://paapi5-na.amazon.com"

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="DealHunter AI API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Token Cache ──────────────────────────────────────────────────────────────
_token_cache = {"token": None, "expires_at": 0}


async def get_access_token() -> Optional[str]:
    """Get OAuth2 access token using client_credentials grant."""
    import time

    # Return cached token if still valid
    if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    if not CREATORS_CLIENT_ID or not CREATORS_CLIENT_SECRET:
        logger.warning("Creators API credentials not set")
        return None

    # Basic Auth header: base64(client_id:client_secret)
    credentials = f"{CREATORS_CLIENT_ID}:{CREATORS_CLIENT_SECRET}"
    encoded     = base64.b64encode(credentials.encode()).decode()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                TOKEN_URL,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": f"Basic {encoded}",
                },
                data={
                    "grant_type": "client_credentials",
                    "scope": "creatorsapi/default",
                }
            )

            if response.status_code == 200:
                data = response.json()
                token = data.get("access_token")
                expires_in = data.get("expires_in", 3600)
                _token_cache["token"] = token
                _token_cache["expires_at"] = time.time() + expires_in
                logger.info("✅ Got Creators API token successfully")
                return token
            else:
                logger.error(f"Token error: {response.status_code} - {response.text}")
                return None

    except Exception as e:
        logger.error(f"Token request failed: {e}")
        return None


# ─── Search Products via Creators API ─────────────────────────────────────────
async def search_amazon_products(keywords: str, category: str = "All", limit: int = 10) -> list:
    """Search Amazon products using Creators API (OAuth2)."""
    token = await get_access_token()

    if not token:
        logger.warning("No token available — using mock deals")
        return get_mock_deals(keywords, limit)

    payload = {
        "Keywords": keywords,
        "Resources": [
            "Images.Primary.Large",
            "ItemInfo.Title",
            "Offers.Listings.Price",
            "Offers.Listings.SavingBasis",
            "CustomerReviews.Count",
            "CustomerReviews.StarRating",
        ],
        "SearchIndex": category if category != "All" else "All",
        "ItemCount": min(limit, 10),
        "PartnerTag": AMAZON_PARTNER_TAG,
        "PartnerType": "Associates",
        "Marketplace": "www.amazon.com",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{CREATORS_API}/paapi5/searchitems",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                    "x-amz-target": "com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems",
                },
                json=payload
            )

            if response.status_code == 200:
                data  = response.json()
                items = data.get("SearchResult", {}).get("Items", [])
                logger.info(f"✅ Got {len(items)} products from Creators API")
                return parse_amazon_items(items)
            else:
                logger.error(f"Creators API error: {response.status_code} - {response.text}")
                return get_mock_deals(keywords, limit)

    except Exception as e:
        logger.error(f"Creators API request failed: {e}")
        return get_mock_deals(keywords, limit)


def parse_amazon_items(items: list) -> list:
    """Parse Amazon API response into clean deal objects."""
    deals = []
    for item in items:
        try:
            asin  = item.get("ASIN", "")
            title = item.get("ItemInfo", {}).get("Title", {}).get("DisplayValue", "Unknown")
            image = item.get("Images", {}).get("Primary", {}).get("Large", {}).get("URL", "")

            listings      = item.get("Offers", {}).get("Listings", [{}])
            listing       = listings[0] if listings else {}
            price_info    = listing.get("Price", {})
            savings_basis = listing.get("SavingBasis", {})

            price    = price_info.get("Amount", 0)
            original = savings_basis.get("Amount", price)
            discount = round(((original - price) / original) * 100) if original > price else 0

            reviews   = item.get("CustomerReviews", {})
            rating    = reviews.get("StarRating", {}).get("Value", 0)
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
    """Return mock deals with real affiliate links as fallback."""
    tag = AMAZON_PARTNER_TAG or "tarek-us-20"
    mock = [
        {
            "asin": "MOCK001",
            "title": f"Top Deals on {query.title()} — Click to See All Amazon Offers",
            "price": 29.99,
            "original_price": 59.99,
            "discount_percent": 50,
            "rating": 4.5,
            "review_count": 12543,
            "image_url": "",
            "affiliate_url": f"https://www.amazon.com/s?k={query}&tag={tag}",
            "ai_score": 8.5,
            "ai_summary": "Great discount — click to see live prices on Amazon."
        },
        {
            "asin": "MOCK002",
            "title": f"Best Selling {query.title()} — Highly Rated",
            "price": 49.99,
            "original_price": 89.99,
            "discount_percent": 44,
            "rating": 4.3,
            "review_count": 8901,
            "image_url": "",
            "affiliate_url": f"https://www.amazon.com/s?k={query}+deals&tag={tag}",
            "ai_score": 7.8,
            "ai_summary": "Popular choice with strong customer satisfaction."
        },
        {
            "asin": "MOCK003",
            "title": f"Budget Pick: {query.title()} — Best Value",
            "price": 19.99,
            "original_price": 34.99,
            "discount_percent": 43,
            "rating": 4.1,
            "review_count": 5234,
            "image_url": "",
            "affiliate_url": f"https://www.amazon.com/s?k={query}+best+value&tag={tag}",
            "ai_score": 7.2,
            "ai_summary": "Budget-friendly option with great value for money."
        },
    ]
    return mock[:limit]


# ─── AI Deal Scoring ──────────────────────────────────────────────────────────
async def ai_score_deals(deals: list, query: str) -> list:
    """Score deals algorithmically."""
    for deal in deals:
        if "ai_score" not in deal:
            discount  = deal.get("discount_percent", 0)
            rating    = deal.get("rating", 0)
            reviews   = min(deal.get("review_count", 0) / 1000, 10)
            deal["ai_score"]   = round((discount * 0.5) + (rating * 5) + reviews, 1)
            deal["ai_summary"] = f"{discount}% off with {rating}★ rating — solid deal."
    return sorted(deals, key=lambda x: x.get("ai_score", 0), reverse=True)


# ─── API Routes ───────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "🟢 DealHunter AI API v2.0 running", "api": "Creators API v2.1"}


@app.get("/health")
async def health():
    token = await get_access_token()
    return {
        "status": "healthy",
        "creators_api": "connected ✅" if token else "not connected ❌",
        "partner_tag": AMAZON_PARTNER_TAG,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/search")
async def search(
    q: str = Query(...),
    category: str = Query("All"),
    limit: int = Query(10, ge=1, le=20),
):
    try:
        deals  = await search_amazon_products(q, category, limit)
        scored = await ai_score_deals(deals, q)
        return {"query": q, "count": len(scored), "deals": scored}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hot-deals")
async def hot_deals():
    queries   = ["electronics sale", "kitchen deals", "gaming deals", "fashion sale"]
    all_deals = []
    for q in queries:
        deals = await search_amazon_products(q, limit=3)
        all_deals.extend(deals)
    scored = await ai_score_deals(all_deals, "hot deals")
    return {"count": len(scored), "deals": scored[:12]}


@app.get("/categories")
async def get_categories():
    return {"categories": [
        {"id": "Electronics",    "name": "Electronics",      "emoji": "💻"},
        {"id": "Apparel",        "name": "Fashion & Apparel", "emoji": "👗"},
        {"id": "HomeAndKitchen", "name": "Home & Kitchen",   "emoji": "🏠"},
        {"id": "VideoGames",     "name": "Gaming",           "emoji": "🎮"},
        {"id": "Books",          "name": "Books",             "emoji": "📚"},
        {"id": "Beauty",         "name": "Beauty & Care",    "emoji": "💄"},
    ]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
