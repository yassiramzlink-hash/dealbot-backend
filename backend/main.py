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
GROQ_API_KEY           = os.getenv("GROQ_API_KEY", "")
SUPABASE_URL           = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY           = os.getenv("SUPABASE_KEY", "")

# OAuth2 Token endpoint (Creators API - Cognito)
TOKEN_URL    = "https://creatorsapi.auth.us-west-2.amazoncognito.com/oauth2/token"
CREATORS_API = "https://affiliate-program.amazon.com"

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
                f"{CREATORS_API}/creatorsapi/v2/search",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
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


# ─── AI Deal Scoring with Groq ────────────────────────────────────────────────
async def ai_score_deals(deals: list, query: str) -> list:
    """Score deals using Groq AI (free) if available, else algorithmic fallback."""

    # ── Algorithmic fallback (always runs first) ──
    for deal in deals:
        if "ai_score" not in deal:
            discount = deal.get("discount_percent", 0)
            rating   = deal.get("rating", 0)
            reviews  = min(deal.get("review_count", 0) / 1000, 10)
            deal["ai_score"]   = round((discount * 0.5) + (rating * 5) + reviews, 1)
            deal["ai_summary"] = f"{discount}% off with {rating}★ rating."

    # ── Groq AI scoring (if API key is set) ──
    if not GROQ_API_KEY or not deals:
        return sorted(deals, key=lambda x: x.get("ai_score", 0), reverse=True)

    try:
        deals_text = "\n".join([
            f"{i+1}. {d.get('title','')[:60]} | ${d.get('price')} (was ${d.get('original_price')}) | "
            f"{d.get('discount_percent')}% off | {d.get('rating')}★ ({d.get('review_count')} reviews)"
            for i, d in enumerate(deals[:5])
        ])

        prompt = f"""You are a deal analysis expert. A user searched for: "{query}"

Here are the top Amazon deals found:
{deals_text}

For each deal, provide:
1. A score from 0-10 (considering: discount%, rating, review count, value for money)
2. A short 1-sentence summary in English (max 12 words)

Reply ONLY in this exact JSON format (no extra text):
[
  {{"index": 1, "score": 8.5, "summary": "Excellent value with strong reviews and deep discount."}},
  {{"index": 2, "score": 7.2, "summary": "Good mid-range option with reliable customer ratings."}}
]"""

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama3-8b-8192",
                    "max_tokens": 500,
                    "temperature": 0.3,
                    "messages": [
                        {"role": "system", "content": "You are a deal analysis expert. Always reply in valid JSON only."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )

            if response.status_code == 200:
                import json
                content = response.json()["choices"][0]["message"]["content"].strip()
                # Clean JSON if wrapped in backticks
                content = content.replace("```json", "").replace("```", "").strip()
                ai_results = json.loads(content)

                for result in ai_results:
                    idx = result["index"] - 1
                    if 0 <= idx < len(deals):
                        deals[idx]["ai_score"]   = result["score"]
                        deals[idx]["ai_summary"]  = result["summary"]

                logger.info("✅ Groq AI scored deals successfully")
            else:
                logger.warning(f"Groq API error: {response.status_code} - {response.text}")

    except Exception as e:
        logger.warning(f"Groq AI scoring failed, using algorithmic: {e}")

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
        "groq_ai": "enabled ✅" if GROQ_API_KEY else "not configured ❌",
        "supabase": "connected ✅" if SUPABASE_URL else "not configured ❌",
        "partner_tag": AMAZON_PARTNER_TAG,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/debug-auth")
async def debug_auth():
    """
    يختبر كل طرق المصادقة الممكنة ويُرجع تقريراً كاملاً.
    """
    import time
    import json

    client_id     = CREATORS_CLIENT_ID
    client_secret = CREATORS_CLIENT_SECRET

    results = {
        "env_vars": {
            "CREATORS_CLIENT_ID_set":     bool(client_id),
            "CREATORS_CLIENT_ID_length":  len(client_id),
            "CREATORS_CLIENT_ID_value":   client_id[:8] + "..." if client_id else "EMPTY",
            "CREATORS_CLIENT_SECRET_set": bool(client_secret),
            "CREATORS_CLIENT_SECRET_length": len(client_secret),
        },
        "tests": {}
    }

    if not client_id or not client_secret:
        results["conclusion"] = "❌ المتغيرات فارغة — تحقق من Railway Variables"
        return results

    # ── طريقة 1: Basic Auth + scope=creatorsapi/default ──
    try:
        encoded = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://creatorsapi.auth.us-west-2.amazoncognito.com/oauth2/token",
                headers={"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {encoded}"},
                data={"grant_type": "client_credentials", "scope": "creatorsapi/default"}
            )
            results["tests"]["method1_basic_auth_default_scope"] = {
                "status": r.status_code, "response": r.text[:200]
            }
    except Exception as e:
        results["tests"]["method1_basic_auth_default_scope"] = {"error": str(e)}

    # ── طريقة 2: Body params + بدون scope ──
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://creatorsapi.auth.us-west-2.amazoncognito.com/oauth2/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
            )
            results["tests"]["method2_body_params_no_scope"] = {
                "status": r.status_code, "response": r.text[:200]
            }
    except Exception as e:
        results["tests"]["method2_body_params_no_scope"] = {"error": str(e)}

    # ── طريقة 3: Basic Auth + بدون scope ──
    try:
        encoded = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://creatorsapi.auth.us-west-2.amazoncognito.com/oauth2/token",
                headers={"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {encoded}"},
                data={"grant_type": "client_credentials"}
            )
            results["tests"]["method3_basic_auth_no_scope"] = {
                "status": r.status_code, "response": r.text[:200]
            }
    except Exception as e:
        results["tests"]["method3_basic_auth_no_scope"] = {"error": str(e)}

    # ── طريقة 4: LWA endpoint ──
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://api.amazon.com/auth/o2/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret, "scope": "creatorsapi/default"}
            )
            results["tests"]["method4_lwa_endpoint"] = {
                "status": r.status_code, "response": r.text[:200]
            }
    except Exception as e:
        results["tests"]["method4_lwa_endpoint"] = {"error": str(e)}

    # ── الخلاصة ──
    success = [k for k, v in results["tests"].items() if v.get("status") == 200]
    if success:
        results["conclusion"] = f"✅ نجحت الطريقة: {success[0]}"
    else:
        results["conclusion"] = "❌ كل الطرق فشلت — المشكلة في الـ credentials نفسها"

    return results


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


# ─── Supabase Helper ──────────────────────────────────────────────────────────
async def supabase_request(method: str, table: str, data: dict = None, params: str = "") -> dict:
    """Generic Supabase REST API request."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    url = f"{SUPABASE_URL}/rest/v1/{table}{params}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if method == "GET":
                r = await client.get(url, headers=headers)
            elif method == "POST":
                r = await client.post(url, headers=headers, json=data)
            elif method == "PATCH":
                r = await client.patch(url, headers=headers, json=data)
            elif method == "DELETE":
                r = await client.delete(url, headers=headers)
            if r.status_code in [200, 201]:
                return r.json()
            logger.error(f"Supabase error: {r.status_code} - {r.text}")
            return None
    except Exception as e:
        logger.error(f"Supabase request failed: {e}")
        return None


# ─── User Functions ───────────────────────────────────────────────────────────
async def upsert_user(telegram_id: int, username: str = None, first_name: str = None):
    """Register or update user."""
    # Check if exists
    existing = await supabase_request("GET", "users", params=f"?telegram_id=eq.{telegram_id}")
    if existing:
        # Update last_active
        await supabase_request("PATCH", "users",
            data={"last_active": datetime.now().isoformat()},
            params=f"?telegram_id=eq.{telegram_id}")
        return existing[0]
    else:
        # Create new user
        return await supabase_request("POST", "users", data={
            "telegram_id": telegram_id,
            "username": username,
            "first_name": first_name,
        })


async def save_search(telegram_id: int, query: str, results_count: int):
    """Save search to history."""
    await supabase_request("POST", "search_history", data={
        "telegram_id": telegram_id,
        "query": query,
        "results_count": results_count,
    })
    # Update total_searches counter
    existing = await supabase_request("GET", "users", params=f"?telegram_id=eq.{telegram_id}&select=total_searches")
    if existing:
        count = existing[0].get("total_searches", 0) + 1
        await supabase_request("PATCH", "users",
            data={"total_searches": count},
            params=f"?telegram_id=eq.{telegram_id}")


async def get_recommendations(telegram_id: int) -> list:
    """Get personalized recommendations based on search history."""
    history = await supabase_request("GET", "search_history",
        params=f"?telegram_id=eq.{telegram_id}&order=searched_at.desc&limit=5")
    if not history:
        return []
    return [h["query"] for h in history]


# ─── Supabase API Routes ───────────────────────────────────────────────────────
@app.get("/user/register")
async def register_user(telegram_id: int, username: str = None, first_name: str = None):
    """Register or update user in database."""
    result = await upsert_user(telegram_id, username, first_name)
    if result:
        return {"success": True, "user": result}
    return {"success": False, "message": "Supabase not configured"}


@app.get("/user/favorite")
async def add_favorite(
    telegram_id: int,
    asin: str,
    title: str = None,
    price: float = None,
    affiliate_url: str = None
):
    """Add product to favorites."""
    await upsert_user(telegram_id)
    result = await supabase_request("POST", "favorites", data={
        "telegram_id": telegram_id,
        "asin": asin,
        "title": title,
        "price": price,
        "affiliate_url": affiliate_url,
    })
    if result:
        return {"success": True, "message": "Added to favorites ❤️"}
    return {"success": False, "message": "Already in favorites or error"}


@app.get("/user/unfavorite")
async def remove_favorite(telegram_id: int, asin: str):
    """Remove product from favorites."""
    await supabase_request("DELETE", "favorites",
        params=f"?telegram_id=eq.{telegram_id}&asin=eq.{asin}")
    return {"success": True, "message": "Removed from favorites"}


@app.get("/user/favorites")
async def get_favorites(telegram_id: int):
    """Get user's favorite products."""
    favorites = await supabase_request("GET", "favorites",
        params=f"?telegram_id=eq.{telegram_id}&order=added_at.desc")
    return {"telegram_id": telegram_id, "count": len(favorites or []), "favorites": favorites or []}


@app.get("/user/history")
async def get_history(telegram_id: int):
    """Get user's search history."""
    history = await supabase_request("GET", "search_history",
        params=f"?telegram_id=eq.{telegram_id}&order=searched_at.desc&limit=20")
    return {"telegram_id": telegram_id, "history": history or []}


@app.get("/user/recommendations")
async def get_user_recommendations(telegram_id: int):
    """Get personalized recommendations based on history."""
    queries = await get_recommendations(telegram_id)
    if not queries:
        # Default recommendations for new users
        return {"recommendations": await search_amazon_products("best deals today", limit=5)}
    # Search based on last query
    deals = await search_amazon_products(queries[0], limit=5)
    scored = await ai_score_deals(deals, queries[0])
    return {
        "based_on": queries[0],
        "recommendations": scored
    }


@app.get("/stats")
async def get_stats():
    """Get bot statistics."""
    users = await supabase_request("GET", "users", params="?select=count")
    favorites = await supabase_request("GET", "favorites", params="?select=count")
    searches = await supabase_request("GET", "search_history", params="?select=count")
    return {
        "total_users": users[0]["count"] if users else 0,
        "total_favorites": favorites[0]["count"] if favorites else 0,
        "total_searches": searches[0]["count"] if searches else 0,
    }

# ─── Messaging System ─────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = "https://api.telegram.org"


async def send_telegram_message(telegram_id: int, text: str, keyboard: list = None) -> bool:
    """إرسال رسالة لمستخدم عبر Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set")
        return False
    payload = {"chat_id": telegram_id, "text": text, "parse_mode": "Markdown"}
    if keyboard:
        payload["reply_markup"] = {"inline_keyboard": keyboard}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json=payload
            )
            return r.status_code == 200
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


@app.get("/broadcast")
async def broadcast(message: str, secret: str = ""):
    """إرسال رسالة يدوية لكل المستخدمين."""
    if secret != os.getenv("ADMIN_SECRET", "dealbot2026"):
        return {"error": "Unauthorized ❌"}
    users = await supabase_request("GET", "users", params="?select=telegram_id")
    if not users:
        return {"error": "No users found"}
    sent, failed = 0, 0
    for user in users:
        success = await send_telegram_message(user["telegram_id"], message)
        if success:
            sent += 1
        else:
            failed += 1
    return {"success": True, "sent": sent, "failed": failed, "total": len(users)}


@app.get("/daily-deals")
async def send_daily_deals(secret: str = ""):
    """إرسال عروض يومية تلقائية لكل المستخدمين."""
    if secret != os.getenv("ADMIN_SECRET", "dealbot2026"):
        return {"error": "Unauthorized ❌"}
    # جلب أفضل العروض
    deals = await search_amazon_products("best deals today", limit=5)
    scored = await ai_score_deals(deals, "daily deals")
    top3 = scored[:3]
    # بناء الرسالة
    msg = "🔥 *Today's Top Deals* — Don't miss out!\n\n"
    keyboard = []
    for i, deal in enumerate(top3, 1):
        msg += f"*{i}. {deal.get('title','')[:40]}...*\n"
        msg += f"💰 ${deal.get('price','N/A')} | 🏷️ {deal.get('discount_percent',0)}% OFF\n\n"
        if deal.get("affiliate_url"):
            keyboard.append([{"text": f"🛒 Buy #{i}", "url": deal["affiliate_url"]}])
    # إرسال لكل المستخدمين
    users = await supabase_request("GET", "users", params="?select=telegram_id")
    if not users:
        return {"error": "No users found"}
    sent, failed = 0, 0
    for user in users:
        success = await send_telegram_message(user["telegram_id"], msg, keyboard)
        if success:
            sent += 1
        else:
            failed += 1
    return {"success": True, "sent": sent, "failed": failed, "deals_count": len(top3)}


@app.get("/welcome/{telegram_id}")
async def send_welcome(telegram_id: int, first_name: str = "there"):
    """إرسال رسالة ترحيب لمستخدم جديد."""
    msg = (
        f"👋 Welcome *{first_name}* to DealHunter AI!\n\n"
        "🎯 Here's what you can do:\n"
        "🔍 Search any product for deals\n"
        "❤️ Save favorites\n"
        "🔥 Get daily deal alerts\n\n"
        "Type any product name to start! 🚀"
    )
    keyboard = [[{"text": "🛍️ Find Deals Now", "url": "https://t.me/deealshunter_ai_bot"}]]
    success = await send_telegram_message(telegram_id, msg, keyboard)
    return {"success": success}


# ─── Telegram Channel Parser ──────────────────────────────────────────────────
import re

def parse_channel_message(text: str, image_url: str = None) -> dict:
    """
    يحلل رسائل قناة @USA_Deals_and_Coupons ويستخرج بيانات المنتج.
    يدعم 3 أشكال مختلفة من الرسائل.
    """
    if not text:
        return None

    deal = {
        "title": None,
        "price": None,
        "original_price": None,
        "discount_percent": 0,
        "rating": None,
        "reviews": None,
        "affiliate_url": None,
        "image_url": image_url,
        "category": None,
        "is_prime": False,
        "is_price_error": False,
        "source": "telegram_channel",
    }

    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    # ── العنوان: السطر بعد Deal Alert / Great Deal / PRICE ERROR ──
    trigger_keywords = ["deal alert", "great deal", "price error", "special deal", "hot deal", "flash deal"]
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in trigger_keywords):
            if i + 1 < len(lines):
                deal["title"] = lines[i + 1]
            break

    # إذا لم يُوجد title جرب أول سطر نصي طويل
    if not deal["title"]:
        for line in lines:
            if len(line) > 20 and not line.startswith("http") and not line.startswith("#"):
                deal["title"] = line
                break

    # ── السعر الحالي ──
    price_patterns = [
        r'(?:Error Price:|Only|💰|💵|🏷️|Price:)?\s*\$?([\d,]+\.?\d*)\s*\$?',
        r'([\d,]+\.?\d*)\$',
    ]
    for pattern in price_patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                deal["price"] = float(matches[0].replace(",", ""))
                break
            except:
                pass

    # ── السعر الأصلي (مشطوب) ──
    original_patterns = [
        r'(?:Should Be:|~~\$?|Was:?\s*)\$?([\d,]+\.?\d*)',
        r'(?:44,98|41,99|Should Be)[^\d]*([\d,]+\.?\d*)',
        r'([\d,]+\.?\d*)\$?\s*(?:~~|→)',
    ]
    for pattern in original_patterns:
        m = re.search(pattern, text)
        if m:
            try:
                orig = float(m.group(1).replace(",", "."))
                if deal["price"] and orig > deal["price"]:
                    deal["original_price"] = orig
                    break
            except:
                pass

    # ── نسبة الخصم ──
    discount_match = re.search(r'(\d+)%', text)
    if discount_match:
        deal["discount_percent"] = int(discount_match.group(1))
    elif deal["price"] and deal["original_price"]:
        deal["discount_percent"] = round((1 - deal["price"] / deal["original_price"]) * 100)

    # ── التقييم وعدد المراجعات ──
    rating_match = re.search(r'([\d.]+)\s*/\s*5', text)
    if rating_match:
        deal["rating"] = float(rating_match.group(1))

    reviews_match = re.search(r'([\d,]+)\s*Reviews?', text, re.IGNORECASE)
    if reviews_match:
        deal["reviews"] = int(reviews_match.group(1).replace(",", ""))

    # ── رابط Amazon ──
    url_match = re.search(r'(https?://(?:www\.amazon\.com|amzn\.to)/[^\s\)]+)', text)
    if url_match:
        deal["affiliate_url"] = url_match.group(1)

    # ── الفئة من الهاشتاق ──
    hashtag_match = re.search(r'#(\w+)', text)
    if hashtag_match:
        deal["category"] = hashtag_match.group(1)

    # ── PRIME + Price Error ──
    deal["is_prime"] = "PRIME" in text.upper()
    deal["is_price_error"] = "PRICE ERROR" in text.upper() or "PRICING MISTAKE" in text.upper()

    # تجاهل الرسائل بدون رابط أو عنوان
    if not deal["affiliate_url"] or not deal["title"]:
        return None

    return deal


# ─── Fetch from Telegram Channel via Bot API ──────────────────────────────────
_channel_cache = {"deals": [], "last_fetch": 0}

async def fetch_channel_deals(limit: int = 20) -> list:
    """سحب أحدث المنتجات من القناة عبر Telegram Bot API."""
    import time

    # Cache لمدة ساعة
    if _channel_cache["deals"] and time.time() - _channel_cache["last_fetch"] < 3600:
        return _channel_cache["deals"]

    if not TELEGRAM_BOT_TOKEN:
        return []

    channel = "@USA_Deals_and_Coupons"
    deals = []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params={"limit": 100, "allowed_updates": ["channel_post"]}
            )
            if r.status_code != 200:
                # جرب forwardFromChat
                logger.warning(f"getUpdates failed: {r.text[:100]}")
                return await fetch_channel_via_web(limit)

            data = r.json()
            posts = [u.get("channel_post") for u in data.get("result", []) if u.get("channel_post")]

            for post in posts[-limit:]:
                text = post.get("text") or post.get("caption") or ""
                # صورة المنتج
                image_url = None
                if post.get("photo"):
                    # أكبر صورة متاحة
                    photos = post["photo"]
                    file_id = photos[-1]["file_id"]
                    img_r = await client.get(
                        f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/getFile",
                        params={"file_id": file_id}
                    )
                    if img_r.status_code == 200:
                        file_path = img_r.json()["result"]["file_path"]
                        image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"

                deal = parse_channel_message(text, image_url)
                if deal:
                    deals.append(deal)

    except Exception as e:
        logger.error(f"Channel fetch error: {e}")
        return await fetch_channel_via_web(limit)

    _channel_cache["deals"] = deals
    _channel_cache["last_fetch"] = time.time()
    return deals


async def fetch_channel_via_web(limit: int = 20) -> list:
    """سحب من preview القناة العامة كـ fallback."""
    import html as html_module
    deals = []
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            r = await client.get(
                "https://t.me/s/USA_Deals_and_Coupons",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            if r.status_code == 200:
                raw = r.text
                # استخراج الرسائل مع الصور
                post_blocks = re.findall(
                    r'<div class="tgme_widget_message_wrap[^>]*>.*?</div>\s*</div>\s*</div>',
                    raw, re.DOTALL
                )
                for block in post_blocks[-limit:]:
                    # استخراج النص
                    text_match = re.search(
                        r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
                        block, re.DOTALL
                    )
                    # استخراج الصورة
                    img_match = re.search(r"background-image:url\('(https?://[^']+)'\)", block)
                    if not img_match:
                        img_match = re.search(r'src="(https?://[^"]+)"', block)


                    if not text_match:
                        continue

                    msg_html = text_match.group(1)
                    # تنظيف HTML بشكل كامل
                    text = re.sub(r'<br\s*/?>', '\n', msg_html)
                    text = re.sub(r'<a[^>]+href="([^"]+)"[^>]*>.*?</a>', r'\1', text, flags=re.DOTALL)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = html_module.unescape(text)
                    text = re.sub(r' +', ' ', text).strip()

                    image_url = img_match.group(1) if img_match else None
                    deal = parse_channel_message(text, image_url)
                    if deal:
                        deals.append(deal)
    except Exception as e:
        logger.error(f"Web channel fetch error: {e}")
    return deals


@app.get("/channel-deals")
async def get_channel_deals(limit: int = 10):
    """جلب أحدث العروض من القناة."""
    deals = await fetch_channel_deals(limit)
    if not deals:
        deals = await fetch_channel_via_web(limit)
    return {
        "source": "telegram_channel @USA_Deals_and_Coupons",
        "count": len(deals),
        "deals": deals
    }


@app.get("/channel-deals/refresh")
async def refresh_channel_deals():
    """إعادة تحميل العروض من القناة (مسح الـ cache)."""
    _channel_cache["last_fetch"] = 0
    deals = await fetch_channel_deals(20)
    if not deals:
        deals = await fetch_channel_via_web(20)
    return {"refreshed": True, "count": len(deals), "deals": deals}
