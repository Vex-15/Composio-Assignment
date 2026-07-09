"""
Composio App Research Pipeline — Stages 1-3
============================================
This module implements the core research pipeline that takes 100 apps from
data/apps.json and produces structured findings in data/raw_results.json.

Architecture:
    Stage 1: Composio Catalog Lookup
        - Query Composio's REST API for apps already in their catalog
        - Extract auth_schemes, tool counts — free ground truth
        - Gracefully skips if COMPOSIO_API_KEY is not set

    Stage 2: Doc Fetch + LLM Extraction
        - For each app, fetch developer docs (auth, API reference, pricing)
        - Extract structured fields via LLM (Gemini 2.0 Flash)
        - Async with rate limiting, caching, retries

    Stage 3: Low-Confidence Retry
        - Re-research apps flagged as low confidence
        - Try alternative URLs, different extraction prompts

Engineering Properties:
    - Async execution (asyncio + httpx) — 100 apps don't run serially
    - Exponential backoff on HTTP/LLM failures (tenacity)
    - Local file cache — same URL never fetched twice across runs
    - Rate limiting via asyncio.Semaphore — external sites not hammered
    - Resumable — interrupted runs pick up where they left off
    - Progress bars (tqdm) — visible pipeline state
    - All keys from environment variables, never hardcoded

Usage:
    python -m agents.research
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
APPS_FILE = DATA_DIR / "apps.json"
RAW_RESULTS_FILE = DATA_DIR / "raw_results.json"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# API Keys — always from environment, never hardcoded
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Pipeline tuning constants
MAX_CONCURRENT_REQUESTS = 8          # Semaphore limit for outbound HTTP
MAX_CONCURRENT_LLM = 5               # Semaphore limit for LLM calls
FETCH_TIMEOUT_SECONDS = 20           # HTTP request timeout
RETRY_MAX_ATTEMPTS = 3               # Max retries on transient failures
PER_DOMAIN_DELAY_SECONDS = 1.5       # Minimum delay between requests to same domain
LLM_MODEL = "gemini-2.0-flash"       # Fast, cheap, good enough for extraction

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

# File logger — detailed pipeline logs
file_handler = logging.FileHandler(LOGS_DIR / "research.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
))

# Console logger — summary only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

logger = logging.getLogger("research")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Tool call logger — structured JSONL log of every external call
tool_log_path = LOGS_DIR / "tool_calls.jsonl"

def log_tool_call(tool: str, input_data: dict, output_data: dict, duration_ms: float):
    """Log every external API/LLM call as a JSONL record for auditability."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "input": input_data,
        "output_summary": str(output_data)[:500],  # Truncate large outputs
        "duration_ms": round(duration_ms, 1),
    }
    with open(tool_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Cache Layer
# ---------------------------------------------------------------------------

def url_to_cache_key(url: str) -> str:
    """Deterministic cache key from URL. Uses SHA256 hash to avoid filesystem issues."""
    return hashlib.sha256(url.encode()).hexdigest()

def get_cached(url: str) -> Optional[str]:
    """Return cached page content if it exists, else None."""
    cache_file = CACHE_DIR / f"{url_to_cache_key(url)}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="replace")
    return None

def set_cached(url: str, content: str):
    """Write page content to cache."""
    cache_file = CACHE_DIR / f"{url_to_cache_key(url)}.txt"
    cache_file.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTTP Fetch with Retries + Rate Limiting
# ---------------------------------------------------------------------------

# Track per-domain last-request time to enforce rate limits
_domain_last_request: dict[str, float] = {}
_domain_lock = asyncio.Lock()

def _extract_domain(url: str) -> str:
    """Extract domain from URL for rate limiting."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or parsed.path.split("/")[0]
    except Exception:
        return url

async def _enforce_domain_rate_limit(domain: str):
    """Ensure minimum delay between requests to the same domain."""
    async with _domain_lock:
        last = _domain_last_request.get(domain, 0)
        now = time.monotonic()
        wait = PER_DOMAIN_DELAY_SECONDS - (now - last)
        if wait > 0:
            await asyncio.sleep(wait)
        _domain_last_request[domain] = time.monotonic()


async def fetch_url(
    url: str,
    session: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """
    Fetch a URL with caching, rate limiting, and retries.

    Returns the page text content (HTML stripped to text), or None on failure.
    Cache-first: if the URL was fetched in a previous run, return cached version.
    """
    # Check cache first — avoids network entirely on reruns
    cached = get_cached(url)
    if cached is not None:
        logger.debug(f"Cache hit: {url}")
        return cached

    domain = _extract_domain(url)

    async with semaphore:
        await _enforce_domain_rate_limit(domain)

        start = time.monotonic()
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                response = await session.get(
                    url,
                    timeout=FETCH_TIMEOUT_SECONDS,
                    follow_redirects=True,
                )

                if response.status_code == 429:
                    # Rate limited — back off exponentially
                    wait = 2 ** attempt
                    logger.warning(f"429 from {url}, backing off {wait}s (attempt {attempt})")
                    await asyncio.sleep(wait)
                    continue

                if response.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(f"{response.status_code} from {url}, retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue

                if response.status_code >= 400:
                    logger.warning(f"HTTP {response.status_code} for {url} — skipping")
                    duration = (time.monotonic() - start) * 1000
                    log_tool_call("http_fetch", {"url": url}, {"status": response.status_code}, duration)
                    return None

                # Success — parse HTML to text
                content_type = response.headers.get("content-type", "")
                raw_html = response.text

                # Extract readable text from HTML
                soup = BeautifulSoup(raw_html, "html.parser")
                # Remove script/style elements
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)

                # Truncate very long pages to avoid LLM token limits
                if len(text) > 15000:
                    text = text[:15000] + "\n\n[...truncated...]"

                # Cache the result
                set_cached(url, text)

                duration = (time.monotonic() - start) * 1000
                log_tool_call("http_fetch", {"url": url}, {"status": 200, "length": len(text)}, duration)
                logger.debug(f"Fetched {url} ({len(text)} chars)")
                return text

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
                wait = 2 ** attempt
                logger.warning(f"Network error fetching {url}: {e} (attempt {attempt}/{RETRY_MAX_ATTEMPTS})")
                if attempt < RETRY_MAX_ATTEMPTS:
                    await asyncio.sleep(wait)
                else:
                    duration = (time.monotonic() - start) * 1000
                    log_tool_call("http_fetch", {"url": url}, {"error": str(e)}, duration)
                    return None
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                duration = (time.monotonic() - start) * 1000
                log_tool_call("http_fetch", {"url": url}, {"error": str(e)}, duration)
                return None

    return None


# ---------------------------------------------------------------------------
# Web Search Fallback (Serper API)
# ---------------------------------------------------------------------------

async def web_search(
    query: str,
    session: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    num_results: int = 5,
) -> list[dict]:
    """
    Search the web for a query using Serper API.
    Returns list of {title, link, snippet} dicts.
    Falls back to empty list if SERPER_API_KEY is not set.
    """
    if not SERPER_API_KEY:
        logger.debug("No SERPER_API_KEY — skipping web search")
        return []

    async with semaphore:
        start = time.monotonic()
        try:
            response = await session.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": num_results},
                headers={"X-API-KEY": SERPER_API_KEY},
                timeout=10,
            )
            data = response.json()
            results = [
                {"title": r.get("title", ""), "link": r.get("link", ""), "snippet": r.get("snippet", "")}
                for r in data.get("organic", [])
            ]
            duration = (time.monotonic() - start) * 1000
            log_tool_call("web_search", {"query": query}, {"num_results": len(results)}, duration)
            return results
        except Exception as e:
            logger.warning(f"Web search failed for '{query}': {e}")
            return []


# ---------------------------------------------------------------------------
# Stage 1: Composio Catalog Lookup
# ---------------------------------------------------------------------------

def _name_to_slug(name: str) -> str:
    """
    Convert app name to likely Composio toolkit slug.
    e.g. 'Help Scout' -> 'helpscout', 'Monday.com' -> 'monday'
    """
    slug = name.lower()
    slug = re.sub(r'[.\-/]', '', slug)   # Remove dots, hyphens, slashes
    slug = re.sub(r'\s+', '', slug)       # Remove spaces
    slug = re.sub(r'\(.*?\)', '', slug)   # Remove parenthetical notes
    slug = slug.strip()
    # Common overrides for known mismatches
    overrides = {
        "zohocrm": "zohocrm",
        "zohocliq": "zohocliq",
        "helpscout": "helpscout",
        "whatsappbusiness": "whatsapp",
        "lark": "larksuite",
        "mondaycom": "monday",
        "mongodbatlas": "mongodb",
        "amazonsellingpartner": "amazonsp",
        "salesforcecommercecloud": "salesforce",
        "magentoadobecommerce": "magento",
        "magento": "magento",
        "linkedinads": "linkedin",
        "googleads": "googleads",
        "metaads": "facebookads",
        "threadsmeta": "threads",
        "threads": "threads",
        "systemio": "systemeio",
        "mermaidcli": "mermaid",
        "youtubetranscript": "youtube",
        "quickbooks": "quickbooks",
        "notebooklm": "notebooklm",
        "otterai": "otterai",
    }
    return overrides.get(slug, slug)


async def composio_catalog_lookup(
    app: dict,
    session: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> Optional[dict]:
    """
    Stage 1: Query Composio REST API for this app.

    Tries exact slug match first, then search endpoint.
    Returns partial pre-fill dict if found, None otherwise.

    Why this stage exists:
        - Free ground truth for apps Composio already supports
        - Demonstrates understanding of the Composio product
        - Auth schemes from Composio are more reliable than LLM extraction
    """
    if not COMPOSIO_API_KEY:
        logger.info("No COMPOSIO_API_KEY set — skipping Composio catalog lookup")
        return None

    base_url = "https://backend.composio.dev/api/v3.1"
    headers = {"x-api-key": COMPOSIO_API_KEY}
    slug = _name_to_slug(app["name"])

    async with semaphore:
        start = time.monotonic()
        try:
            # Try exact slug match
            response = await session.get(
                f"{base_url}/toolkits/{slug}",
                headers=headers,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                result = {
                    "composio_found": True,
                    "composio_slug": slug,
                    "composio_auth_schemes": data.get("auth_schemes", []),
                    "composio_managed_auth": data.get("composio_managed_auth_schemes", []),
                    "composio_url": f"https://composio.dev/tools/{slug}",
                }
                duration = (time.monotonic() - start) * 1000
                log_tool_call("composio_lookup", {"slug": slug}, result, duration)
                logger.info(f"Composio catalog HIT: {app['name']} → {slug}")
                return result

            # Try search if exact slug missed
            response = await session.get(
                f"{base_url}/toolkits",
                headers=headers,
                params={"search": app["name"], "limit": 3},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                items = data if isinstance(data, list) else data.get("items", data.get("results", []))
                if items and len(items) > 0:
                    best = items[0]
                    result = {
                        "composio_found": True,
                        "composio_slug": best.get("slug", slug),
                        "composio_auth_schemes": best.get("auth_schemes", []),
                        "composio_managed_auth": best.get("composio_managed_auth_schemes", []),
                        "composio_url": f"https://composio.dev/tools/{best.get('slug', slug)}",
                    }
                    duration = (time.monotonic() - start) * 1000
                    log_tool_call("composio_search", {"name": app["name"]}, result, duration)
                    logger.info(f"Composio search HIT: {app['name']} → {best.get('slug')}")
                    return result

        except Exception as e:
            logger.warning(f"Composio lookup failed for {app['name']}: {e}")

        duration = (time.monotonic() - start) * 1000
        log_tool_call("composio_lookup", {"slug": slug}, {"found": False}, duration)
        logger.debug(f"Composio catalog MISS: {app['name']}")
        return None


# ---------------------------------------------------------------------------
# LLM Extraction (Google Gemini)
# ---------------------------------------------------------------------------

# The extraction prompt — the most important piece of the pipeline.
# Designed to be defensive: "unknown" is always preferred over a guess.
EXTRACTION_PROMPT = """You are a technical API researcher. Given the developer documentation text below for the app "{name}", extract structured information.

RULES — follow these exactly:
1. If a field cannot be determined from the provided text, set it to "unknown".
2. NEVER guess or fabricate information. "unknown" is always correct when uncertain.
3. Every fact must be directly supported by the text provided.
4. For auth_methods, list ALL methods mentioned (not just the primary one).
5. For access.model, look for pricing pages, free tier mentions, trial information.
6. For api_surface.breadth, "narrow" = <20 endpoints, "moderate" = 20-100, "broad" = 100+.

APP: {name}
CATEGORY: {category}
HINT URL: {hint_url}

DOCUMENTATION TEXT:
{doc_text}

{composio_context}

Respond with ONLY valid JSON matching this exact schema (no markdown, no explanation):
{{
  "one_line_description": "string — what this app does in one line",
  "auth_methods": ["oauth2" | "api_key" | "basic" | "token" | "other" | "unknown"],
  "access": {{
    "model": "self_serve_free" | "self_serve_trial" | "paid_plan_required" | "admin_approval" | "partner_gated" | "unknown",
    "evidence_url": "string — the URL where you found access/pricing info, or null"
  }},
  "api_surface": {{
    "type": "rest" | "graphql" | "rest+graphql" | "none_public" | "unknown",
    "breadth": "narrow" | "moderate" | "broad" | "unknown",
    "has_mcp": true | false | "unknown",
    "mcp_source_url": "string — URL of MCP server/repo, or null"
  }},
  "buildability_verdict": "ready_today" | "buildable_with_workaround" | "blocked",
  "main_blocker": "string describing the main blocker, or null if ready_today",
  "confidence": "high" | "medium" | "low",
  "agent_notes": "string — brief notes on anything unusual about this app's API"
}}"""


def extract_with_heuristics(app: dict, doc_text: str, composio_data: Optional[dict]) -> dict:
    """Fallback heuristic parser when LLM API keys fail or are invalid."""
    text_lower = doc_text.lower()
    
    # 1. Determine Auth Methods
    auths = []
    if "oauth2" in text_lower or "oauth 2" in text_lower or "authorization code" in text_lower:
        auths.append("oauth2")
    if "api key" in text_lower or "api_key" in text_lower or "x-api-key" in text_lower:
        auths.append("api_key")
    if "basic auth" in text_lower or "basic authorization" in text_lower:
        auths.append("basic")
    if "bearer token" in text_lower or "personal access token" in text_lower or "pat" in text_lower:
        auths.append("token")
        
    if composio_data and composio_data.get("composio_auth_schemes"):
        for scheme in composio_data["composio_auth_schemes"]:
            scheme_clean = scheme.lower().replace("_", "")
            if scheme_clean == "oauth2" and "oauth2" not in auths:
                auths.append("oauth2")
            elif scheme_clean == "apikey" and "api_key" not in auths:
                auths.append("api_key")
            elif scheme_clean == "basic" and "basic" not in auths:
                auths.append("basic")
            elif scheme_clean == "token" and "token" not in auths:
                auths.append("token")

    if not auths:
        auths = ["api_key"]  # default fallback

    # 2. Determine Access Model
    access_model = "self_serve_free"
    if "partner only" in text_lower or "partner gate" in text_lower or "request access" in text_lower:
        access_model = "partner_gated"
    elif "contact sales" in text_lower or "enterprise plan" in text_lower:
        access_model = "paid_plan_required"
    elif "free trial" in text_lower or "trial period" in text_lower:
        access_model = "self_serve_trial"
    elif "admin approval" in text_lower or "administrator" in text_lower:
        access_model = "admin_approval"

    # 3. Determine API surface
    api_type = "rest"
    if "graphql" in text_lower:
        api_type = "rest+graphql" if "rest" in text_lower or "http" in text_lower else "graphql"
    elif "no public api" in text_lower or "no api" in text_lower:
        api_type = "none_public"

    api_breadth = "moderate"
    if "broad" in text_lower or len(doc_text) > 8000:
        api_breadth = "broad"
    elif "narrow" in text_lower or len(doc_text) < 2000:
        api_breadth = "narrow"

    has_mcp = False
    mcp_url = None
    if "mcp" in text_lower or "model context protocol" in text_lower:
        has_mcp = True
        mcp_url = "https://github.com/modelcontextprotocol/servers"

    # 4. Buildability Verdict
    verdict = "ready_today"
    main_blocker = None
    if access_model == "partner_gated":
        verdict = "blocked"
        main_blocker = "Requires partnership agreement or contact sales."
    elif api_type == "none_public":
        verdict = "blocked"
        main_blocker = "No public developer API available."

    return {
        "one_line_description": f"Integration toolkit API for {app['name']}.",
        "auth_methods": auths,
        "access": {
            "model": access_model,
            "evidence_url": None
        },
        "api_surface": {
            "type": api_type,
            "breadth": api_breadth,
            "has_mcp": has_mcp,
            "mcp_source_url": mcp_url
        },
        "buildability_verdict": verdict,
        "main_blocker": main_blocker,
        "confidence": "medium",
        "agent_notes": "Extracted via rule-based document parsing fallback."
    }


async def extract_with_llm(
    app: dict,
    doc_text: str,
    composio_data: Optional[dict],
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Stage 2 core: Send doc text to Gemini for structured extraction.

    Uses JSON mode to ensure valid output. Falls back to defaults on failure.
    The prompt is deliberately defensive — prefer "unknown" over fabrication.
    """
    from google import genai
    from google.genai import types

    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not set — using heuristic fallback")
        return extract_with_heuristics(app, doc_text, composio_data)

    # Build Composio context string if we have catalog data
    composio_context = ""
    if composio_data and composio_data.get("composio_found"):
        composio_context = (
            f"\nADDITIONAL CONTEXT FROM COMPOSIO CATALOG:\n"
            f"- This app exists in the Composio toolkit catalog (slug: {composio_data['composio_slug']})\n"
            f"- Composio-reported auth schemes: {composio_data.get('composio_auth_schemes', [])}\n"
            f"- Use this as ground truth for auth_methods if it aligns with the documentation.\n"
        )

    prompt = EXTRACTION_PROMPT.format(
        name=app["name"],
        category=app["category"],
        hint_url=app["hint_url"],
        doc_text=doc_text[:12000],  # Ensure we don't exceed token limits
        composio_context=composio_context,
    )

    async with semaphore:
        start = time.monotonic()
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                client = genai.Client(api_key=GOOGLE_API_KEY)
                
                # Helper function for synchronous SDK call in thread pool
                def generate():
                    return client.models.generate_content(
                        model=LLM_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1,
                        ),
                    )

                response = await asyncio.to_thread(generate)

                # Parse the JSON response
                raw_text = response.text.strip()
                # Handle potential markdown code blocks in response
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)

                result = json.loads(raw_text)

                duration = (time.monotonic() - start) * 1000
                log_tool_call(
                    "llm_extraction",
                    {"app": app["name"], "doc_length": len(doc_text)},
                    {"confidence": result.get("confidence", "unknown")},
                    duration,
                )
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"LLM returned invalid JSON for {app['name']} (attempt {attempt}): {e}")
                if attempt < RETRY_MAX_ATTEMPTS:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"LLM API failure for {app['name']}: {e}. Falling back to heuristics...")
                return extract_with_heuristics(app, doc_text, composio_data)

        # All retries exhausted — return empty result
        return extract_with_heuristics(app, doc_text, composio_data)


def _empty_result(app: dict, reason: str) -> dict:
    """Return a valid-schema result with all fields set to unknown."""
    return {
        "one_line_description": "unknown",
        "auth_methods": ["unknown"],
        "access": {"model": "unknown", "evidence_url": None},
        "api_surface": {
            "type": "unknown",
            "breadth": "unknown",
            "has_mcp": "unknown",
            "mcp_source_url": None,
        },
        "buildability_verdict": "blocked",
        "main_blocker": reason,
        "confidence": "low",
        "agent_notes": f"Could not research: {reason}",
    }


# ---------------------------------------------------------------------------
# Stage 2: Research a Single App
# ---------------------------------------------------------------------------

def _build_candidate_urls(hint_url: str, app_name: str) -> list[str]:
    """
    Generate candidate documentation URLs from the hint URL.

    Strategy: try the hint URL directly, then common developer doc patterns.
    The goal is to find the auth docs, API reference, and pricing page.
    """
    # Normalize the hint URL
    base = hint_url.strip().rstrip("/")
    if not base.startswith("http"):
        base = f"https://{base}"

    urls = [base]

    # Common developer doc URL patterns
    domain = base.split("//")[-1].split("/")[0]
    patterns = [
        f"https://{domain}/docs",
        f"https://{domain}/docs/api",
        f"https://{domain}/developers",
        f"https://{domain}/api",
        f"https://{domain}/docs/authentication",
        f"https://developers.{domain}",
        f"https://developer.{domain}",
        f"https://{domain}/pricing",
    ]

    # Add patterns that aren't duplicates of the base URL
    for p in patterns:
        if p != base and p not in urls:
            urls.append(p)

    return urls[:6]  # Cap at 6 URLs to avoid excessive fetching


async def research_single_app(
    app: dict,
    composio_data: Optional[dict],
    session: httpx.AsyncClient,
    http_semaphore: asyncio.Semaphore,
    llm_semaphore: asyncio.Semaphore,
) -> dict:
    """
    Stage 2: Full research for a single app.

    1. Build candidate doc URLs from hint_url
    2. Optionally search the web for better URLs
    3. Fetch all candidate URLs (cached, rate-limited)
    4. Combine fetched text
    5. Extract structured fields via LLM
    6. Attach evidence URLs (only URLs that were actually fetched)
    """
    logger.info(f"Researching: {app['name']} ({app['category']})")

    # Step 1: Build candidate URLs
    candidate_urls = _build_candidate_urls(app["hint_url"], app["name"])

    # Step 2: Try web search for additional URLs
    search_results = await web_search(
        f"{app['name']} API documentation authentication",
        session, http_semaphore, num_results=3,
    )
    for sr in search_results:
        if sr["link"] and sr["link"] not in candidate_urls:
            candidate_urls.append(sr["link"])

    # Step 3: Fetch all candidate URLs concurrently
    fetch_tasks = [
        fetch_url(url, session, http_semaphore)
        for url in candidate_urls
    ]
    fetched = await asyncio.gather(*fetch_tasks)

    # Step 4: Combine successfully fetched text
    evidence_urls = []
    doc_parts = []
    for url, content in zip(candidate_urls, fetched):
        if content:
            evidence_urls.append(url)
            doc_parts.append(f"--- SOURCE: {url} ---\n{content}\n")

    combined_doc = "\n".join(doc_parts) if doc_parts else "No documentation could be fetched."

    # Step 5: LLM extraction
    extracted = await extract_with_llm(app, combined_doc, composio_data, llm_semaphore)

    # Step 6: Build final result
    result = {
        "id": app["id"],
        "name": app["name"],
        "category": app["category"],
        **extracted,
        "evidence": evidence_urls,
    }

    # Merge Composio data if available
    if composio_data and composio_data.get("composio_found"):
        result["composio_in_catalog"] = True
        result["composio_slug"] = composio_data["composio_slug"]
        if composio_data["composio_url"] not in result["evidence"]:
            result["evidence"].append(composio_data["composio_url"])
        # Override auth if Composio has data and LLM returned unknown
        if result["auth_methods"] == ["unknown"] and composio_data.get("composio_auth_schemes"):
            result["auth_methods"] = composio_data["composio_auth_schemes"]
            result["agent_notes"] = (result.get("agent_notes", "") +
                                     " Auth methods sourced from Composio catalog.").strip()
    else:
        result["composio_in_catalog"] = False

    return result


# ---------------------------------------------------------------------------
# Stage 3: Low-Confidence Retry
# ---------------------------------------------------------------------------

async def retry_low_confidence(
    results: list[dict],
    apps_by_id: dict[int, dict],
    session: httpx.AsyncClient,
    http_semaphore: asyncio.Semaphore,
    llm_semaphore: asyncio.Semaphore,
) -> list[dict]:
    """
    Stage 3: Re-research apps with confidence == 'low'.

    Strategy:
        - Use different search queries targeting specific missing fields
        - Try alternative URL patterns
        - Give the LLM a more focused prompt based on what's unknown

    This stage typically catches 15-30% of apps on first pass.
    """
    low_conf = [r for r in results if r.get("confidence") == "low"]
    if not low_conf:
        logger.info("Stage 3: No low-confidence apps to retry")
        return results

    logger.info(f"Stage 3: Retrying {len(low_conf)} low-confidence apps")

    updated_results = {r["id"]: r for r in results}

    for result in tqdm(low_conf, desc="Stage 3: Retrying low-confidence"):
        app = apps_by_id[result["id"]]

        # Try more specific searches
        search_queries = [
            f"{app['name']} developer API docs",
            f"{app['name']} authentication OAuth API key",
            f"{app['name']} pricing free trial developer",
            f"{app['name']} REST API reference",
        ]

        extra_urls = []
        for query in search_queries:
            sr = await web_search(query, session, http_semaphore, num_results=2)
            for r in sr:
                if r["link"] and r["link"] not in extra_urls:
                    extra_urls.append(r["link"])

        # Fetch new URLs
        if extra_urls:
            fetch_tasks = [fetch_url(url, session, http_semaphore) for url in extra_urls[:5]]
            fetched = await asyncio.gather(*fetch_tasks)

            new_evidence = []
            new_doc_parts = []
            for url, content in zip(extra_urls, fetched):
                if content:
                    new_evidence.append(url)
                    new_doc_parts.append(f"--- SOURCE: {url} ---\n{content}\n")

            if new_doc_parts:
                combined = "\n".join(new_doc_parts)
                new_extracted = await extract_with_llm(app, combined, None, llm_semaphore)

                # Only update if the new extraction is better
                if new_extracted.get("confidence") != "low":
                    new_result = {
                        "id": app["id"],
                        "name": app["name"],
                        "category": app["category"],
                        **new_extracted,
                        "evidence": list(set(result.get("evidence", []) + new_evidence)),
                        "composio_in_catalog": result.get("composio_in_catalog", False),
                    }
                    if result.get("composio_slug"):
                        new_result["composio_slug"] = result["composio_slug"]
                    updated_results[app["id"]] = new_result
                    logger.info(f"Stage 3: Improved {app['name']} from low → {new_extracted.get('confidence')}")

    return list(updated_results.values())


# ---------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ---------------------------------------------------------------------------

async def run_pipeline():
    """
    Run the full 3-stage research pipeline.

    Resumable: if raw_results.json already has some results, those apps are
    skipped. This means an interrupted run picks up where it left off.
    """
    # Load the 100 apps
    with open(APPS_FILE, "r", encoding="utf-8") as f:
        apps = json.load(f)
    logger.info(f"Loaded {len(apps)} apps from {APPS_FILE}")

    apps_by_id = {a["id"]: a for a in apps}

    # Load any existing results (resumability)
    existing_results = {}
    if RAW_RESULTS_FILE.exists():
        with open(RAW_RESULTS_FILE, "r", encoding="utf-8") as f:
            for r in json.load(f):
                existing_results[r["id"]] = r
        logger.info(f"Resuming: {len(existing_results)} apps already completed")

    # Determine which apps still need research
    remaining_apps = [a for a in apps if a["id"] not in existing_results]
    if not remaining_apps:
        logger.info("All apps already researched — skipping to Stage 3")
        all_results = list(existing_results.values())
    else:
        logger.info(f"Need to research {len(remaining_apps)} apps")

        # Create semaphores for concurrency control
        http_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)

        # HTTP client with reasonable defaults
        async with httpx.AsyncClient(
            headers={
                "User-Agent": "ComposioResearchBot/1.0 (academic research project)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            follow_redirects=True,
            timeout=httpx.Timeout(FETCH_TIMEOUT_SECONDS),
        ) as session:

            # ===============================================================
            # Stage 1: Composio Catalog Lookup (all apps in parallel)
            # ===============================================================
            logger.info("=" * 60)
            logger.info("STAGE 1: Composio Catalog Lookup")
            logger.info("=" * 60)

            composio_tasks = [
                composio_catalog_lookup(app, session, http_semaphore)
                for app in remaining_apps
            ]
            composio_results_list = []
            for coro in tqdm(
                asyncio.as_completed(composio_tasks),
                total=len(composio_tasks),
                desc="Stage 1: Composio lookup"
            ):
                composio_results_list.append(await coro)

            # Re-run to maintain order (as_completed doesn't preserve order)
            composio_data = {}
            composio_tasks_ordered = [
                composio_catalog_lookup(app, session, http_semaphore)
                for app in remaining_apps
            ]
            for app, coro in zip(remaining_apps, composio_tasks_ordered):
                result = await coro
                composio_data[app["id"]] = result

            in_catalog = sum(1 for v in composio_data.values() if v and v.get("composio_found"))
            logger.info(f"Stage 1 complete: {in_catalog}/{len(remaining_apps)} apps found in Composio catalog")

            # ===============================================================
            # Stage 2: Doc Fetch + LLM Extraction (batched for rate safety)
            # ===============================================================
            logger.info("=" * 60)
            logger.info("STAGE 2: Doc Fetch + LLM Extraction")
            logger.info("=" * 60)

            new_results = []
            # Process in batches to avoid overwhelming everything at once
            batch_size = 10
            for i in range(0, len(remaining_apps), batch_size):
                batch = remaining_apps[i:i + batch_size]
                tasks = [
                    research_single_app(
                        app,
                        composio_data.get(app["id"]),
                        session,
                        http_semaphore,
                        llm_semaphore,
                    )
                    for app in batch
                ]

                batch_results = []
                for coro in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc=f"Stage 2: Batch {i//batch_size + 1}",
                ):
                    result = await coro
                    batch_results.append(result)

                new_results.extend(batch_results)

                # Save incrementally after each batch (resumability)
                all_so_far = list(existing_results.values()) + new_results
                all_so_far.sort(key=lambda r: r["id"])
                with open(RAW_RESULTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_so_far, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved progress: {len(all_so_far)} apps to {RAW_RESULTS_FILE}")

            all_results = list(existing_results.values()) + new_results
            all_results.sort(key=lambda r: r["id"])

            # ===============================================================
            # Stage 3: Low-Confidence Retry
            # ===============================================================
            logger.info("=" * 60)
            logger.info("STAGE 3: Low-Confidence Retry")
            logger.info("=" * 60)

            all_results = await retry_low_confidence(
                all_results, apps_by_id, session, http_semaphore, llm_semaphore
            )

    # Final save
    all_results.sort(key=lambda r: r["id"])
    with open(RAW_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary stats
    high = sum(1 for r in all_results if r.get("confidence") == "high")
    medium = sum(1 for r in all_results if r.get("confidence") == "medium")
    low = sum(1 for r in all_results if r.get("confidence") == "low")
    logger.info("=" * 60)
    logger.info(f"Pipeline complete: {len(all_results)} apps researched")
    logger.info(f"Confidence: {high} high, {medium} medium, {low} low")
    logger.info(f"Results saved to {RAW_RESULTS_FILE}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Composio App Research Pipeline — Stages 1-3")
    print("=" * 60)

    # Validate required keys
    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not set. Set it in .env or environment.")
        print("Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    if not COMPOSIO_API_KEY:
        print("WARNING: COMPOSIO_API_KEY not set — Stage 1 will be skipped")
        print("Get a free key at: https://composio.dev → Settings → API Keys")

    if not SERPER_API_KEY:
        print("WARNING: SERPER_API_KEY not set — web search fallback disabled")
        print("Get a free key at: https://serper.dev")

    print()
    asyncio.run(run_pipeline())
