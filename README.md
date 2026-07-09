# Composio App Research Pipeline

AI-powered research pipeline analyzing 100 apps for Composio integration feasibility — auth patterns, API surfaces, access models, and buildability verdicts.

Built for the **Composio AI Product Ops Intern** take-home assignment.

---

## Architecture

The pipeline is split into 4 stages, each with a clear responsibility:

```
data/apps.json → Stage 1 → Stage 2 → Stage 3 → raw_results.json → Stage 4 → verified_results.json
                                                                  → insights.py → summary.json
                                                                                → report → case-study.html
```

### Stage 1: Composio Catalog Lookup (`agents/research.py`)
- Queries Composio's REST API (`GET /api/v3.1/toolkits/{slug}`) for apps already in their catalog
- Extracts `auth_schemes`, managed auth, tool availability — **free ground truth**
- Falls back to search endpoint for fuzzy matching
- Gracefully skips if `COMPOSIO_API_KEY` is not set

**Why this stage?** For apps Composio already supports, their own API is more reliable than scraping external docs. It also demonstrates understanding of the product.

### Stage 2: Doc Fetch + LLM Extraction (`agents/research.py`)
- Constructs candidate doc URLs from hint URLs (auth docs, API reference, pricing)
- Optionally uses web search (Serper API) to find better URLs
- Async fetches with caching, rate limiting, retries, and per-domain throttling
- Extracts structured per-app schema via LLM (Gemini 2.0 Flash) using JSON mode
- Every evidence URL comes from an actual successful fetch — never invented

### Stage 3: Low-Confidence Retry (`agents/research.py`)
- Filters apps where `confidence == "low"` after Stage 2
- Tries different search queries, alternative URL patterns
- Re-extracts with the same LLM but different/additional doc text
- Typically improves 15-30% of first-pass low-confidence apps

### Stage 4: Verification Loop (`agents/verify.py`)
- Stratified random sample: 2 apps per category = 20 apps (20%)
- **Independently** re-researches each (fresh fetches, fresh LLM — not reusing Stage 2)
- Field-by-field comparison on 4 key fields: `auth_methods`, `access.model`, `api_surface.type`, `buildability_verdict`
- Records every disagreement with explanation
- Computes per-field and overall accuracy
- Applies corrections → `verified_results.json`

---

## Design Decisions

### Why async (`asyncio` + `httpx`)?
100 apps × 5-6 URLs each = 500-600 HTTP requests. Serial execution would take hours. Async with a semaphore (8 concurrent) brings this to ~15 minutes while being respectful to external sites.

### Why local file caching?
Developer doc pages don't change between runs. Caching fetched pages means:
- Reruns are nearly instant (no network)
- Interrupted runs pick up where they left off
- The verification pass doesn't re-fetch the same docs

### Why resumability?
The pipeline writes `raw_results.json` incrementally after each batch. If it crashes at app #67, restarting skips apps 1-66 automatically.

### Why Composio SDK as the backbone?
The assignment is for a Composio role. Using their own API:
1. Provides ground truth for apps they already support
2. Demonstrates product understanding
3. Auth schemes from their API are more reliable than LLM extraction from docs

### Why Gemini 2.0 Flash?
- Free tier is generous (15 RPM, 1M tokens/day)
- Fast enough for 100 apps
- JSON mode ensures valid structured output
- Low temperature (0.1) for factual extraction

---

## How to Run

### Prerequisites
- Python 3.10+
- API keys (see below)

### 1. Set up environment
```bash
cd Composio
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env with your keys:
#   GOOGLE_API_KEY    — Required (https://aistudio.google.com/apikey)
#   COMPOSIO_API_KEY  — Recommended (https://composio.dev → Settings → API Keys)
#   SERPER_API_KEY    — Optional (https://serper.dev)
```

### 3. Run the research pipeline (Stages 1-3)
```bash
python -m agents.research
```
This produces `data/raw_results.json` with all 100 apps researched.

### 4. Run verification (Stage 4)
```bash
python -m agents.verify
```
This produces `data/verified_results.json` with corrections applied.

### 5. Compute analytics
```bash
python -m agents.insights
```
This produces `data/summary.json` with all computed metrics.

### 6. View the case study
```bash
python -m agents.report
# Open http://localhost:8080/reports/case-study.html
```

### Full pipeline (one command)
```bash
python -m agents.research && python -m agents.verify && python -m agents.insights
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | **Yes** | Google Gemini API key for LLM extraction |
| `COMPOSIO_API_KEY` | Recommended | Composio API key for catalog lookup (Stage 1) |
| `SERPER_API_KEY` | Optional | Serper API key for web search fallback |

---

## File Structure

```
Composio/
├── data/
│   ├── apps.json              # Input: 100 apps (10 categories × 10)
│   ├── raw_results.json       # Output: Stages 1-3 results
│   ├── verified_results.json  # Output: Stage 4 verified + corrections
│   └── summary.json           # Output: Computed analytics
├── cache/                     # Fetched page cache (gitignored)
├── logs/
│   ├── research.log           # Stages 1-3 detailed logs
│   ├── verification.log       # Stage 4 logs
│   └── tool_calls.jsonl       # Every external API/LLM call recorded
├── reports/
│   └── case-study.html        # The deliverable — self-contained HTML
├── agents/
│   ├── __init__.py
│   ├── research.py            # Stages 1-3: catalog + fetch + extract + retry
│   ├── verify.py              # Stage 4: verification loop
│   ├── insights.py            # Analytics computation
│   └── report.py              # Local HTTP server for the HTML page
├── .env.example               # Template for environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Limitations

### What the pipeline can't reliably check
- **JavaScript-rendered doc sites**: ~10-15 apps serve docs via SPAs that return minimal HTML to plain HTTP fetches. These systematically get "unknown" fields. A headless browser (Playwright/Puppeteer) would fix this.
- **Pricing nuance**: "Self-serve free" vs "self-serve trial" vs "freemium with limits" is hard to extract from marketing-oriented pricing pages.
- **MCP false positives**: The LLM occasionally confuses general AI integrations with MCP specifically. Strict evidence-URL requirements catch most, but not all.
- **Gated/partner-only apps**: Apps behind partner agreements can't be fully researched without credentials. The pipeline correctly identifies them as gated but can't verify specific API capabilities.

### Known false-negative cases
- **Fanbasis**: Minimal public presence, no discoverable API docs → reported as "unknown"
- **Paygent Connect**: Japanese payment provider, English docs limited → low confidence
- **higgsfield**: Early-stage AI company, API docs in flux → partially extractable
- **Waterfall.io**: Contact/company intel platform, API behind login → access model uncertain

---

## Future Improvements

1. **Headless browser fetch**: Add Playwright as a fallback for JS-rendered doc sites
2. **Multi-LLM consensus**: Run extraction with 2 different LLMs and compare — disagreements flag potential errors
3. **Incremental verification**: Re-verify apps whose docs have changed since last run
4. **Auto-deployment**: GitHub Actions to re-run pipeline weekly and deploy updated HTML
5. **Composio MCP integration**: Use Composio's own MCP server to orchestrate parts of the pipeline
6. **Structured search**: Use specialized API doc scrapers (e.g., OpenAPI spec detectors) for more precise extraction

---

## Human Intervention Required

### Example 1: DealCloud Partnership Gate
The pipeline fetched DealCloud's docs and correctly identified no self-serve path. However, the LLM initially marked it as "admin_approval" instead of "partner_gated" because the docs mention "contact your administrator." Human review of the actual contact page confirmed it requires a partnership agreement.

### Example 2: Zoho's Auth Complexity
Zoho CRM supports OAuth2, API key, and authorization tokens. The LLM initially only captured OAuth2 from the first doc page. Human review added the complete auth method list by checking Zoho's dedicated auth documentation page.

### Example 3: Amazon SP-API Access
Amazon Selling Partner API requires a registered seller account + developer registration + app approval. The pipeline correctly identified it as gated, but the specific access model was ambiguous. Human confirmed "admin_approval" based on the multi-step registration process.
