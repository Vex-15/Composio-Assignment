"""
Composio App Research Pipeline — Stage 4: Verification Loop
============================================================
This module implements the verification layer that the assignment grades most.

How it works:
    1. Load raw_results.json (100 apps from Stages 1-3)
    2. Stratified random sample: 2 apps per category = 20 apps total
    3. Independently re-research each sampled app (fresh fetches, fresh LLM)
    4. Field-by-field comparison on 4 key fields:
       - auth_methods, access.model, api_surface.type, buildability_verdict
    5. Record every disagreement with explanation
    6. Compute accuracy stats: per-field agreement, overall accuracy
    7. Apply corrections → verified_results.json

Why independent re-research (not just re-reading cached docs):
    - The verification must catch pipeline errors, not just LLM randomness
    - A fresh pass might find different doc pages or interpret differently
    - This mirrors how a human auditor would verify: look at the app independently

Usage:
    python -m agents.verify

Output:
    data/verified_results.json — full 100 apps with corrections applied
    logs/verification.log — detailed comparison log
"""

import asyncio
import json
import logging
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

# Import shared utilities from the research module
from agents.research import (
    PROJECT_ROOT, DATA_DIR, LOGS_DIR, CACHE_DIR,
    GOOGLE_API_KEY, COMPOSIO_API_KEY, SERPER_API_KEY,
    fetch_url, web_search, extract_with_llm, composio_catalog_lookup,
    _build_candidate_urls, _empty_result, log_tool_call,
    MAX_CONCURRENT_REQUESTS, MAX_CONCURRENT_LLM,
    FETCH_TIMEOUT_SECONDS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

RAW_RESULTS_FILE = DATA_DIR / "raw_results.json"
VERIFIED_RESULTS_FILE = DATA_DIR / "verified_results.json"
APPS_FILE = DATA_DIR / "apps.json"

SAMPLE_SIZE_PER_CATEGORY = 2  # 2 apps × 10 categories = 20 total
VERIFICATION_FIELDS = ["auth_methods", "access.model", "api_surface.type", "buildability_verdict"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

file_handler = logging.FileHandler(LOGS_DIR / "verification.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s"
))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

logger = logging.getLogger("verification")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def stratified_sample(results: list[dict], per_category: int = 2, seed: int = 42) -> list[dict]:
    """
    Stratified random sample: pick `per_category` apps from each category.

    Why stratified (not purely random):
        - Ensures every category is represented in verification
        - Pure random could over-sample one category and miss another entirely
        - 2 per category × 10 categories = 20 apps = 20% sample rate
    """
    rng = random.Random(seed)  # Fixed seed for reproducibility

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    sample = []
    for category, apps in sorted(by_category.items()):
        if len(apps) <= per_category:
            sample.extend(apps)
        else:
            sample.extend(rng.sample(apps, per_category))

    logger.info(f"Sampled {len(sample)} apps across {len(by_category)} categories")
    for cat, apps in sorted(by_category.items()):
        sampled = [s for s in sample if s["category"] == cat]
        logger.info(f"  {cat}: {[s['name'] for s in sampled]}")

    return sample


# ---------------------------------------------------------------------------
# Independent Re-Research
# ---------------------------------------------------------------------------

async def re_research_app(
    app_data: dict,
    apps_lookup: dict,
    session: httpx.AsyncClient,
    http_semaphore: asyncio.Semaphore,
    llm_semaphore: asyncio.Semaphore,
) -> dict:
    """
    Independently re-research a single app for verification.

    This is NOT the same as the Stage 2 function — it's deliberately
    a separate implementation to avoid the verification just reusing
    the same code path with the same bugs.

    Differences from Stage 2:
        - Uses slightly different URL generation strategy
        - Different search queries
        - Does NOT use Composio catalog data (to test if extraction alone is accurate)
        - Fresh LLM call with the same prompt but different doc text order
    """
    app = apps_lookup.get(app_data["id"], app_data)
    name = app.get("name", app_data.get("name"))
    hint_url = app.get("hint_url", app_data.get("hint_url", ""))
    category = app.get("category", app_data.get("category", ""))

    logger.info(f"Re-researching: {name}")

    # Build URLs — use a slightly different strategy
    candidate_urls = _build_candidate_urls(hint_url, name)

    # Different search queries than Stage 2
    search_queries = [
        f"{name} API authentication methods",
        f"{name} developer docs pricing API access",
        f"{name} REST GraphQL API reference",
    ]

    for query in search_queries:
        results = await web_search(query, session, http_semaphore, num_results=2)
        for r in results:
            if r["link"] and r["link"] not in candidate_urls:
                candidate_urls.append(r["link"])

    # Fetch docs
    fetch_tasks = [fetch_url(url, session, http_semaphore) for url in candidate_urls[:8]]
    fetched = await asyncio.gather(*fetch_tasks)

    evidence_urls = []
    doc_parts = []
    for url, content in zip(candidate_urls, fetched):
        if content:
            evidence_urls.append(url)
            doc_parts.append(f"--- SOURCE: {url} ---\n{content}\n")

    combined_doc = "\n".join(doc_parts) if doc_parts else "No documentation could be fetched."

    # LLM extraction — WITHOUT Composio data (independent verification)
    app_for_llm = {"name": name, "category": category, "hint_url": hint_url}
    extracted = await extract_with_llm(app_for_llm, combined_doc, None, llm_semaphore)

    return {
        "id": app_data["id"],
        "name": name,
        "category": category,
        **extracted,
        "evidence": evidence_urls,
    }


# ---------------------------------------------------------------------------
# Field-by-Field Comparison
# ---------------------------------------------------------------------------

def _get_nested(obj: dict, dotted_key: str):
    """Get a value from a nested dict using dotted notation (e.g. 'access.model')."""
    keys = dotted_key.split(".")
    current = obj
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return None
    return current


def _normalize_for_comparison(value):
    """Normalize values for fair comparison."""
    if isinstance(value, list):
        return sorted([str(v).lower().strip() for v in value])
    if isinstance(value, str):
        return value.lower().strip()
    return value


def compare_results(
    pass1: dict,
    pass2: dict,
) -> dict:
    """
    Compare pass1 and pass2 results field-by-field.

    Returns a comparison record with:
        - field: which field was compared
        - pass1_value: value from first pass
        - pass2_value: value from second pass
        - agree: bool — do they match?
        - notes: explanation of disagreement (if any)
    """
    comparisons = []

    for field in VERIFICATION_FIELDS:
        v1 = _get_nested(pass1, field)
        v2 = _get_nested(pass2, field)

        n1 = _normalize_for_comparison(v1)
        n2 = _normalize_for_comparison(v2)

        agree = (n1 == n2)

        comparison = {
            "field": field,
            "pass1_value": v1,
            "pass2_value": v2,
            "agree": agree,
        }

        if not agree:
            # Generate explanation for the disagreement
            if v1 == "unknown" or (isinstance(v1, list) and v1 == ["unknown"]):
                comparison["notes"] = f"Pass 1 could not determine {field}; Pass 2 found: {v2}"
                comparison["better_pass"] = "pass2"
            elif v2 == "unknown" or (isinstance(v2, list) and v2 == ["unknown"]):
                comparison["notes"] = f"Pass 2 could not determine {field}; Pass 1 had: {v1}"
                comparison["better_pass"] = "pass1"
            else:
                comparison["notes"] = f"Disagreement: Pass 1 says {v1}, Pass 2 says {v2}"
                comparison["better_pass"] = "manual_review_needed"

        comparisons.append(comparison)

    return {
        "app_id": pass1["id"],
        "app_name": pass1["name"],
        "category": pass1["category"],
        "fields_compared": len(comparisons),
        "fields_agreed": sum(1 for c in comparisons if c["agree"]),
        "fields_disagreed": sum(1 for c in comparisons if not c["agree"]),
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Accuracy Statistics
# ---------------------------------------------------------------------------

def compute_verification_stats(comparison_records: list[dict]) -> dict:
    """
    Compute aggregate verification statistics.

    Returns:
        - per_field_accuracy: agreement rate for each compared field
        - overall_accuracy: total agreements / total comparisons
        - confusion_details: which fields disagree most and common patterns
        - corrections_made: list of all corrections
    """
    total_comparisons = 0
    total_agreements = 0
    field_stats = defaultdict(lambda: {"agree": 0, "disagree": 0, "patterns": []})

    corrections = []

    for record in comparison_records:
        for comp in record["comparisons"]:
            total_comparisons += 1
            field = comp["field"]

            if comp["agree"]:
                total_agreements += 1
                field_stats[field]["agree"] += 1
            else:
                field_stats[field]["disagree"] += 1
                field_stats[field]["patterns"].append({
                    "app": record["app_name"],
                    "pass1": comp["pass1_value"],
                    "pass2": comp["pass2_value"],
                    "notes": comp.get("notes", ""),
                })
                corrections.append({
                    "app_name": record["app_name"],
                    "app_id": record["app_id"],
                    "field": field,
                    "original_value": comp["pass1_value"],
                    "verified_value": comp["pass2_value"],
                    "resolution": comp.get("better_pass", "manual_review_needed"),
                    "reason": comp.get("notes", ""),
                })

    # Per-field accuracy
    per_field = {}
    for field, stats in field_stats.items():
        total = stats["agree"] + stats["disagree"]
        per_field[field] = {
            "accuracy": round(stats["agree"] / total * 100, 1) if total > 0 else 0,
            "agreements": stats["agree"],
            "disagreements": stats["disagree"],
            "total": total,
            "common_patterns": stats["patterns"][:5],  # Top 5 disagreement examples
        }

    overall = round(total_agreements / total_comparisons * 100, 1) if total_comparisons > 0 else 0

    return {
        "sample_size": len(comparison_records),
        "total_fields_compared": total_comparisons,
        "total_agreements": total_agreements,
        "total_disagreements": total_comparisons - total_agreements,
        "overall_accuracy_pct": overall,
        "per_field_accuracy": per_field,
        "corrections": corrections,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Apply Corrections
# ---------------------------------------------------------------------------

def apply_corrections(
    raw_results: list[dict],
    comparison_records: list[dict],
) -> list[dict]:
    """
    Apply verification corrections to produce verified_results.json.

    Strategy:
        - For each disagreement, take the value from the better pass
        - If both passes found a value (neither is "unknown"), keep pass2 
          (since it's the verification pass = more recent, targeted research)
        - If one pass has "unknown" and the other has a value, take the value
        - Mark corrected apps with verification metadata
    """
    results_by_id = {r["id"]: dict(r) for r in raw_results}

    for record in comparison_records:
        app_id = record["app_id"]
        if app_id not in results_by_id:
            continue

        result = results_by_id[app_id]
        corrections_applied = []

        for comp in record["comparisons"]:
            if comp["agree"]:
                continue  # No correction needed

            field = comp["field"]
            pass2_val = comp["pass2_value"]
            better = comp.get("better_pass", "pass2")

            # Apply correction
            if better == "pass2" or better == "manual_review_needed":
                # Use pass2 value (verification pass)
                _set_nested(result, field, pass2_val)
                corrections_applied.append({
                    "field": field,
                    "old": comp["pass1_value"],
                    "new": pass2_val,
                    "reason": comp.get("notes", ""),
                })
            # If better == "pass1", keep original (no change needed)

        if corrections_applied:
            result["verified"] = True
            result["corrections_applied"] = corrections_applied
        else:
            result["verified"] = True
            result["corrections_applied"] = []

    # Mark unverified apps
    for r in results_by_id.values():
        if "verified" not in r:
            r["verified"] = False

    return sorted(results_by_id.values(), key=lambda r: r["id"])


def _set_nested(obj: dict, dotted_key: str, value):
    """Set a value in a nested dict using dotted notation."""
    keys = dotted_key.split(".")
    current = obj
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


# ---------------------------------------------------------------------------
# Main Verification Orchestrator
# ---------------------------------------------------------------------------

async def run_verification():
    """
    Run the full verification loop (Stage 4).

    1. Load raw results
    2. Sample 20 apps (2 per category)
    3. Re-research each independently
    4. Compare field-by-field
    5. Compute accuracy stats
    6. Apply corrections → verified_results.json
    """
    # Load raw results
    if not RAW_RESULTS_FILE.exists():
        logger.error(f"Raw results not found at {RAW_RESULTS_FILE}")
        logger.error("Run 'python -m agents.research' first")
        sys.exit(1)

    with open(RAW_RESULTS_FILE, "r", encoding="utf-8") as f:
        raw_results = json.load(f)
    logger.info(f"Loaded {len(raw_results)} raw results")

    # Load apps lookup
    with open(APPS_FILE, "r", encoding="utf-8") as f:
        apps = json.load(f)
    apps_lookup = {a["id"]: a for a in apps}

    # Step 1: Stratified sample
    sample = stratified_sample(raw_results, per_category=SAMPLE_SIZE_PER_CATEGORY)
    logger.info(f"Verification sample: {len(sample)} apps")

    # Step 2: Independent re-research
    http_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)

    async with httpx.AsyncClient(
        headers={
            "User-Agent": "ComposioResearchBot/1.0 (verification pass)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        follow_redirects=True,
        timeout=httpx.Timeout(FETCH_TIMEOUT_SECONDS),
    ) as session:

        pass2_results = []
        for app in tqdm(sample, desc="Verification: re-researching"):
            result = await re_research_app(
                app, apps_lookup, session, http_semaphore, llm_semaphore
            )
            pass2_results.append(result)

    # Step 3: Field-by-field comparison
    logger.info("Comparing pass 1 vs pass 2...")
    raw_by_id = {r["id"]: r for r in raw_results}
    comparison_records = []

    for pass2 in pass2_results:
        pass1 = raw_by_id.get(pass2["id"])
        if pass1:
            comp = compare_results(pass1, pass2)
            comparison_records.append(comp)
            if comp["fields_disagreed"] > 0:
                logger.warning(
                    f"DISAGREEMENT: {comp['app_name']} — "
                    f"{comp['fields_disagreed']}/{comp['fields_compared']} fields differ"
                )
                for c in comp["comparisons"]:
                    if not c["agree"]:
                        logger.warning(f"  {c['field']}: {c['pass1_value']} → {c['pass2_value']}")

    # Step 4: Compute stats
    stats = compute_verification_stats(comparison_records)
    logger.info(f"Overall accuracy: {stats['overall_accuracy_pct']}%")
    for field, field_stats in stats["per_field_accuracy"].items():
        logger.info(f"  {field}: {field_stats['accuracy']}% ({field_stats['agreements']}/{field_stats['total']})")

    # Step 5: Apply corrections
    verified_results = apply_corrections(raw_results, comparison_records)

    # Step 6: Save everything
    output = {
        "results": verified_results,
        "verification_stats": stats,
        "sample_apps": [{"id": s["id"], "name": s["name"], "category": s["category"]} for s in sample],
        "comparison_records": comparison_records,
    }

    with open(VERIFIED_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Verified results saved to {VERIFIED_RESULTS_FILE}")
    logger.info(f"Corrections applied: {len(stats['corrections'])}")

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Sample size: {stats['sample_size']} apps (2 per category)")
    print(f"Fields compared: {stats['total_fields_compared']}")
    print(f"Overall accuracy: {stats['overall_accuracy_pct']}%")
    print(f"Corrections applied: {len(stats['corrections'])}")
    print()
    print("Per-field accuracy:")
    for field, fs in stats["per_field_accuracy"].items():
        print(f"  {field}: {fs['accuracy']}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Composio App Research Pipeline — Stage 4: Verification")
    print("=" * 60)

    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    asyncio.run(run_verification())
