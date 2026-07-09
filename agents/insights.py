"""
Composio App Research Pipeline — Analytics & Insights
======================================================
Pure computation module — no LLM calls, no network requests.

Reads raw_results.json and verified_results.json, computes all analytics
required for the case study, and writes summary.json.

All metrics are auto-computed from actual data. Nothing is hand-written.

Usage:
    python -m agents.insights
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_RESULTS_FILE = DATA_DIR / "raw_results.json"
VERIFIED_RESULTS_FILE = DATA_DIR / "verified_results.json"
SUMMARY_FILE = DATA_DIR / "summary.json"


def load_results() -> tuple[list[dict], dict]:
    """Load raw and verified results. Returns (results_list, verification_stats)."""
    if VERIFIED_RESULTS_FILE.exists():
        with open(VERIFIED_RESULTS_FILE, "r", encoding="utf-8") as f:
            verified_data = json.load(f)
        results = verified_data.get("results", [])
        verification_stats = verified_data.get("verification_stats", {})
        comparison_records = verified_data.get("comparison_records", [])
    elif RAW_RESULTS_FILE.exists():
        with open(RAW_RESULTS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        verification_stats = {}
        comparison_records = []
    else:
        print(f"ERROR: No results found. Run the research pipeline first.")
        sys.exit(1)

    return results, verification_stats, comparison_records


def compute_auth_distribution(results: list[dict]) -> dict:
    """
    Auth method distribution across all 100 apps.

    Note: One app can have multiple auth methods (e.g., OAuth2 + API key),
    so totals may exceed 100.
    """
    counter = Counter()
    for r in results:
        methods = r.get("auth_methods", ["unknown"])
        if isinstance(methods, list):
            for m in methods:
                counter[m.lower()] += 1
        else:
            counter[str(methods).lower()] += 1

    total = len(results)
    return {
        "counts": dict(counter.most_common()),
        "percentages": {k: round(v / total * 100, 1) for k, v in counter.most_common()},
        "total_apps": total,
        "note": "Percentages may exceed 100% because apps can support multiple auth methods",
    }


def compute_api_type_distribution(results: list[dict]) -> dict:
    """API type distribution: REST vs GraphQL vs both vs none."""
    counter = Counter()
    for r in results:
        api = r.get("api_surface", {})
        api_type = api.get("type", "unknown") if isinstance(api, dict) else "unknown"
        counter[api_type.lower()] += 1

    total = len(results)
    return {
        "counts": dict(counter.most_common()),
        "percentages": {k: round(v / total * 100, 1) for k, v in counter.most_common()},
    }


def compute_access_model_distribution(results: list[dict]) -> dict:
    """Self-serve vs gated split."""
    counter = Counter()
    for r in results:
        access = r.get("access", {})
        model = access.get("model", "unknown") if isinstance(access, dict) else "unknown"
        counter[model] += 1

    total = len(results)

    # Compute self-serve total (free + trial)
    self_serve = counter.get("self_serve_free", 0) + counter.get("self_serve_trial", 0)
    gated = counter.get("paid_plan_required", 0) + counter.get("admin_approval", 0) + counter.get("partner_gated", 0)

    return {
        "counts": dict(counter.most_common()),
        "percentages": {k: round(v / total * 100, 1) for k, v in counter.most_common()},
        "self_serve_total": self_serve,
        "self_serve_pct": round(self_serve / total * 100, 1) if total > 0 else 0,
        "gated_total": gated,
        "gated_pct": round(gated / total * 100, 1) if total > 0 else 0,
    }


def compute_mcp_adoption(results: list[dict]) -> dict:
    """MCP adoption rate."""
    has_mcp = 0
    no_mcp = 0
    unknown_mcp = 0

    for r in results:
        api = r.get("api_surface", {})
        mcp = api.get("has_mcp", "unknown") if isinstance(api, dict) else "unknown"
        if mcp is True or mcp == "true":
            has_mcp += 1
        elif mcp is False or mcp == "false":
            no_mcp += 1
        else:
            unknown_mcp += 1

    total = len(results)
    return {
        "has_mcp": has_mcp,
        "no_mcp": no_mcp,
        "unknown": unknown_mcp,
        "adoption_rate_pct": round(has_mcp / total * 100, 1) if total > 0 else 0,
        "mcp_apps": [
            {"id": r["id"], "name": r["name"], "mcp_url": r.get("api_surface", {}).get("mcp_source_url")}
            for r in results
            if r.get("api_surface", {}).get("has_mcp") in [True, "true"]
        ],
    }


def compute_buildability(results: list[dict]) -> dict:
    """Buildability verdict distribution."""
    counter = Counter()
    for r in results:
        verdict = r.get("buildability_verdict", "unknown")
        counter[verdict] += 1

    total = len(results)
    return {
        "counts": dict(counter.most_common()),
        "percentages": {k: round(v / total * 100, 1) for k, v in counter.most_common()},
        "ready_today_pct": round(counter.get("ready_today", 0) / total * 100, 1) if total > 0 else 0,
    }


def compute_blocker_ranking(results: list[dict]) -> list[dict]:
    """Most common blockers, ranked by frequency."""
    counter = Counter()
    for r in results:
        blocker = r.get("main_blocker")
        if blocker and blocker.lower() not in ("null", "none", "n/a"):
            # Normalize similar blockers
            blocker_lower = blocker.lower().strip()
            counter[blocker] += 1

    return [
        {"blocker": blocker, "count": count}
        for blocker, count in counter.most_common(15)
    ]


def compute_category_breakdown(results: list[dict]) -> dict:
    """Per-category breakdown of key metrics."""
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    breakdown = {}
    for category, apps in sorted(by_category.items()):
        total = len(apps)

        # Auth distribution
        auth_counter = Counter()
        for a in apps:
            for m in a.get("auth_methods", ["unknown"]):
                auth_counter[m] += 1

        # Access model
        access_counter = Counter()
        for a in apps:
            model = a.get("access", {}).get("model", "unknown") if isinstance(a.get("access"), dict) else "unknown"
            access_counter[model] += 1

        self_serve = access_counter.get("self_serve_free", 0) + access_counter.get("self_serve_trial", 0)

        # Buildability
        build_counter = Counter()
        for a in apps:
            build_counter[a.get("buildability_verdict", "unknown")] += 1

        # MCP
        mcp_count = sum(1 for a in apps if a.get("api_surface", {}).get("has_mcp") in [True, "true"])

        breakdown[category] = {
            "total_apps": total,
            "auth_distribution": dict(auth_counter),
            "access_distribution": dict(access_counter),
            "self_serve_pct": round(self_serve / total * 100, 1),
            "buildability": dict(build_counter),
            "ready_today_pct": round(build_counter.get("ready_today", 0) / total * 100, 1),
            "mcp_count": mcp_count,
            "apps": [{"id": a["id"], "name": a["name"]} for a in apps],
        }

    return breakdown


def compute_easy_wins(results: list[dict]) -> list[dict]:
    """
    Top easy wins for Composio:
    Self-serve + broad/moderate API + no MCP yet = clear next build.

    These are apps that are immediately buildable and represent the
    highest ROI for Composio's integration team.
    """
    wins = []
    for r in results:
        access_model = r.get("access", {}).get("model", "unknown") if isinstance(r.get("access"), dict) else "unknown"
        api = r.get("api_surface", {})
        breadth = api.get("breadth", "unknown") if isinstance(api, dict) else "unknown"
        has_mcp = api.get("has_mcp", "unknown") if isinstance(api, dict) else "unknown"
        verdict = r.get("buildability_verdict", "unknown")
        composio_in_catalog = r.get("composio_in_catalog", False)

        # Easy win criteria
        is_self_serve = access_model in ("self_serve_free", "self_serve_trial")
        has_broad_api = breadth in ("broad", "moderate")
        no_mcp = has_mcp in (False, "false", "unknown")
        is_ready = verdict in ("ready_today", "buildable_with_workaround")
        not_in_composio = not composio_in_catalog

        if is_self_serve and has_broad_api and is_ready:
            score = 0
            if not_in_composio:
                score += 3  # Not yet in Composio = highest value
            if no_mcp:
                score += 2  # No MCP = opportunity
            if breadth == "broad":
                score += 1
            if access_model == "self_serve_free":
                score += 1

            wins.append({
                "id": r["id"],
                "name": r["name"],
                "category": r["category"],
                "access_model": access_model,
                "api_breadth": breadth,
                "has_mcp": has_mcp,
                "in_composio": composio_in_catalog,
                "score": score,
                "reason": f"Self-serve {'free' if access_model == 'self_serve_free' else 'trial'}, "
                         f"{breadth} API, {'no MCP' if no_mcp else 'has MCP'}, "
                         f"{'not yet' if not_in_composio else 'already'} in Composio",
            })

    wins.sort(key=lambda w: w["score"], reverse=True)
    return wins[:15]


def compute_enterprise_only(results: list[dict]) -> list[dict]:
    """
    Top enterprise-only integrations:
    Partner-gated, no self-serve path — requires outreach.
    """
    enterprise = []
    for r in results:
        access_model = r.get("access", {}).get("model", "unknown") if isinstance(r.get("access"), dict) else "unknown"
        verdict = r.get("buildability_verdict", "unknown")

        if access_model in ("partner_gated", "admin_approval", "paid_plan_required"):
            enterprise.append({
                "id": r["id"],
                "name": r["name"],
                "category": r["category"],
                "access_model": access_model,
                "buildability": verdict,
                "main_blocker": r.get("main_blocker"),
            })

    return enterprise


def compute_confidence_distribution(results: list[dict]) -> dict:
    """Confidence level distribution of the research."""
    counter = Counter()
    for r in results:
        counter[r.get("confidence", "unknown")] += 1

    total = len(results)
    return {
        "counts": dict(counter.most_common()),
        "percentages": {k: round(v / total * 100, 1) for k, v in counter.most_common()},
    }


def generate_executive_summary(
    auth_dist: dict,
    access_dist: dict,
    mcp_adoption: dict,
    buildability: dict,
    easy_wins: list,
    category_breakdown: dict,
) -> list[str]:
    """
    Generate 3-4 plain-English pattern statements for the executive summary.
    These are auto-generated from actual data, not hand-written.
    """
    statements = []

    # Auth pattern
    top_auth = list(auth_dist["counts"].keys())[:2]
    if top_auth:
        pcts = [f"{auth_dist['percentages'].get(a, 0)}%" for a in top_auth]
        statements.append(
            f"{top_auth[0].replace('_', ' ').title()} dominates authentication "
            f"({pcts[0]} of apps)"
            + (f", followed by {top_auth[1].replace('_', ' ').title()} ({pcts[1]})" if len(top_auth) > 1 else "")
            + "."
        )

    # Self-serve pattern
    ss_pct = access_dist.get("self_serve_pct", 0)
    statements.append(
        f"{ss_pct}% of apps offer self-serve API access, "
        f"making them immediate integration candidates for Composio."
    )

    # MCP adoption
    mcp_pct = mcp_adoption.get("adoption_rate_pct", 0)
    statements.append(
        f"Only {mcp_pct}% of apps have MCP servers — "
        f"a massive opportunity for Composio to provide AI-native access."
    )

    # Buildability
    ready_pct = buildability.get("ready_today_pct", 0)
    statements.append(
        f"{ready_pct}% of apps are ready for toolkit integration today, "
        f"with {len(easy_wins)} identified as high-priority easy wins."
    )

    return statements


def main():
    """Compute all analytics and write summary.json."""
    print("=" * 60)
    print("Computing Analytics & Insights")
    print("=" * 60)

    results, verification_stats, comparison_records = load_results()
    print(f"Loaded {len(results)} app results")

    # Compute all metrics
    auth_dist = compute_auth_distribution(results)
    api_type_dist = compute_api_type_distribution(results)
    access_dist = compute_access_model_distribution(results)
    mcp_adoption = compute_mcp_adoption(results)
    buildability = compute_buildability(results)
    blocker_ranking = compute_blocker_ranking(results)
    category_breakdown = compute_category_breakdown(results)
    easy_wins = compute_easy_wins(results)
    enterprise_only = compute_enterprise_only(results)
    confidence_dist = compute_confidence_distribution(results)

    # Generate executive summary
    exec_summary = generate_executive_summary(
        auth_dist, access_dist, mcp_adoption, buildability, easy_wins, category_breakdown
    )

    # Assemble summary
    summary = {
        "executive_summary": exec_summary,
        "total_apps": len(results),
        "auth_distribution": auth_dist,
        "api_type_distribution": api_type_dist,
        "access_model_distribution": access_dist,
        "mcp_adoption": mcp_adoption,
        "buildability": buildability,
        "blocker_ranking": blocker_ranking,
        "category_breakdown": category_breakdown,
        "easy_wins": easy_wins,
        "enterprise_only": enterprise_only,
        "confidence_distribution": confidence_dist,
        "verification": verification_stats,
        "comparison_records": comparison_records,
    }

    # Write summary
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary written to {SUMMARY_FILE}")
    print(f"\nExecutive Summary:")
    for i, stmt in enumerate(exec_summary, 1):
        print(f"  {i}. {stmt}")

    print(f"\nKey Metrics:")
    print(f"  Auth: {auth_dist['counts']}")
    print(f"  Self-serve: {access_dist['self_serve_pct']}%")
    print(f"  MCP adoption: {mcp_adoption['adoption_rate_pct']}%")
    print(f"  Ready today: {buildability['ready_today_pct']}%")
    print(f"  Easy wins: {len(easy_wins)}")
    print(f"  Enterprise-only: {len(enterprise_only)}")

    if verification_stats:
        print(f"\nVerification:")
        print(f"  Overall accuracy: {verification_stats.get('overall_accuracy_pct', 'N/A')}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
