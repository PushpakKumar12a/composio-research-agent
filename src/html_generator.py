import json
import os
import sys
import time
from collections import Counter
from typing import Dict, Any, List

from jinja2 import Environment, FileSystemLoader

from config.settings import (
    RESULTS_FILE, VERIFIED_FILE, PATTERNS_FILE, TEMPLATES_DIR,
    OUTPUT_HTML, CATEGORIES
)

def analyze_patterns(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)

    auth_counts = Counter()
    serve_counts = Counter()
    type_counts = Counter()
    build_counts = Counter()
    blocker_counts = Counter()
    mcp_count = 0
    easy_wins = []

    build_by_cat: Dict[str, Counter] = {}
    for cat in CATEGORIES:
        build_by_cat[cat] = Counter()

    for r in results:
        for method in r.get("auth_methods", []):
            auth_counts[method] += 1

        serve_counts[r.get("self_serve", "Unknown")] += 1
        type_counts[r.get("api_type", "Unknown")] += 1
        verdict = r.get("buildability_verdict", "Unknown")
        build_counts[verdict] += 1

        if r.get("has_mcp"):
            mcp_count += 1

        blocker = r.get("main_blocker", "")
        if blocker and blocker != "None":
            blocker_counts[blocker] += 1

        cat = r.get("category", "")
        if cat in build_by_cat:
            build_by_cat[cat][verdict] += 1

        if verdict == "Easy" and "Self-Serve" in r.get("self_serve", ""):
            easy_wins.append({
                "id": r["id"],
                "name": r["name"],
                "category": r["category"]
            })

    category_patterns = {}
    for cat, cnt in build_by_cat.items():
        category_patterns[cat] = dict(cnt)

    patterns = {
        "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_apps": total,
        "auth_distribution": dict(auth_counts.most_common()),
        "self_serve_distribution": dict(serve_counts.most_common()),
        "api_type_distribution": dict(type_counts.most_common()),
        "buildability_distribution": dict(build_counts.most_common()),
        "mcp_count": mcp_count,
        "mcp_pct": round(mcp_count / total * 100) if total else 0,
        "top_blockers": dict(blocker_counts.most_common(5)),
        "buildability_by_category": category_patterns,
        "easy_wins_count": len(easy_wins),
        "easy_wins": easy_wins,
    }

    return patterns

def generate_html_report():
    print("Loading data...")
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        print("   Run research_agent.py first.")
        sys.exit(1)

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    verification = {}
    if os.path.exists(VERIFIED_FILE):
        with open(VERIFIED_FILE, "r", encoding="utf-8") as f:
            verification = json.load(f)

    print(f"Analyzing patterns across {len(results)} apps...")
    patterns = analyze_patterns(results)

    with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)
    print(f"Patterns saved to {PATTERNS_FILE}")

    total = len(results)
    auth_counts = patterns.get("auth_distribution", {})
    top_auth = list(auth_counts.keys())[0] if auth_counts else "OAuth2"
    top_auth_pct = round(auth_counts.get(top_auth, 0) / total * 100) if total else 0

    serve_counts = patterns.get("self_serve_distribution", {})
    self_serve_total = sum(v for k, v in serve_counts.items() if "Self-Serve" in k)
    self_serve_pct = round(self_serve_total / total * 100) if total else 0

    build_counts = patterns.get("buildability_distribution", {})
    easy_count = build_counts.get("Easy", 0)
    easy_pct = round(easy_count / total * 100) if total else 0

    type_counts = patterns.get("api_type_distribution", {})
    rest_count = type_counts.get("REST", 0)
    rest_pct = round(rest_count / total * 100) if total else 0

    v_metrics = verification.get("metrics", {})
    v_accuracy = v_metrics.get("overall_accuracy_pct", "N/A")
    v_sample = verification.get("sample_size", 0)

    print("Generating HTML...")
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    template = env.get_template("report_template.html")

    html_content = template.render(
        total_apps=total,
        top_auth=top_auth,
        top_auth_pct=top_auth_pct,
        self_serve_pct=self_serve_pct,
        easy_pct=easy_pct,
        rest_pct=rest_pct,
        mcp_pct=patterns.get("mcp_pct", 0),
        v_sample=v_sample,
        v_accuracy=v_accuracy,
        generation_date=time.strftime("%Y-%m-%d %H:%M:%S"),
        results_json=json.dumps(results, ensure_ascii=False),
        patterns_json=json.dumps(patterns, ensure_ascii=False),
        verification_json=json.dumps(verification, ensure_ascii=False),
        categories_json=json.dumps(CATEGORIES, ensure_ascii=False),
    )

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML deliverable generated: {OUTPUT_HTML}")
    print(f"   Size: {len(html_content):,} bytes")

if __name__ == "__main__":
    generate_html_report()
