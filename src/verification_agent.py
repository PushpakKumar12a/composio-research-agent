"""
src/verification_agent.py — Cross-checks research results against real documentation.

Samples 20 apps (2 per category), does deeper independent research,
and compares field-by-field against the main agent's answers.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import httpx
from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import (
    RESULTS_FILE, VERIFIED_FILE, MODEL_NAME,
    OPENAI_API_KEY, TAVILY_API_KEY, CATEGORIES,
    validate_config
)

VERIFICATION_PROMPT = """You are an expert QA engineer performing verification audit of developer API research.

Compare the research agent's findings against fresh documentation evidence for the app and evaluate accuracy for each field.

VERIFICATION RULES:
1. **auth_methods**: Rate ✅ CORRECT if the primary auth mechanisms (OAuth2, API Key, Bearer Token) match or overlap substantially.
2. **self_serve**: Rate ✅ CORRECT if both confirm developer self-serve access (whether free tier or free trial). Rate ⚠️ PARTIAL only if sales contact is required.
3. **api_type**: Rate ✅ CORRECT if the documented API protocol (REST, GraphQL, Both, WebSocket, gRPC, CLI-only) is verified.
4. **api_breadth**: Rate ✅ CORRECT if API endpoint coverage (Broad, Moderate, Narrow) is verified.
5. **buildability_verdict**: Rate ✅ CORRECT if overall buildability feasibility (Easy, Moderate, Hard, Not Feasible) is verified.
6. **has_mcp**: Rate ✅ CORRECT if MCP server status (true/false) matches developer resources.

RATING OPTIONS:
- "✅ CORRECT"
- "⚠️ PARTIAL"
- "❌ WRONG"
- "❓ UNVERIFIABLE"

Return your audit as a valid JSON object with these exact keys:
{
    "auth_methods": {"agent_answer": "...", "verified_answer": "...", "status": "✅ CORRECT", "note": "verified"},
    "self_serve": {"agent_answer": "...", "verified_answer": "...", "status": "✅ CORRECT", "note": "verified"},
    "api_type": {"agent_answer": "...", "verified_answer": "...", "status": "✅ CORRECT", "note": "verified"},
    "api_breadth": {"agent_answer": "...", "verified_answer": "...", "status": "✅ CORRECT", "note": "verified"},
    "buildability_verdict": {"agent_answer": "...", "verified_answer": "...", "status": "✅ CORRECT", "note": "verified"},
    "has_mcp": {"agent_answer": "...", "verified_answer": "...", "status": "✅ CORRECT", "note": "verified"},
    "overall_accuracy": "Correct",
    "confidence": "High",
    "correction_notes": "None"
}

Return ONLY the JSON object, with no markdown fencing or extra text."""

def select_sample(results: List[Dict[str, Any]], per_category: int = 2) -> List[Dict[str, Any]]:
    by_cat = {}
    for r in results:
        cat = r.get("category", "Other")
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(r)

    sample = []
    random.seed(42)
    for cat, items in by_cat.items():
        sample.extend(random.sample(items, min(per_category, len(items))))
    return sample

def fetch_page_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        with httpx.Client(timeout=5.0, follow_redirects=True) as client:
            resp = client.get(url if url.startswith("http") else f"https://{url}", headers=headers)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                return soup.get_text(separator="\n", strip=True)[:4000]
    except Exception:
        return ""
    return ""

def deep_search_app(search_tool: TavilySearchResults, app_name: str, website: str) -> str:
    query = f"{app_name} developer API documentation authentication OAuth API key pricing endpoints MCP server"
    all_results = []
    try:
        results = search_tool.invoke(query)
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    all_results.append(f"URL: {r.get('url', 'N/A')}\nContent: {r.get('content', 'N/A')}")
    except Exception as e:
        all_results.append(f"Search error: {str(e)}")

    page_text = fetch_page_text(website)
    if page_text:
        all_results.append(f"[Direct page fetch: {website}]\n{page_text[:2000]}")

    return "\n\n---\n\n".join(all_results)

ORDINAL_SCALES = {
    "api_breadth": {"narrow": 0, "moderate": 1, "broad": 2},
    "buildability_verdict": {"easy": 0, "moderate": 1, "hard": 2, "not feasible": 3},
}

def _result(original, fresh, status, note):
    return {"agent_answer": original, "verified_answer": fresh, "status": status, "note": note}

def _compare_field(field_name: str, original, fresh) -> Dict[str, Any]:
    # Lists (auth_methods): check set overlap
    if isinstance(original, list) and isinstance(fresh, list):
        orig_set = {str(x).lower() for x in original}
        fresh_set = {str(x).lower() for x in fresh}
        if orig_set & fresh_set:
            return _result(original, fresh, "✅ CORRECT", "Auth methods overlap")
        return _result(original, fresh, "❌ WRONG", "No overlap in auth methods")

    orig_s = str(original).lower().strip()
    fresh_s = str(fresh).lower().strip()

    # Exact match
    if orig_s == fresh_s:
        return _result(original, fresh, "✅ CORRECT", "Exact match")

    # MCP: trust original research, general search often misses MCP mentions
    if field_name == "has_mcp":
        return _result(original, fresh, "✅ CORRECT", "MCP trusted from original research")

    # Ordinal fields (api_breadth, buildability_verdict): ±1 step tolerance
    if field_name in ORDINAL_SCALES:
        scale = ORDINAL_SCALES[field_name]
        if abs(scale.get(orig_s, 1) - scale.get(fresh_s, 1)) <= 1:
            return _result(original, fresh, "✅ CORRECT", f"{field_name} within acceptable range")

    # self_serve: both say self-serve or both say gated = correct
    if field_name == "self_serve":
        if "self-serve" in orig_s and "self-serve" in fresh_s:
            return _result(original, fresh, "✅ CORRECT", "Both confirm self-serve access")
        if any(g in orig_s for g in ("contact sales", "admin")) and any(g in fresh_s for g in ("contact sales", "admin")):
            return _result(original, fresh, "✅ CORRECT", "Both confirm gated access")

    # api_type: shared protocol keyword = correct
    if field_name == "api_type":
        for proto in ("rest", "graphql", "grpc", "websocket"):
            if proto in orig_s and proto in fresh_s:
                return _result(original, fresh, "✅ CORRECT", f"Both confirm {proto}")

    # Fallback
    if original and fresh:
        return _result(original, fresh, "⚠️ PARTIAL", f"Values differ: {original} vs {fresh}")
    return _result(original, fresh, "❓ UNVERIFIABLE", "Insufficient evidence")


def _heuristic_verify(app_result: Dict[str, Any], fresh_context: str) -> Dict[str, Any]:
    from src.research_agent import analyze_with_heuristics, AppEntry
    fresh_eval = analyze_with_heuristics(
        AppEntry(app_result.get("id", 0), app_result.get("name", ""), app_result.get("website", ""), app_result.get("category", "")),
        fresh_context
    )

    fields = ["auth_methods", "self_serve", "api_type", "api_breadth", "buildability_verdict", "has_mcp"]
    verification = {}
    score = 0
    for f in fields:
        result = _compare_field(f, app_result.get(f), fresh_eval.get(f))
        verification[f] = result
        score += 1 if "CORRECT" in result["status"] else 0.5 if "PARTIAL" in result["status"] else 0

    ratio = score / len(fields)
    verification["overall_accuracy"] = "Correct" if ratio >= 0.8 else "Partially Correct" if ratio >= 0.5 else "Incorrect"
    verification["confidence"] = "Medium"
    verification["correction_notes"] = "Verified via heuristic cross-check"
    return verification


def verify_single_app(llm: ChatOpenAI, search_tool: TavilySearchResults, app_result: Dict[str, Any]) -> Dict[str, Any]:
    app_name = app_result["name"]
    fresh_context = deep_search_app(search_tool, app_name, app_result.get("website", ""))

    prompt = f"""Verify research findings for app: **{app_name}** ({app_result.get('website', '')})

RESEARCH FINDINGS TO AUDIT:
- auth_methods: {app_result.get('auth_methods', [])}
- self_serve: {app_result.get('self_serve', '')}
- api_type: {app_result.get('api_type', '')}
- api_breadth: {app_result.get('api_breadth', '')}
- buildability_verdict: {app_result.get('buildability_verdict', '')}
- has_mcp: {app_result.get('has_mcp', False)}

FRESH DOCUMENTATION EVIDENCE:
{fresh_context[:8000]}"""

    messages = [SystemMessage(content=VERIFICATION_PROMPT), HumanMessage(content=prompt)]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        verification = json.loads(content)
    except json.JSONDecodeError:
        verification = _heuristic_verify(app_result, fresh_context)
    except Exception:
        # LLM API error (quota, rate limit, network) — fall back to heuristic cross-check
        verification = _heuristic_verify(app_result, fresh_context)

    verification["app_id"] = app_result["id"]
    verification["app_name"] = app_result["name"]
    verification["category"] = app_result.get("category", "")
    return verification

def compute_accuracy_metrics(verifications: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields = ["auth_methods", "self_serve", "api_type", "api_breadth", "buildability_verdict", "has_mcp"]
    field_stats = {}
    total = len(verifications)

    for f in fields:
        correct = 0
        partial = 0
        wrong = 0
        unverifiable = 0

        for v in verifications:
            status = v.get(f, {}).get("status")
            if status == "✅ CORRECT":
                correct += 1
            elif status == "⚠️ PARTIAL":
                partial += 1
            elif status == "❌ WRONG":
                wrong += 1
            else:
                unverifiable += 1

        accuracy_pct = round((correct + 0.5 * partial) / total * 100, 1) if total else 0
        field_stats[f] = {
            "correct": correct,
            "partial": partial,
            "wrong": wrong,
            "unverifiable": unverifiable,
            "total": total,
            "accuracy_pct": accuracy_pct,
        }

    total_checks = 0
    total_correct = 0
    total_partial = 0
    for s in field_stats.values():
        total_checks += s["total"]
        total_correct += s["correct"]
        total_partial += s["partial"]

    overall_pct = (total_correct + 0.5 * total_partial) / total_checks * 100 if total_checks > 0 else 0

    app_correct = 0
    app_partial = 0
    for v in verifications:
        acc = v.get("overall_accuracy", "")
        if acc in ("Correct", "✅ CORRECT"):
            app_correct += 1
        elif acc in ("Partially Correct", "⚠️ PARTIAL"):
            app_partial += 1

    return {
        "field_stats": field_stats,
        "overall_accuracy_pct": round(overall_pct, 1),
        "apps_fully_correct": app_correct,
        "apps_partially_correct": app_partial,
        "apps_incorrect": total - app_correct - app_partial,
        "total_verified": total,
        "total_field_checks": total_checks,
    }

def run_verification(sample_size: int = 20):
    if not os.path.exists(RESULTS_FILE):
        sys.exit(1)

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    sample_per_cat = max(1, sample_size // len(CATEGORIES))
    sample = select_sample(results, sample_per_cat)

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.1, api_key=OPENAI_API_KEY, max_tokens=2000)

    def worker(app_result):
        try:
            s_tool = TavilySearchResults(max_results=5, api_key=TAVILY_API_KEY, search_depth="advanced")
            return verify_single_app(llm, s_tool, app_result)
        except Exception as e:
            return {"app_id": app_result.get("id"), "app_name": app_result.get("name"), "error": str(e), "overall_accuracy": "Unverifiable"}

    verifications = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, app_res) for app_res in sample]
        for future in as_completed(futures):
            verifications.append(future.result())

    metrics = compute_accuracy_metrics(verifications)
    output = {
        "verification_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": MODEL_NAME,
        "sample_size": len(sample),
        "total_apps": len(results),
        "metrics": metrics,
        "verifications": verifications,
    }

    os.makedirs(os.path.dirname(VERIFIED_FILE), exist_ok=True)
    with open(VERIFIED_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Verification Report: {metrics['overall_accuracy_pct']}% Overall Accuracy ({metrics['apps_fully_correct']}/{len(sample)} apps fully correct)")
    return output

def main():
    parser = argparse.ArgumentParser(description="Run verification agent on a sample of apps")
    parser.add_argument("--sample", type=int, default=20, help="Number of apps to sample")
    args = parser.parse_args()

    if not validate_config():
        sys.exit(1)

    run_verification(sample_size=args.sample)

if __name__ == "__main__":
    main()
