import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

from composio import Composio
from composio_langchain import LangchainProvider

from config.settings import (
    parse_apps_file, AppEntry, AppResearchResult,
    RESULTS_FILE, CHECKPOINT_FILE, MODEL_NAME, BATCH_SIZE,
    OPENAI_API_KEY, TAVILY_API_KEY, COMPOSIO_API_KEY, validate_config
)

RESEARCH_SYSTEM_PROMPT = """You are an expert API analyst researching apps for Composio (composio.dev), a platform that turns apps into tools AI agents can call.

For each app, you must determine:
1. **Description**: What the app does in ONE line (max 15 words).
2. **Auth Methods**: Which authentication methods the API supports. Choose from: OAuth2, API Key, Basic Auth, Bearer Token, JWT, Session/Cookie, HMAC, No Auth, Other. List ALL that apply.
3. **Self-Serve Access**: Can a developer get API credentials themselves?
   - "Self-Serve Free" = free tier with API access, no approval needed
   - "Self-Serve Trial" = free trial available with API access
   - "Paid Plan Required" = need a paid subscription to access API
   - "Admin/Partner Gated" = need admin approval or partnership
   - "Contact Sales" = must contact sales team for API access
4. **API Type**: Determine the exact API surface(s) supported based on official developer documentation:
   - "Both" = platform offers BOTH REST and GraphQL APIs (e.g. Shopify, GitHub, HubSpot, Twenty, Salesforce) or REST + WebSockets (e.g. Slack, Discord, Binance, ClickUp)
   - "GraphQL" = GraphQL is the primary API surface (e.g. Linear, Monday.com, Plain)
   - "REST" = RESTful HTTP JSON API (e.g. Stripe, Plaid, Intercom, Zendesk, Notion, Airtable, Meta Ads)
   - "WebSocket" = Realtime WebSocket streaming API primary
   - "gRPC" = gRPC / Protocol Buffers API primary (e.g. Google Ads API)
   - "CLI-only" = Command-line interface without a public web HTTP API (e.g. Sherlock, Mermaid CLI)
   - "SDK-only" = Python/Node SDK wrapper primary
   - "None" = No documented public API
5. **API Breadth**: How comprehensive is the API coverage?
   - "Very Broad" = 50+ endpoints covering most app features
   - "Broad" = 20-50 endpoints, good coverage
   - "Moderate" = 10-20 endpoints, covers core features
   - "Narrow" = <10 endpoints or limited functionality
6. **Has MCP**: Does this app have an existing MCP (Model Context Protocol) server? Answer true/false and provide details if yes.
7. **Existing in Composio**: Is this app already available as a Composio toolkit? Answer true/false.
8. **Buildability Verdict**: Could this become an agent toolkit today?
   - "Easy" = well-documented REST/GraphQL API, self-serve auth, standard patterns
   - "Moderate" = good API but complex auth or limited docs
   - "Hard" = API exists but gated, poorly documented, or non-standard
   - "Not Feasible" = no public API, or entirely CLI/desktop-only
9. **Main Blocker**: If not "Easy", what's the primary obstacle?
10. **Evidence URL**: The most authoritative official developer documentation URL you found.

IMPORTANT RULES:
- Base answers strictly on EVIDENCE from search results.
- For API Type, check if the documentation mentions GraphQL endpoints, schema definitions, WebSockets, gRPC, or standard REST endpoints. If both REST and GraphQL/WebSockets are documented, select "Both".
- For auth methods, look for the actual developer docs authentication page.
- For self-serve, check if there's a free signup/trial or if it says "Contact Sales".

Return your analysis as a valid JSON object with these exact keys:
{
    "description": "string",
    "auth_methods": ["string"],
    "self_serve": "string",
    "api_type": "string",
    "api_breadth": "string",
    "has_mcp": false,
    "mcp_details": "string or empty",
    "existing_composio": false,
    "buildability_verdict": "string",
    "main_blocker": "string or empty",
    "evidence_url": "string",
    "raw_search_notes": "brief notes on what you found"
}

Return ONLY the JSON object, no markdown fencing or additional text."""

def init_composio_client():
    if not COMPOSIO_API_KEY:
        return None
    try:
        return Composio(provider=LangchainProvider(), api_key=COMPOSIO_API_KEY)
    except Exception:
        return None

def check_composio_toolkit(composio_client, app_name: str) -> dict:
    if composio_client is None:
        return {"exists": False, "details": "Composio API key not configured"}
    try:
        session = composio_client.sessions.create(user_id="research-agent")
        result = session.search(query=f"{app_name} integration")
        if result and hasattr(result, 'items') and len(result.items) > 0:
            tool_names = [item.slug for item in result.items[:3]]
            return {"exists": True, "details": f"Found toolkits: {', '.join(tool_names)}"}
        return {"exists": False, "details": "No matching toolkit found"}
    except Exception as e:
        return {"exists": False, "details": f"Search error: {str(e)[:100]}"}

def get_base_domain(url_or_domain: str) -> str:
    clean = url_or_domain.lower().strip()
    clean = re.sub(r'\s*\(.*?\)', '', clean)
    clean = clean.replace('https://', '').replace('http://', '').split('/')[0].split('?')[0]
    parts = clean.split('.')
    if len(parts) >= 2:
        if parts[-2] in ('co', 'com', 'org', 'net', 'edu', 'gov') and len(parts) >= 3:
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])
    return clean

THIRD_PARTY_BLACKLIST = [
    'medium.com', 'dev.to', 'flosum.com', 'getphyllo.com', 'apis.io', 
    'postman.com', 'stackoverflow.com', 'wikipedia.org', 'zapier.com', 'make.com',
    'quora.com', 'reddit.com'
]

def extract_official_docs_url(app: AppEntry, search_context: str) -> str:
    domain = get_base_domain(app.website)
    urls = re.findall(r'https?://[^\s\n"\'<>]+', search_context)

    official_doc_urls = []
    official_urls = []

    for u in urls:
        u_lower = u.lower()
        if any(b in u_lower for b in THIRD_PARTY_BLACKLIST):
            continue

        u_domain = get_base_domain(u)
        if u_domain == domain or domain in u_domain or u_domain in domain:
            official_urls.append(u)
            path = urllib.parse.urlparse(u).path.lower()
            if any(k in path or k in u_lower for k in ['doc', 'dev', 'api', 'reference', 'guide', 'auth', 'developer', 'endpoint']):
                official_doc_urls.append(u)

    if official_doc_urls:
        return official_doc_urls[0]
    if official_urls:
        return official_urls[0]

    web = app.website.strip().split()[0]
    return web if web.startswith('http') else f"https://{web}"

def create_search_tool() -> TavilySearchResults:
    return TavilySearchResults(
        max_results=6,
        api_key=TAVILY_API_KEY,
        search_depth="advanced",
    )

def search_app_with_tool(search_tool: TavilySearchResults, app: AppEntry) -> str:
    query = f"{app.name} official developer API documentation authentication pricing MCP server"
    all_results = []
    try:
        results = search_tool.invoke(query)
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    all_results.append(f"URL: {r.get('url', 'N/A')}\nContent: {r.get('content', 'N/A')}")
                else:
                    all_results.append(str(r))
        else:
            all_results.append(str(results))
    except Exception as e:
        all_results.append(f"Search error for '{query}': {e}")
    return "\n\n---\n\n".join(all_results)

AUTH_PATTERNS = [
    (r'\boauth\s*2(?:\.0)?\b|\boauth\b|\bauthorization_code\b', "OAuth2"),
    (r'\bapi[-_]?key\b|\bsecret[-_]?key\b|\bx-api-key\b|\bpersonal\s*access\s*token\b', "API Key"),
    (r'\bbasic\s*auth\b', "Basic Auth"),
    (r'\bjwt\b|\bjson\s*web\s*token\b', "JWT"),
    (r'\bhmac\b|\bsignature\b', "HMAC"),
]

SELF_SERVE_PATTERNS = [
    (r'\bcontact\s*sales\b.*\bapi\b|\bapi\b.*\bcontact\s*sales\b|\benterprise\s*api\b|\brequest\s*api\s*access\b', "Contact Sales"),
    (r'\bpartner\s*program\b|\bdeveloper\s*approval\b|\bapp\s*review\b|\bgated\s*api\b|\badmin\s*approval\b', "Admin/Partner Gated"),
    (r'\bpaid\s*plan\b.*\bapi\b|\bpaid\s*subscription\b|\bapi\b.*\bpaid\s*tier\b', "Paid Plan Required"),
    (r'\bfree\s*tier\b|\bfree\s*developer\b|\bfree\s*account\b|\bfree\s*plan\b|\bself-serve\b|\bcreate\s*an\s*account\b', "Self-Serve Free"),
    (r'\bfree\s*trial\b|\b14-day\s*trial\b|\b30-day\s*trial\b', "Self-Serve Trial"),
]

def analyze_with_heuristics(app: AppEntry, search_context: str) -> Dict[str, Any]:
    text = search_context.lower()

    auth_methods = [label for pattern, label in AUTH_PATTERNS if re.search(pattern, text)] or ["API Key"]

    self_serve = next((label for pattern, label in SELF_SERVE_PATTERNS if re.search(pattern, text)),
                      "Self-Serve Free" if any(m in auth_methods for m in ("OAuth2", "API Key")) else "Paid Plan Required")

    has_graphql = bool(re.search(r'\bgraphql\b|\bgraph ql\b|\bapollo\b', text))
    has_rest = bool(re.search(r'\brest\b|\brestful\b|\bjson\s*api\b|\bhttp\s*api\b|\bendpoint\b', text))
    has_grpc = bool(re.search(r'\bgrpc\b|\bprotobuf\b', text))
    has_websocket = bool(re.search(r'\bwebsocket\b|\bweb\s*sockets\b|\bwss://\b|\brealtime\b', text))
    has_cli = bool(re.search(r'\bcli\b|\bcommand\s*line\b|\bterminal\b', text))

    if (has_graphql and has_rest) or (has_websocket and has_rest) or (has_grpc and has_rest):
        api_type = "Both"
    elif has_graphql:
        api_type = "GraphQL"
    elif has_grpc:
        api_type = "gRPC"
    elif has_websocket:
        api_type = "WebSocket"
    elif has_cli and not has_rest and not has_graphql:
        api_type = "CLI-only"
    else:
        api_type = "REST" if (has_rest or len(search_context) > 200) else "None"

    api_breadth = "Broad" if (len(text) > 4000 or re.search(r'\bcomprehensive\b|\b50\+\b|\bhundreds\b', text)) else \
                  "Narrow" if re.search(r'\blimited\b|\bread-only\b|\bnarrow\b', text) else "Moderate"

    has_mcp = bool(re.search(r'\bmcp\s*server\b|\bmodel\s+context\s+protocol\s+server\b|\bmcp-server-\w+\b|\bofficial\s+mcp\b', text))

    if api_type in ("REST", "GraphQL", "Both") and self_serve in ("Self-Serve Free", "Self-Serve Trial") and any(m in auth_methods for m in ("OAuth2", "API Key")):
        verdict, blocker = "Easy", ""
    elif self_serve in ("Contact Sales", "Admin/Partner Gated"):
        verdict, blocker = "Hard", "Partner/Sales gated access or approval required"
    elif api_type == "None" or "no public api" in text or "discontinued api" in text:
        verdict, blocker = "Not Feasible", "No documented public API"
    else:
        verdict, blocker = "Moderate", "Requires paid plan or complex auth setup"

    return {
        "description": f"{app.name} — {app.category} platform",
        "auth_methods": auth_methods,
        "self_serve": self_serve,
        "api_type": api_type,
        "api_breadth": api_breadth,
        "has_mcp": has_mcp,
        "mcp_details": "MCP server or integration reference found" if has_mcp else "",
        "existing_composio": False,
        "buildability_verdict": verdict,
        "main_blocker": blocker,
        "evidence_url": extract_official_docs_url(app, search_context),
        "raw_search_notes": f"Analyzed via Tavily search context heuristic extraction ({len(search_context)} chars)",
    }

AUTH_CANONICAL_MAP = [
    ("oauth", "OAuth2"),
    ("api key", "API Key"), ("api-key", "API Key"), ("api token", "API Key"), ("service key", "API Key"), ("private app", "API Key"),
    ("basic", "Basic Auth"),
    ("jwt", "JWT"), ("json web token", "JWT"),
    ("hmac", "HMAC"),
    ("session", "Session/Cookie"), ("cookie", "Session/Cookie"),
    ("no auth", "No Auth"), ("without auth", "No Auth"),
]

SELF_SERVE_CANONICAL_MAP = [
    ("sales", "Contact Sales"), ("contact", "Contact Sales"),
    ("partner", "Admin/Partner Gated"), ("gated", "Admin/Partner Gated"), ("admin", "Admin/Partner Gated"), ("approval", "Admin/Partner Gated"),
    ("paid", "Paid Plan Required"), ("subscription", "Paid Plan Required"),
    ("trial", "Self-Serve Trial"),
    ("free", "Self-Serve Free"), ("self", "Self-Serve Free"),
]

def normalize_auth_methods(auth_methods: Any) -> List[str]:
    if not isinstance(auth_methods, list):
        auth_methods = [auth_methods] if auth_methods else []

    normalized = []
    for raw in auth_methods:
        low = str(raw).strip().lower()
        if "bearer" in low:
            continue
        canonical = next((label for kw, label in AUTH_CANONICAL_MAP if kw in low), "Other")
        if canonical not in normalized:
            normalized.append(canonical)

    return normalized or ["Other"]

def normalize_self_serve(value: str) -> str:
    low = str(value or "").strip().lower()
    return next((label for kw, label in SELF_SERVE_CANONICAL_MAP if kw in low), "Self-Serve Free")

def analyze_with_llm(llm: ChatOpenAI, app: AppEntry, search_context: str) -> Dict[str, Any]:
    prompt = (
        f"Research the following app and provide your analysis:\n\n"
        f"**App**: {app.name}\n"
        f"**Website**: {app.website}\n"
        f"**Category**: {app.category}\n\n"
        f"**Search Results**:\n{search_context[:8000]}\n\n"
        f"Analyze the above search results and provide your structured JSON assessment."
    )

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        try:
            result = json.loads(content)
            result["auth_methods"] = normalize_auth_methods(result.get("auth_methods", []))
            result["self_serve"] = normalize_self_serve(result.get("self_serve", ""))
            return result
        except json.JSONDecodeError:
            return analyze_with_heuristics(app, search_context)
    except Exception as e:
        return analyze_with_heuristics(app, search_context)

def research_single_app(
    llm: ChatOpenAI,
    search_tool: TavilySearchResults,
    app: AppEntry,
    composio_client=None,
) -> AppResearchResult:
    print(f"  [Research] #{app.id}: {app.name} ({app.website})...")

    search_context = search_app_with_tool(search_tool, app)
    analysis = analyze_with_llm(llm, app, search_context)

    composio_check = check_composio_toolkit(composio_client, app.name)
    if composio_check["exists"]:
        analysis["existing_composio"] = True
        analysis["mcp_details"] = (
            analysis.get("mcp_details", "") +
            f" | Composio: {composio_check['details']}"
        ).strip(" |")

    result = AppResearchResult(
        id=app.id,
        name=app.name,
        website=app.website,
        category=app.category,
        description=analysis.get("description", ""),
        auth_methods=analysis.get("auth_methods", []),
        self_serve=analysis.get("self_serve", ""),
        api_type=analysis.get("api_type", ""),
        api_breadth=analysis.get("api_breadth", ""),
        has_mcp=analysis.get("has_mcp", False),
        mcp_details=analysis.get("mcp_details", ""),
        existing_composio=analysis.get("existing_composio", False),
        buildability_verdict=analysis.get("buildability_verdict", ""),
        main_blocker=analysis.get("main_blocker", ""),
        evidence_url=extract_official_docs_url(app, search_context),
        raw_search_notes=analysis.get("raw_search_notes", ""),
    )

    auth_str = ", ".join(result.auth_methods)
    print(f"  [Done]     #{app.id} {app.name}: "
          f"{result.buildability_verdict} | Auth: {auth_str} | {result.self_serve}")
    return result

# ── Checkpoint & Persistence ────────────────────────────────────────────────

def load_checkpoint() -> Dict[int, Dict]:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {item["id"]: item for item in data}
    return {}

def save_checkpoint(results: List[Dict]):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def save_results(results: List[Dict]):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Results -> {RESULTS_FILE}")

# ── Pipeline Orchestration ───────────────────────────────────────────────────

def run_research(apps: List[AppEntry], resume: bool = False):
    if not validate_config():
        sys.exit(1)

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.1, api_key=OPENAI_API_KEY, max_tokens=1500)
    composio_client = init_composio_client()

    completed = load_checkpoint() if resume else {}
    if completed:
        print(f"[Resume] {len(completed)} apps already done")

    remaining = []
    for a in apps:
        if a.id not in completed:
            remaining.append(a)

    all_results = list(completed.values())
    total = len(apps)
    done = len(completed)

    print(f"\n[Start] Researching {len(remaining)} apps ({done}/{total} done)")

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n{'='*60}\n[Batch {batch_num}/{total_batches}] {len(batch)} apps\n{'='*60}")

        def worker(app):
            try:
                t_tool = create_search_tool()
                res = research_single_app(llm, t_tool, app, composio_client)
                return app.id, res.to_dict()
            except Exception as e:
                print(f"  [Error] {app.name}: {e}")
                return app.id, AppResearchResult(
                    id=app.id, name=app.name, website=app.website,
                    category=app.category, description=f"Error: {e}",
                    buildability_verdict="Unknown", main_blocker=f"Research error: {e}"
                ).to_dict()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for app in batch:
                futures.append(executor.submit(worker, app))
            for future in as_completed(futures):
                app_id, res_dict = future.result()
                all_results.append(res_dict)
                completed[app_id] = res_dict
                done += 1

        save_checkpoint(list(completed.values()))
        print(f"\n[Checkpoint] {done}/{total} complete")

    all_results.sort(key=lambda x: x["id"])
    save_results(all_results)
    return all_results

def run_test(count: int = 5):
    apps = parse_apps_file()
    test_apps = apps[:count]
    print(f"[Test] Researching first {len(test_apps)} sample apps:\n" + ", ".join(a.name for a in test_apps) + "\n")
    return run_research(test_apps)

def main():
    parser = argparse.ArgumentParser(description="Research 100 apps for Composio toolkit buildability")
    parser.add_argument("--test", action="store_true", help="Test on 3 sample apps")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        apps = parse_apps_file()
        run_research(apps, resume=args.resume)

if __name__ == "__main__":
    main()
