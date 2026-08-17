import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def join_path(*paths) -> str:
    return os.path.join(*paths)

OPENAI_API_KEY = get_env("OPENAI_API_KEY")
TAVILY_API_KEY = get_env("TAVILY_API_KEY")
COMPOSIO_API_KEY = get_env("COMPOSIO_API_KEY")
MODEL_NAME = get_env("MODEL_NAME", "gpt-4o-mini")
BATCH_SIZE = int(get_env("BATCH_SIZE", "10"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = join_path(BASE_DIR, "data")
TEMPLATES_DIR = join_path(BASE_DIR, "templates")

APPS_FILE = join_path(DATA_DIR, "100_apps.json")
RESULTS_FILE = join_path(DATA_DIR, "results.json")
VERIFIED_FILE = join_path(DATA_DIR, "verified_results.json")
PATTERNS_FILE = join_path(DATA_DIR, "patterns.json")
CHECKPOINT_FILE = join_path(DATA_DIR, "checkpoint.json")
OUTPUT_HTML = join_path(BASE_DIR, "report.html")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

@dataclass
class AppEntry:
    id: int
    name: str
    website: str
    category: str

    def to_dict(self):
        return asdict(self)

@dataclass
class AppResearchResult:
    id: int
    name: str
    website: str
    category: str
    description: str = ""
    auth_methods: List[str] = field(default_factory=list)
    self_serve: str = ""
    api_type: str = ""
    api_breadth: str = ""
    has_mcp: bool = False
    mcp_details: str = ""
    existing_composio: bool = False
    buildability_verdict: str = ""
    main_blocker: str = ""
    evidence_url: str = ""
    raw_search_notes: str = ""

    def to_dict(self):
        return asdict(self)

CATEGORIES = [
    "CRM and Sales",
    "Support and Helpdesk",
    "Communications and Messaging",
    "Marketing, Ads, Email and Social",
    "Ecommerce",
    "Data, SEO and Scraping",
    "Developer, Infra and Data platforms",
    "Productivity and Project Management",
    "Finance and Fintech",
    "AI, Research and Media-native",
]

def parse_apps_file(filepath: str = APPS_FILE) -> List[AppEntry]:
    import json
    apps = []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            app = AppEntry(
                id=int(item["id"]),
                name=item["name"],
                website=item["website"],
                category=item["category"],
            )
            apps.append(app)
    return apps

def validate_config() -> bool:
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")

    if missing:
        print(f"Missing required API keys: {', '.join(missing)}")
        print("   Create a .env file and fill in your API keys.")
        return False
    return True
