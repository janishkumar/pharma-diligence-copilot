import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"


def load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


cfg = load_config()


def get(key: str, default=None):
    """Dot-separated key lookup into cfg, e.g. 'generation.model'."""
    parts = key.split(".")
    node = cfg
    for part in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(part, default)
    return node


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "0") == "1"
COST_BUDGET_USD = float(os.getenv("COST_BUDGET_USD", str(cfg["budgets"]["cost_budget_usd_per_session"])))
PROJECT_ROOT = _ROOT
