#!/usr/bin/env python3
"""Environment validation — run before any other script."""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
REQUIRED_PYTHON = (3, 11)
MIN_DISK_GB = 50
ENV_FILE = ROOT / ".env"

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
WARN = "\033[93m  WARN\033[0m"
INFO = "\033[94m  INFO\033[0m"

failures = []


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False):
    tag = PASS if ok else (WARN if warn_only else FAIL)
    print(f"{tag}  {label}")
    if detail:
        print(f"       {detail}")
    if not ok and not warn_only:
        failures.append(label)


print("\n=== Pharma Diligence Copilot — Environment Check ===\n")

# Python version
ver = sys.version_info
check(
    f"Python >= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}",
    ver >= REQUIRED_PYTHON,
    f"Found {ver.major}.{ver.minor}.{ver.micro}",
)

# Disk space
total, used, free = shutil.disk_usage(ROOT)
free_gb = free / 1e9
check(
    f"Disk space >= {MIN_DISK_GB} GB free",
    free_gb >= MIN_DISK_GB,
    f"{free_gb:.1f} GB free at {ROOT}",
    warn_only=free_gb < MIN_DISK_GB,
)

# .env file
env_exists = ENV_FILE.exists()
check(".env file present", env_exists, "Copy .env.example to .env and fill in keys" if not env_exists else "")

# Load env if present
if env_exists:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)

# API keys (mask values)
anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
openai_key = os.getenv("OPENAI_API_KEY", "")
check("ANTHROPIC_API_KEY set", bool(anthropic_key), "Required for Phase 1 generator (D1=Anthropic)")
check("OPENAI_API_KEY set", bool(openai_key), "Required for judge model (D6=OpenAI)")

# Network reachability
import urllib.request

def reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=5)
        return True
    except Exception:
        return False

check("SEC EDGAR reachable", reachable("https://data.sec.gov"), "https://data.sec.gov")
check("HuggingFace reachable", reachable("https://huggingface.co"), "https://huggingface.co")
check("Anthropic API reachable", reachable("https://api.anthropic.com"), "https://api.anthropic.com")

# Core packages
def pkg_ok(name: str) -> bool:
    import importlib
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

for pkg in ["yaml", "dotenv", "pydantic", "structlog", "httpx", "tqdm"]:
    check(f"Package: {pkg}", pkg_ok(pkg), f"pip install {pkg}" if not pkg_ok(pkg) else "")

# pip version
import subprocess
result = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True)
print(f"{INFO}  pip: {result.stdout.strip()}")

print()
if failures:
    print(f"\033[91m{len(failures)} check(s) failed: {', '.join(failures)}\033[0m")
    print("Fix the issues above, then re-run this script.\n")
    sys.exit(1)
else:
    print("\033[92mAll checks passed. Ready to build.\033[0m")
    print("Next step: python scripts/01_ingest.py\n")
