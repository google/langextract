#!/usr/bin/env python3
# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Publish a new version to Zenodo via the InvenioRDM Records API.

Zenodo migrated to InvenioRDM in late 2023. The legacy /api/deposit
endpoints still exist but /actions/newversion now rejects records with
files attached ("Please remove all files first" on field files.enabled).
This script uses the modern /api/records flow instead.

Reads project name from pyproject.toml. ZENODO_RECORD_ID may be either
the concept ID or any version's record ID — Zenodo resolves to the
latest version.
"""

import glob
import os
import sys
import tomllib
import urllib.request

import requests

API = "https://zenodo.org/api"
TOKEN = os.environ["ZENODO_TOKEN"]
RECORD_ID = os.environ["ZENODO_RECORD_ID"]
VERSION = os.environ["RELEASE_TAG"].lstrip("v")
REPO = os.environ["GITHUB_REPOSITORY"]
SERVER = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
AUTH = {"Authorization": f"Bearer {TOKEN}"}
JSON_HEADERS = {**AUTH, "Content-Type": "application/json"}

try:
  with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
    PROJECT_META = pyproject["project"]
    PROJECT = PROJECT_META["name"]
except (KeyError, FileNotFoundError) as e:
  print(f"❌ Error loading project metadata: {e}", file=sys.stderr)
  sys.exit(1)


def _check(r: requests.Response, op: str) -> None:
  """raise_for_status with the response body included for debugging."""
  if not r.ok:
    body = r.text[:2000] if r.text else "<empty>"
    print(f"❌ Zenodo {op} failed: {r.status_code} {r.reason}", file=sys.stderr)
    print(f"   URL: {r.request.url}", file=sys.stderr)
    print(f"   Response body: {body}", file=sys.stderr)
    r.raise_for_status()


def new_version_draft(record_id: str) -> dict:
  """Create a new version draft via the InvenioRDM Records API.

  If a draft already exists for the record, the API returns it instead
  of creating a duplicate.
  """
  r = requests.post(
      f"{API}/records/{record_id}/versions",
      headers=JSON_HEADERS,
      timeout=30,
  )
  _check(r, "create new version")
  return r.json()


def upload_file(draft_id: str, path: str, dest_name: str = None) -> None:
  """Register, upload, and commit a file on a draft (3-step RDM flow)."""
  dest = dest_name or os.path.basename(path)
  files_url = f"{API}/records/{draft_id}/draft/files"

  init = requests.post(
      files_url, headers=JSON_HEADERS, json=[{"key": dest}], timeout=30
  )
  if init.status_code == 400 and "already exists" in init.text.lower():
    # Re-running over an existing draft; remove the stale entry first.
    del_r = requests.delete(f"{files_url}/{dest}", headers=AUTH, timeout=30)
    _check(del_r, f"delete stale file {dest}")
    init = requests.post(
        files_url, headers=JSON_HEADERS, json=[{"key": dest}], timeout=30
    )
  _check(init, f"register file {dest}")

  with open(path, "rb") as fp:
    up = requests.put(
        f"{files_url}/{dest}/content",
        data=fp,
        headers={**AUTH, "Content-Type": "application/octet-stream"},
        timeout=300,
    )
  _check(up, f"upload {dest}")

  commit = requests.post(f"{files_url}/{dest}/commit", headers=AUTH, timeout=30)
  _check(commit, f"commit {dest}")


def update_metadata(draft_id: str) -> None:
  """Patch only the version-specific fields; the rest is inherited."""
  r = requests.get(f"{API}/records/{draft_id}/draft", headers=AUTH, timeout=30)
  _check(r, "GET draft")
  draft = r.json()
  metadata = dict(draft.get("metadata", {}))
  metadata["title"] = f"{PROJECT.replace('-', ' ').title()} v{VERSION}"
  metadata["version"] = VERSION
  # InvenioRDM resource type schema differs from the legacy upload_type
  # enum; "software" is the canonical id.
  metadata["resource_type"] = {"id": "software"}

  r = requests.put(
      f"{API}/records/{draft_id}/draft",
      headers=JSON_HEADERS,
      json={"metadata": metadata},
      timeout=30,
  )
  _check(r, "PUT metadata")


def publish_draft(draft_id: str) -> dict:
  r = requests.post(
      f"{API}/records/{draft_id}/draft/actions/publish",
      headers=JSON_HEADERS,
      timeout=60,
  )
  _check(r, "publish")
  return r.json()


def main() -> int:
  try:
    draft = new_version_draft(RECORD_ID)
    draft_id = draft["id"]
    print(f"ℹ️  Working on draft id={draft_id}", file=sys.stderr)

    tarball = f"/tmp/{PROJECT}-v{VERSION}.tar.gz"
    src_url = f"{SERVER}/{REPO}/archive/refs/tags/v{VERSION}.tar.gz"
    urllib.request.urlretrieve(src_url, tarball)
    upload_file(draft_id, tarball, f"{PROJECT}-{VERSION}.tar.gz")

    for path in glob.glob("dist/*"):
      upload_file(draft_id, path)

    update_metadata(draft_id)
    record = publish_draft(draft_id)

    doi = record.get("doi") or record.get("pids", {}).get("doi", {}).get(
        "identifier"
    )
    record_id = record.get("id")
    print(f"✅ Published to Zenodo: https://doi.org/{doi}")

    if "GITHUB_OUTPUT" in os.environ:
      with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"doi={doi}\n")
        f.write(f"record_id={record_id}\n")
        f.write(f"zenodo_url=https://zenodo.org/records/{record_id}\n")

    return 0

  except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
