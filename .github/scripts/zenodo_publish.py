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

"""Publish a new version to Zenodo via REST API.

This script reads project metadata from pyproject.toml to avoid duplication.
For subsequent releases, it creates new versions from the existing Zenodo record,
inheriting most metadata automatically.
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
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

try:
  with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
    PROJECT_META = pyproject["project"]
    PROJECT = PROJECT_META["name"]
except (KeyError, FileNotFoundError) as e:
  print(f"❌ Error loading project metadata: {e}", file=sys.stderr)
  sys.exit(1)


def new_version_from_record(record_id: str):
  """Create a new draft that inherits metadata from the latest published record."""
  r = requests.post(
      f"{API}/deposit/depositions/{record_id}/actions/newversion",
      headers=HEADERS,
      timeout=30,
  )
  r.raise_for_status()
  # Zenodo returns a link to the draft, not the draft itself
  latest_draft_url = r.json()["links"]["latest_draft"]
  return requests.get(latest_draft_url, headers=HEADERS, timeout=30).json()


def upload_file(bucket_url: str, path: str, dest_name: str = None):
  """Upload a file to the deposition bucket."""
  dest = dest_name or os.path.basename(path)
  with open(path, "rb") as fp:
    r = requests.put(
        f"{bucket_url}/{dest}",
        data=fp,
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=60,
    )
    r.raise_for_status()


def main():
  """Main workflow."""
  try:
    draft = new_version_from_record(RECORD_ID)

    bucket = draft["links"]["bucket"]
    dep_id = draft["id"]

    # GitHub auto-generates source archives for tags
    tarball = f"/tmp/{PROJECT}-v{VERSION}.tar.gz"
    src_url = f"{SERVER}/{REPO}/archive/refs/tags/v{VERSION}.tar.gz"
    urllib.request.urlretrieve(src_url, tarball)
    upload_file(bucket, tarball, f"{PROJECT}-{VERSION}.tar.gz")

    for path in glob.glob("dist/*"):
      upload_file(bucket, path)

    # Update only version-specific metadata; rest is inherited
    meta = {
        "metadata": {
            "title": f"{PROJECT.replace('-', ' ').title()} v{VERSION}",
            "version": VERSION,
            "upload_type": "software",
        }
    }
    r = requests.put(
        f"{API}/deposit/depositions/{dep_id}",
        headers=HEADERS,
        json=meta,
        timeout=30,
    )
    r.raise_for_status()

    # Publish to mint DOI
    r = requests.post(
        f"{API}/deposit/depositions/{dep_id}/actions/publish",
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    record = r.json()

    doi = record.get("doi")
    record_id = record.get("record_id")

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
