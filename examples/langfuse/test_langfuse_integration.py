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

"""Runnable example to validate optional Langfuse observability."""

import os
import pathlib
import sys
import textwrap

try:
  import dotenv
except ImportError:  # pragma: no cover - optional helper dependency
  dotenv = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import langextract as lx

# Was used for internal testing, feel free to change to your own model.
DEFAULT_MODEL_ID = "llama3"
DEFAULT_MODEL_URL = "http://localhost:11434"
DEFAULT_DOCUMENT_ID = "invoice_test_doc_001.pdf"

PROMPT = textwrap.dedent("""
    You are an invoice extraction assistant.
    Extract the following fields exactly from the text:
    - Invoice Number
    - Invoice Date
    - Sender Name
    - Receiver Name
    - Gross Amount
""").strip()

EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "Invoice Number: INV-00000\n"
            "Invoice Date: 2026-01-15\n"
            "Sender Name: Some GmBH\n"
            "Receiver Name: City Clinic\n"
            "Gross Amount: EUR 250.00\n"
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="Invoice Number",
                extraction_text="INV-00000",
            ),
            lx.data.Extraction(
                extraction_class="Invoice Date",
                extraction_text="2026-01-15",
            ),
            lx.data.Extraction(
                extraction_class="Sender Name",
                extraction_text="Some GmbH",
            ),
            lx.data.Extraction(
                extraction_class="Receiver Name",
                extraction_text="Some Clinic",
            ),
            lx.data.Extraction(
                extraction_class="Gross Amount",
                extraction_text="EUR 250.00",
                attributes={"currency": "EUR"},
            ),
        ],
    )
]

TEST_TEXT = textwrap.dedent("""
    Invoice Number: INV-00000
    Invoice Date: 2026-02-10
    Sender Name: Some Company Ltd
    Receiver Name: Some Company Hospital
    Gross Amount: EUR 1,650.00
    Service Period: 2026-01-01 to 2026-01-31
""").strip()


def _is_truthy(raw_value: str) -> bool:
  """Parses environment booleans."""
  return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _create_observer() -> tuple[object, str]:
  """Creates optional observer based on environment configuration."""
  enable_langfuse = _is_truthy(os.getenv("ENABLE_LANGFUSE", "true"))
  provider = "langfuse" if enable_langfuse else "none"

  observer = lx.observability.create_observer(
      provider=provider,
      provider_kwargs={
          "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
          "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
          "base_url": os.getenv("LANGFUSE_BASE_URL"),
          "host": os.getenv("LANGFUSE_HOST"),
      },
  )
  return observer, provider


def main() -> int:
  """Runs extraction and prints observable status."""
  if dotenv is not None:
    dotenv.load_dotenv(override=True)

  model_id = (
      os.getenv("LANGEXTRACT_TEST_MODEL_ID")
      or os.getenv("OLLAMA_MODEL")
      or DEFAULT_MODEL_ID
  )
  model_url = os.getenv("LANGEXTRACT_TEST_MODEL_URL", DEFAULT_MODEL_URL)
  document_id = os.getenv("LANGEXTRACT_TEST_DOCUMENT_ID", DEFAULT_DOCUMENT_ID)
  observer, provider_name = _create_observer()

  print(
      "Observer provider:",
      provider_name,
      "| enabled:",
      getattr(observer, "enabled", False),
  )
  print("Document ID for trace grouping:", document_id)
  print("Model:", model_id, "| URL:", model_url)

  document = lx.data.Document(text=TEST_TEXT, document_id=document_id)
  config = lx.factory.ModelConfig(
      model_id=model_id,
      provider_kwargs={
          "model_url": model_url,
          "base_url": model_url,
      },
  )
  model = lx.factory.create_model(config, observer=observer)

  try:
    annotated_docs = lx.extract(
        text_or_documents=[document],
        prompt_description=PROMPT,
        examples=EXAMPLES,
        model=model,
        extraction_passes=1,
        fence_output=False,
        use_schema_constraints=False,
        show_progress=False,
    )
  except Exception as e:
    print("\n[ERROR] Extraction failed:", e)
    print("Hints:")
    print("  - Ensure your model endpoint is reachable")
    print("  - Ensure model_id is available on that endpoint")
    print(
        '  - Install optional deps with: pip install -e "langextract[langfuse]"'
    )
    return 1

  if not annotated_docs:
    print("\n[ERROR] No annotated documents returned.")
    return 1

  annotated_doc = annotated_docs[0]
  extractions = annotated_doc.extractions or []

  print("\n[OK] Extraction succeeded.")
  print("Returned document_id:", annotated_doc.document_id)
  print("Total extractions:", len(extractions))
  for extraction in extractions:
    print(f"- {extraction.extraction_class}: {extraction.extraction_text}")

  if getattr(observer, "enabled", False):
    print("\n[OK] Langfuse observer is active.")
    print(
        "Generation traces include document_id metadata and use "
        "session_id=document_id."
    )
    print("Token usage is logged when your model provider returns usage stats.")
  else:
    print("\n[INFO] Langfuse observer is disabled (expected when optional).")
    print("Extraction still runs without observability.")
    print(
        "Set LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY and "
        "ENABLE_LANGFUSE=true to enable traces."
    )

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
