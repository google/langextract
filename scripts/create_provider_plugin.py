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

"""Create a new LangExtract provider plugin with all boilerplate code.

This script automates steps 1-6 of the provider creation checklist:
1. Setup Package Structure
2. Configure Entry Point
3. Implement Provider
4. Add Schema Support (optional)
5. Create and run tests
6. Generate documentation

For detailed documentation, see:
https://github.com/google/langextract/blob/main/langextract/providers/README.md

Usage:
    python create_provider_plugin.py MyProvider
    python create_provider_plugin.py MyProvider --with-schema
    python create_provider_plugin.py MyProvider --patterns "^mymodel" "^custom"
"""

import argparse
import errno
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap

_PACKAGE_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")

# Values pip's own option parser treats as true for boolean env vars.
_PIP_TRUE_VALUES = frozenset(("y", "yes", "t", "true", "on", "1"))

# Minimum setuptools for the generated project, kept in sync with the
# generated [build-system] requires: PEP 660 editable hooks arrived in
# setuptools 64.0.0 and SPDX `project.license` strings plus
# `project.license-files` in 77.0.0.
_MIN_SETUPTOOLS = (77, 0)
_MIN_SETUPTOOLS_STR = "77.0"
_SETUPTOOLS_REQUIREMENT = f"setuptools>={_MIN_SETUPTOOLS_STR}"

_APACHE_LICENSE_TEXT = """\
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


def _replace_placeholder_line(
    content: str, placeholder: str, value: str
) -> str:
  """Replaces a placeholder line, failing loudly if it is missing.

  Args:
    content: Text containing exactly one placeholder line.
    placeholder: The placeholder token occupying its own line.
    value: Replacement text ("" removes the line entirely).

  Returns:
    The content with the placeholder line replaced.

  Raises:
    ValueError: If the placeholder does not occur exactly once.
  """
  token = placeholder + "\n"
  if content.count(token) != 1:
    raise ValueError(
        f"Expected exactly one '{placeholder}' line in template, found"
        f" {content.count(token)}."
    )
  return content.replace(token, value, 1)


def _ensure_safe_output_path(path: Path) -> None:
  """Refuses a generated-file destination that is a symlink."""
  if path.is_symlink():
    _refuse_symlink_destination(path)


def _refuse_symlink_destination(path: Path) -> None:
  """Reports and rejects a generated-file symlink destination."""
  print(f"ERROR: refusing to overwrite symlink destination: {path}")
  print(
      "Remove that link and rerun the generator; its target was not modified."
  )
  raise SystemExit(1)


def _write_generated_file(path: Path, content: str) -> None:
  """Writes generated text without following a destination symlink."""
  _ensure_safe_output_path(path)
  no_follow = getattr(os, "O_NOFOLLOW", 0)
  if not no_follow:
    _replace_generated_file(path, content)
    return
  flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | no_follow
  try:
    file_descriptor = os.open(path, flags, 0o666)
  except OSError as error:
    if no_follow and error.errno == errno.ELOOP:
      _refuse_symlink_destination(path)
    raise
  with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
    output_file.write(content)


def _replace_generated_file(path: Path, content: str) -> None:
  """Atomically replaces a generated file on systems without O_NOFOLLOW."""
  file_descriptor, temporary_name = tempfile.mkstemp(
      dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
  )
  temporary_path = Path(temporary_name)
  try:
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
      output_file.write(content)
    _ensure_safe_output_path(path)
    os.replace(temporary_path, path)
  finally:
    temporary_path.unlink(missing_ok=True)


def _docstring_safe(text: str) -> str:
  """Escapes text for safe embedding inside a generated docstring."""
  return text.replace("\\", "\\\\").replace('"', '\\"')


def _derive_sample_model_id(pattern: str) -> str | None:
  """Derives a model ID genuinely matching pattern, or None if not derivable.

  Takes the longest literal prefix of the pattern (unescaping backslash
  escapes, stopping at the first live regex metacharacter) and returns the
  first candidate built from it that the pattern actually matches.

  Args:
    pattern: The model-ID regex a sample should match.

  Returns:
    A matching sample model ID, or None when no safe sample exists.
  """
  base = pattern[1:] if pattern.startswith("^") else pattern
  specials = ".^$*+?{}[]()|"
  chars = []
  i = 0
  while i < len(base):
    c = base[i]
    if c == "\\" and i + 1 < len(base) and base[i + 1] in specials + "\\":
      chars.append(base[i + 1])
      i += 2
      continue
    if c == "\\" or c in specials:
      break
    chars.append(c)
    i += 1
  prefix = "".join(chars)
  candidates = (
      f"{prefix}-model",
      f"{prefix}-test",
      f"{prefix}0-test",
      f"{prefix}0",
      prefix,
  )
  for candidate in candidates:
    if candidate and re.search(pattern, candidate):
      return candidate
  return None


def create_directory_structure(package_name: str, force: bool = False) -> Path:
  """Step 1: Setup Package Structure."""
  print("\n" + "=" * 60)
  print("STEP 1: Setup Package Structure")
  print("=" * 60)

  base_dir = Path(f"langextract-{package_name}")
  package_dir = base_dir / f"langextract_{package_name}"

  if base_dir.is_symlink():
    print(f"ERROR: {base_dir} already exists as a symlink.")
    print(
        "Remove that link, or choose a different package name; the link"
        " and its target are never modified (not even with --force)."
    )
    sys.exit(1)

  if base_dir.exists() and not base_dir.is_dir():
    print(f"ERROR: {base_dir} already exists and is not a directory.")
    print(
        "Move or rename that file, or choose a different package name;"
        " it is never deleted or overwritten (not even with --force)."
    )
    sys.exit(1)

  if package_dir.is_symlink():
    print(f"ERROR: {package_dir} already exists as a symlink.")
    print(
        "Remove that link, or choose a different package name; the link"
        " and its target are never modified (not even with --force)."
    )
    sys.exit(1)

  if package_dir.exists() and not package_dir.is_dir():
    print(f"ERROR: {package_dir} already exists and is not a directory.")
    print(
        "Move or rename that file, or choose a different package name;"
        " it is never deleted or overwritten (not even with --force)."
    )
    sys.exit(1)

  if base_dir.exists() and any(base_dir.iterdir()) and not force:
    print(f"ERROR: {base_dir} already exists and is not empty.")
    print("Use --force to overwrite or choose a different package name.")
    sys.exit(1)

  base_dir.mkdir(parents=True, exist_ok=True)
  package_dir.mkdir(parents=True, exist_ok=True)

  print(f"✓ Created directory: {base_dir}/")
  print(f"✓ Created package: {package_dir}/")
  print("✅ Step 1 complete: Package structure created")

  return base_dir


def create_pyproject_toml(
    base_dir: Path, provider_name: str, package_name: str
) -> None:
  """Step 2: Configure Entry Point."""
  print("\n" + "=" * 60)
  print("STEP 2: Configure Entry Point")
  print("=" * 60)

  content = textwrap.dedent(f"""\
        [build-system]
        requires = ["{_SETUPTOOLS_REQUIREMENT}"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "langextract-{package_name}"
        version = "0.1.0"
        description = "LangExtract provider plugin for {provider_name}"
        readme = "README.md"
        requires-python = ">=3.10"
        license = "Apache-2.0"
        license-files = ["LICENSE"]
        dependencies = [
            "langextract>=1.0.0",
            # Add your provider's SDK dependencies here
        ]

        [project.entry-points."langextract.providers"]
        {package_name} = "langextract_{package_name}.provider:{provider_name}LanguageModel"

        [tool.setuptools.packages.find]
        where = ["."]
        include = ["langextract_{package_name}*"]
    """)

  _write_generated_file(base_dir / "pyproject.toml", content)
  print("✓ Created pyproject.toml with entry point configuration")
  print("✅ Step 2 complete: Entry point configured")


def create_provider(
    base_dir: Path,
    provider_name: str,
    package_name: str,
    patterns: list[str],
    with_schema: bool,
) -> None:
  """Step 3: Implement Provider."""
  print("\n" + "=" * 60)
  print("STEP 3: Implement Provider")
  print("=" * 60)

  package_dir = base_dir / f"langextract_{package_name}"

  patterns_str = ", ".join(
      json.dumps(pattern, ensure_ascii=False) for pattern in patterns
  )
  patterns_doc = _docstring_safe(str(patterns))
  env_var_safe = re.sub(r"[^A-Z0-9]+", "_", package_name.upper()) + "_API_KEY"

  schema_imports = (
      f"from langextract_{package_name}.schema import {provider_name}Schema\n\n"
      if with_schema
      else ""
  )
  schema_import_placeholder = "__SCHEMA_IMPORTS__"

  schema_init = (
      """
      self.response_schema = kwargs.get("response_schema")
      self.structured_output = kwargs.get("structured_output", False)"""
      if with_schema
      else ""
  )

  schema_methods = f"""

    @classmethod
    def get_schema_class(cls):
      \"\"\"Tell LangExtract about our schema support.\"\"\"
      return {provider_name}Schema

    def apply_schema(self, schema_instance):
      \"\"\"Apply or clear schema configuration.\"\"\"
      super().apply_schema(schema_instance)
      if schema_instance:
        config = schema_instance.to_provider_config()
        self.response_schema = config.get("response_schema")
        self.structured_output = config.get("structured_output", False)
      else:
        self.response_schema = None
        self.structured_output = False""" if with_schema else ""

  schema_infer = (
      """
        api_params = {}
        if self.response_schema:
          api_params["response_schema"] = self.response_schema
        # result = self.client.generate(prompt, **api_params)"""
      if with_schema
      else """
        # result = self.client.generate(prompt, **kwargs)"""
  )

  provider_content = textwrap.dedent(f'''\
  """Provider implementation for {provider_name}."""

  import json
  import os

  {schema_import_placeholder}
  import langextract as lx
  from langextract.core import base_model
  from langextract.core import types


  @lx.providers.registry.register({patterns_str}, priority=10)
  class {provider_name}LanguageModel(base_model.BaseLanguageModel):
    """LangExtract provider for {provider_name}.

    This provider handles model IDs matching: {patterns_doc}
    """

    def __init__(self, model_id: str, api_key: str | None = None, **kwargs):
      """Initialize the {provider_name} provider.

      Args:
          model_id: The model identifier.
          api_key: API key for authentication.
          **kwargs: Additional provider-specific parameters.
      """
      super().__init__()
      self.model_id = model_id
      self.api_key = api_key or os.environ.get("{env_var_safe}"){schema_init}

      # self.client = YourClient(api_key=self.api_key)
      self._extra_kwargs = kwargs{schema_methods}

    def infer(self, batch_prompts, **kwargs):
      """Run inference on a batch of prompts.

      Args:
          batch_prompts: List of prompts to process.
          **kwargs: Additional inference parameters.

      Yields:
          Lists of ScoredOutput objects, one per prompt.
      """
      for prompt in batch_prompts:{schema_infer}
        result = f"Mock response for: {{prompt[:50]}}..."
        if result.strip():
          # Blank output stays blank; anything else carries a
          # parseable payload so lx.extract() can resolve it.
          payload = json.dumps({{"extractions": []}})
          result = f"{{result}}\\n```json\\n{{payload}}\\n```"
        yield [types.ScoredOutput(score=1.0, output=result)]
  ''')
  provider_content = _replace_placeholder_line(
      provider_content, schema_import_placeholder, schema_imports
  )

  _write_generated_file(package_dir / "provider.py", provider_content)
  print("✓ Created provider.py with mock implementation")

  # Create __init__.py
  init_content = textwrap.dedent(f'''\
        """LangExtract provider plugin for {provider_name}."""

        from langextract_{package_name}.provider import {provider_name}LanguageModel

        __all__ = ["{provider_name}LanguageModel"]
        __version__ = "0.1.0"
    ''')

  _write_generated_file(package_dir / "__init__.py", init_content)
  print("✓ Created __init__.py with exports")
  print("✅ Step 3 complete: Provider implementation created")


def create_schema(
    base_dir: Path, provider_name: str, package_name: str
) -> None:
  """Step 4: Add Schema Support."""
  print("\n" + "=" * 60)
  print("STEP 4: Add Schema Support (Optional)")
  print("=" * 60)

  package_dir = base_dir / f"langextract_{package_name}"

  schema_content = textwrap.dedent(f'''\
  """Schema implementation for {provider_name} provider."""

  from langextract.core import schema


  class {provider_name}Schema(schema.BaseSchema):
    """Schema implementation for {provider_name} structured output."""

    def __init__(self, schema_dict: dict):
      """Initialize the schema with a dictionary."""
      self._schema_dict = schema_dict

    @property
    def schema_dict(self) -> dict:
      """The schema dictionary."""
      return self._schema_dict

    @classmethod
    def from_examples(cls, examples_data, attribute_suffix="_attributes"):
      """Build schema from example extractions.

      Args:
          examples_data: Sequence of ExampleData objects.
          attribute_suffix: Suffix for attribute fields.

      Returns:
          A configured {provider_name}Schema instance.
      """
      schema_dict = {{
          "type": "object",
          "properties": {{
              "extractions": {{
                  "type": "array",
                  "items": {{"type": "object"}},
              }},
          }},
          "required": ["extractions"],
      }}

      return cls(schema_dict)

    def to_provider_config(self) -> dict:
      """Convert to provider-specific configuration.

      Returns:
          Dictionary of provider-specific configuration.
      """
      return {{"response_schema": self._schema_dict, "structured_output": True}}

    @property
    def requires_raw_output(self) -> bool:
      """Whether the provider emits raw JSON without fence markers.

      Returns:
          True if the provider outputs syntactically valid JSON
          directly; False if it needs fence markers for structure.
      """
      return False  # Set to True only if your provider emits raw JSON
  ''')

  _write_generated_file(package_dir / "schema.py", schema_content)
  print("✓ Created schema.py with BaseSchema implementation")
  print("✅ Step 4 complete: Schema support added")


def create_test_script(
    base_dir: Path,
    provider_name: str,
    package_name: str,
    patterns: list[str],
    with_schema: bool,
) -> None:
  """Step 5: Create and run tests."""
  print("\n" + "=" * 60)
  print("STEP 5: Create Tests")
  print("=" * 60)

  patterns_literal = json.dumps(patterns, ensure_ascii=False)
  provider_cls_name = f"{provider_name}LanguageModel"

  test_content = textwrap.dedent(f'''\
  #!/usr/bin/env python3
  """Test script for {provider_name} provider (Step 5 checklist)."""

  import importlib.metadata
  import os
  import re
  import sys

  # Test the INSTALLED plugin, never the adjacent source tree: drop this
  # script's directory from the import path so an uninstalled plugin
  # cannot pass by accident. Compare canonical paths so symlink aliases
  # of the directory (e.g. /var vs /private/var) are removed too.
  _SCRIPT_DIR = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
  sys.path[:] = [
      p for p in sys.path if os.path.realpath(p or os.getcwd()) != _SCRIPT_DIR
  ]

  import langextract as lx
  from langextract.providers import registry

  try:
    from langextract_{package_name} import {provider_cls_name}
  except ImportError as e:
    # Only a missing top-level package means "not installed";
    # any other ImportError is an installed plugin whose own
    # import chain is broken (missing SDK, renamed class, ...).
    if isinstance(e, ModuleNotFoundError) and e.name == "langextract_{package_name}":
      print("ERROR: Plugin not installed. Run: pip install -e .")
    else:
      print(f"ERROR: Plugin is installed but failed to import: {{e!r}}")
    sys.exit(1)

  lx.providers.load_plugins_once()

  PROVIDER_CLS_NAME = "{provider_cls_name}"
  PATTERNS = {patterns_literal}
  ENTRY_POINT_NAME = "{package_name}"
  ENTRY_POINT_VALUE = "langextract_{package_name}.provider:{provider_cls_name}"

  FAILURES = 0


  def _fail(message: str) -> None:
    \"\"\"Record a failed check and print its message.\"\"\"
    global FAILURES
    FAILURES += 1
    print(message)


  def _example_id(pattern: str) -> str | None:
    \"\"\"Derive a model ID matching pattern, or None if not derivable.\"\"\"
    base = pattern[1:] if pattern.startswith("^") else pattern
    specials = ".^$*+?{{}}[]()|"
    chars = []
    i = 0
    while i < len(base):
      c = base[i]
      if c == "\\\\" and i + 1 < len(base) and base[i + 1] in specials + "\\\\":
        chars.append(base[i + 1])
        i += 2
        continue
      if c == "\\\\" or c in specials:
        break
      chars.append(c)
      i += 1
    prefix = "".join(chars)
    candidates = (f"{{prefix}}-test", f"{{prefix}}0-test", f"{{prefix}}0", prefix)
    for candidate in candidates:
      if candidate and re.search(pattern, candidate):
        return candidate
    return None


  sample_ids = []
  skipped_patterns = []
  for p in PATTERNS:
    sample_id = _example_id(p)
    if sample_id is None:
      skipped_patterns.append(p)
    else:
      sample_ids.append(sample_id)
  sample_ids.append("unknown-model")

  print("Testing {provider_name} Provider - Step 5 Checklist:")
  print("-" * 50)

  # 1. Entry point registered in the INSTALLED package metadata
  print("1. Installed entry point (importlib.metadata)")
  entry_points = [
      ep
      for ep in importlib.metadata.entry_points(group="langextract.providers")
      if ep.name == ENTRY_POINT_NAME
  ]
  if not entry_points:
    _fail(
        f"   ✗ No '{{ENTRY_POINT_NAME}}' entry point in installed metadata. Run:"
        " pip install -e ."
    )
  elif not any(ep.value == ENTRY_POINT_VALUE for ep in entry_points):
    _fail(
        f"   ✗ Entry point mismatch: expected {{ENTRY_POINT_VALUE}}, found"
        f" {{[ep.value for ep in entry_points]}}"
    )
  else:
    print(f"   ✓ {{ENTRY_POINT_NAME}} = {{ENTRY_POINT_VALUE}}")

  # 2. Provider registration + pattern matching via resolve()
  print("\\n2. Provider registration & pattern matching")
  for p in skipped_patterns:
    print(
        f"   ⚠ {{p!r}}: no sample model ID derivable; add a manual resolve() check"
        " for this pattern"
    )
  for model_id in sample_ids:
    try:
      provider_class = registry.resolve(model_id)
    except Exception as e:
      if model_id == "unknown-model":
        print(f"   ✓ {{model_id}}: No provider found (expected)")
      else:
        _fail(f"   ✗ {{model_id}}: resolve() failed: {{e}}")
      continue
    if provider_class.__name__ == PROVIDER_CLS_NAME:
      print(f"   ✓ {{model_id}} -> {{provider_class.__name__}} expected")
    elif model_id == "unknown-model":
      print(
          f"   ✓ {{model_id}} -> {{provider_class.__name__}} (another provider;"
          " acceptable)"
      )
    else:
      _fail(f"   ✗ {{model_id}} -> {{provider_class.__name__}} unexpected provider")

  # 3. Inference sanity check
  print("\\n3. Test inference with sample prompts")
  try:
    model_id = sample_ids[0] if sample_ids[0] != "unknown-model" else "test-model"
    provider = {provider_cls_name}(model_id=model_id)
    prompts = ["Test prompt 1", "Test prompt 2"]
    results = list(provider.infer(prompts))
    if len(results) != len(prompts):
      _fail(f"   ✗ Expected {{len(prompts)}} results, got {{len(results)}}")
    else:
      print(f"   ✓ Inference returned {{len(results)}} results")
    for i, result in enumerate(results):
      try:
        out = result[0].output if result and result[0] else None
      except Exception:
        out = None
      if out and out.strip():
        print(f"   ✓ Result {{i+1}}: {{out[:60]}}...")
      else:
        _fail(f"   ✗ Result {{i+1}}: missing or empty output: {{result!r}}")
  except Exception as e:
    _fail(f"   ✗ ERROR: {{e}}")
  ''')

  if with_schema:
    test_content += textwrap.dedent(f"""
    # 4. Test schema creation and application
    print("\\n4. Test schema creation and application")
    try:
      from langextract_{package_name}.schema import {provider_name}Schema

      from langextract import data

      examples = [
          data.ExampleData(
              text="Test text",
              extractions=[
                  data.Extraction(
                      extraction_class="entity",
                      extraction_text="test",
                      attributes={{"type": "example"}},
                  )
              ],
          )
      ]

      schema = {provider_name}Schema.from_examples(examples)
      print(f"   ✓ Schema created (keys={{list(schema.schema_dict.keys())}})")
      print(f"   ✓ requires_raw_output = {{schema.requires_raw_output}}")

      schema_class = {provider_cls_name}.get_schema_class()
      print(f"   ✓ Provider schema class: {{schema_class.__name__}}")

      provider = {provider_cls_name}(
          model_id=sample_ids[0]
          if sample_ids[0] != "unknown-model"
          else "test-model"
      )
      provider.apply_schema(schema)
      print(
          "   ✓ Schema applied:"
          f" response_schema={{provider.response_schema is not None}}"
          f" structured={{getattr(provider, 'structured_output', False)}}"
      )
    except Exception as e:
      _fail(f"   ✗ ERROR: {{e}}")
    """)

  factory_step = 5 if with_schema else 4
  test_content += textwrap.dedent(f"""
    # {factory_step}. Test factory integration
    print("\\n{factory_step}. Test factory integration")
    try:
      from langextract import factory

      config = factory.ModelConfig(
          model_id=sample_ids[0]
          if sample_ids[0] != "unknown-model"
          else "test-model",
          provider="{provider_cls_name}",
      )
      model = factory.create_model(config)
      print(f"   ✓ Factory created: {{type(model).__name__}}")
    except Exception as e:
      _fail(f"   ✗ ERROR: {{e}}")

    print("\\n" + "-" * 50)
    if FAILURES:
      print(f"✗ {{FAILURES}} check(s) failed.")
      sys.exit(1)
    if skipped_patterns:
      print(
          f"⚠ {{len(skipped_patterns)}} pattern(s) had no derivable"
          f" sample model ID and were not verified: {{skipped_patterns}}"
      )
      sys.exit(1)
    print("✅ All checks passed!")
    """)

  _write_generated_file(base_dir / "test_plugin.py", test_content)
  print("✓ Created test_plugin.py with comprehensive tests")
  print("✅ Step 5 complete: Test suite created")


def create_readme(
    base_dir: Path, provider_name: str, package_name: str, patterns: list[str]
) -> None:
  """Create README documentation."""
  print("\n" + "=" * 60)
  print("STEP 6: Documentation")
  print("=" * 60)

  def _display(p: str) -> str:
    """Strip leading ^ from pattern for display."""
    return p[1:] if p.startswith("^") else p

  env_var_safe = re.sub(r"[^A-Z0-9]+", "_", package_name.upper()) + "_API_KEY"

  supported = "\n".join(
      f"- `{_display(p)}*`: Models matching pattern {p}" for p in patterns
  )
  supported_models_placeholder = "__SUPPORTED_MODELS__"
  usage_placeholder = "__USAGE_EXAMPLE__"
  sample_model_id = (
      _derive_sample_model_id(patterns[0])
      if patterns
      else f"{package_name}-model"
  )
  if sample_model_id is None:
    usage_block = textwrap.dedent(f"""\
        No runnable example is generated for this provider: no sample model
        ID could be safely derived from pattern `{patterns[0]}`. Call
        `lx.extract` with `text_or_documents=...` and a `model_id` that
        your provider's patterns match.
    """)
  else:
    usage_block = textwrap.dedent(f"""\
        ```python
        import langextract as lx

        result = lx.extract(
            text_or_documents="Your document here",
            model_id={sample_model_id!r},
            prompt_description="Extract entities",
            examples=[...]
        )
        ```
    """)

  readme_content = textwrap.dedent(f"""\
        # LangExtract {provider_name} Provider

        A provider plugin for LangExtract that supports {provider_name} models.

        ## Installation

        ```bash
        pip install -e .
        ```

        ## Supported Model IDs

        {supported_models_placeholder}

        ## Environment Variables

        - `{env_var_safe}`: API key for authentication

        ## Usage

        {usage_placeholder}

        ## Development

        1. Install in development mode: `pip install -e .`
        2. Run tests: `python test_plugin.py`
        3. Build package: `python -m build`
        4. Publish to PyPI: `twine upload dist/*`

        ## License

        Apache License 2.0
    """)
  readme_content = _replace_placeholder_line(
      readme_content, supported_models_placeholder, f"{supported}\n"
  )
  readme_content = _replace_placeholder_line(
      readme_content, usage_placeholder, usage_block
  )

  _write_generated_file(base_dir / "README.md", readme_content)
  print("✓ Created README.md with usage examples")


def create_gitignore(base_dir: Path) -> None:
  """Create .gitignore file with Python-specific entries."""
  gitignore_content = textwrap.dedent("""\
        # Python
        __pycache__/
        *.py[cod]
        *$py.class
        *.so

        # Distribution / packaging
        build/
        dist/
        *.egg-info/
        .eggs/
        *.egg

        # Virtual environments
        .env
        .venv
        env/
        venv/
        ENV/

        # Testing & coverage
        .pytest_cache/
        .tox/
        htmlcov/
        .coverage
        .coverage.*

        # Type checking
        .mypy_cache/
        .dmypy.json
        dmypy.json
        .pytype/

        # IDEs
        .idea/
        .vscode/
        *.swp
        *.swo

        # OS-specific
        .DS_Store
        Thumbs.db

        # Logs
        *.log

        # Temp files
        *.tmp
        *.bak
        *.backup
    """)

  _write_generated_file(base_dir / ".gitignore", gitignore_content)
  print("✓ Created .gitignore file with Python-specific entries")


def create_license(base_dir: Path) -> None:
  """Create LICENSE file matching the pyproject-declared Apache-2.0."""
  _write_generated_file(base_dir / "LICENSE", _APACHE_LICENSE_TEXT)
  print("✓ Created LICENSE file (Apache-2.0, matching pyproject.toml)")
  print("✅ Step 6 complete: Documentation created")


def _retire_stale_schema(base_dir: Path, package_name: str) -> None:
  """Moves a stale schema.py to a non-importable, recoverable backup.

  A --force regeneration from schema mode to non-schema mode must not
  leave the old schema.py importable or packageable. Existing backups
  are never overwritten.

  Args:
    base_dir: Root directory of the generated plugin.
    package_name: Package whose module directory is checked.
  """
  schema_path = base_dir / f"langextract_{package_name}" / "schema.py"
  if not os.path.lexists(schema_path):
    return
  backup = schema_path.with_suffix(".py.bak")
  counter = 1
  while os.path.lexists(backup):
    # Keep the .bak extension on collisions so the generated
    # .gitignore's *.bak rule covers every backup.
    backup = schema_path.with_suffix(f".py.{counter}.bak")
    counter += 1
  schema_path.rename(backup)
  print(f"✓ Moved stale schema.py to {backup.name} (delete or restore it)")


def _find_spec_or_none(name: str):
  """Returns importlib.util.find_spec(name), treating errors as absent.

  Probing a submodule imports its parent package, so a broken setuptools
  installation can raise almost anything here; any failure means the
  backend is not usable locally.
  """
  try:
    return importlib.util.find_spec(name)
  except Exception:
    return None


def _parse_version_tuple(raw_version: str) -> tuple[int, int] | None:
  """Parses the leading major.minor integers of a version string."""
  match = re.match(r"\s*(\d+)(?:\.(\d+))?", raw_version)
  if not match:
    return None
  return int(match.group(1)), int(match.group(2) or 0)


def _local_backend_problem() -> str | None:
  """Returns why the local backend cannot build this project, or None.

  Usable means: setuptools is installed, its PEP 517 backend module
  imports, and its version meets the generated [build-system] minimum
  (PEP 660 editable hooks need setuptools>=64, the generated SPDX
  license metadata needs >=77) — mere importability is not enough:
  older setuptools fails such builds with raw backend tracebacks. The
  version comes from installed distribution metadata, so no
  version-comparison dependency is needed; any probe failure counts as
  an unusable backend.
  """
  if _find_spec_or_none("setuptools") is None:
    return "setuptools is not installed"
  if _find_spec_or_none("setuptools.build_meta") is None:
    return "the installed setuptools cannot import its build backend"
  try:
    raw_version = importlib.metadata.version("setuptools")
  except Exception:
    return "the installed setuptools version could not be determined"
  version = _parse_version_tuple(raw_version)
  if version is None:
    return f"the installed setuptools version {raw_version!r} is not recognized"
  if version < _MIN_SETUPTOOLS:
    return (
        f"the installed setuptools {raw_version} is older than the"
        f" {_MIN_SETUPTOOLS_STR} this project's build needs"
    )
  return None


def _pip_config_no_index_enabled() -> bool:
  """Returns whether pip config enables no-index mode."""
  try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "config", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
  except OSError:
    return False
  if result.returncode:
    return False

  no_index_keys = frozenset(("global.no-index", "install.no-index"))
  for line in result.stdout.splitlines():
    key, separator, value = line.partition("=")
    if separator and key.strip() in no_index_keys:
      normalized = value.strip().strip("'\"").lower()
      if normalized in _PIP_TRUE_VALUES:
        return True
  return False


def _pip_no_index_enabled() -> bool:
  """Returns whether pip is configured to avoid package indexes."""
  if "PIP_NO_INDEX" in os.environ:
    value = os.environ["PIP_NO_INDEX"]
    return value.strip().lower() in _PIP_TRUE_VALUES
  return _pip_config_no_index_enabled()


def install_and_test(base_dir: Path) -> bool:
  """Install the plugin and run tests."""
  print("\n" + "=" * 60)
  print("Installing and testing the plugin...")
  print("=" * 60)

  if _find_spec_or_none("langextract") is None:
    print("\nERROR: LangExtract is not installed in this Python environment.")
    print(
        "Install it first (pip install langextract), then rerun the generator."
    )
    return False

  # Runtime dependencies always come from the already-provisioned dev
  # environment, never from the install itself.
  install_command = [
      sys.executable,
      "-m",
      "pip",
      "install",
      "-e",
      ".",
      "--no-deps",
  ]
  backend_problem = _local_backend_problem()
  if backend_problem is None:
    # Offline-safe editable install: the local setuptools builds the
    # package, so pip never downloads a build environment.
    install_command.append("--no-build-isolation")
  elif _pip_no_index_enabled():
    print(
        f"\nERROR: cannot install the plugin — {backend_problem}, and"
        " pip's no-index mode forbids downloading a build environment."
    )
    print(
        "Install the build prerequisite first (pip install"
        f" {shlex.quote(_SETUPTOOLS_REQUIREMENT)}), or disable no-index mode"
        " so pip can provision an isolated build environment."
    )
    return False
  # Otherwise the local backend is unusable (fresh Python 3.12+
  # environments ship no setuptools at all) but pip has index access:
  # pip's default build isolation provisions the declared build backend.

  print("\nInstalling plugin...")
  result = subprocess.run(
      install_command,
      capture_output=True,
      text=True,
      check=False,
      cwd=base_dir,
  )
  if result.returncode:
    print("Installation failed:")
    # pip splits useful diagnostics across both streams; show each
    # nonempty stream once (identical streams are not repeated).
    shown = []
    for stream_text in (result.stdout, result.stderr):
      cleaned = (stream_text or "").strip()
      if cleaned and cleaned not in shown:
        shown.append(cleaned)
        print(cleaned)
    return False
  print("✓ Plugin installed successfully")

  print("\nRunning tests...")
  result = subprocess.run(
      [sys.executable, "test_plugin.py"],
      capture_output=True,
      text=True,
      check=False,
      cwd=base_dir,
  )
  print(result.stdout)
  if result.returncode:
    print(f"Tests failed: {result.stderr}")
    return False

  return True


def parse_arguments():
  """Parse command line arguments.

  Returns:
    Parsed arguments from argparse.
  """
  parser = argparse.ArgumentParser(
      description="Create a new LangExtract provider plugin",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=textwrap.dedent("""
        Examples:
            python create_provider_plugin.py MyProvider
            python create_provider_plugin.py MyProvider --with-schema
            python create_provider_plugin.py MyProvider --patterns "^mymodel" "^custom"
            python create_provider_plugin.py MyProvider --package-name my_custom_name
        """),
  )

  parser.add_argument(
      "provider_name",
      help="Name of your provider (e.g., MyProvider, CustomLLM)",
  )

  parser.add_argument(
      "--patterns",
      nargs="+",
      default=None,
      help="Regex patterns for model IDs (default: based on provider name)",
  )

  parser.add_argument(
      "--package-name",
      default=None,
      help="Package name (default: lowercase provider name)",
  )

  parser.add_argument(
      "--with-schema",
      action="store_true",
      help="Include schema support (Step 4)",
  )

  parser.add_argument(
      "--no-install", action="store_true", help="Skip installation and testing"
  )

  parser.add_argument(
      "--force",
      action="store_true",
      help="Overwrite existing plugin directory if it exists",
  )

  return parser.parse_args()


def validate_names(provider_name: str, package_name: str) -> None:
  """Validate provider and package names before anything is written.

  Args:
    provider_name: Name used to build generated class names.
    package_name: Name used to build the distribution/module names.

  Raises:
    SystemExit: If either name would generate broken code.
  """
  if not provider_name.isidentifier():
    print(
        f"ERROR: Invalid provider name {provider_name!r}. Use a valid Python"
        " identifier (letters, digits, underscores; no spaces, hyphens,"
        " quotes, or leading digit), e.g. MyProvider."
    )
    sys.exit(1)
  if not _PACKAGE_NAME_RE.fullmatch(package_name):
    print(
        f"ERROR: Invalid package name {package_name!r}. Use a lowercase name"
        " starting with a letter, containing only letters, digits, and"
        " underscores, e.g. my_provider."
    )
    sys.exit(1)


def validate_patterns(patterns: list[str]) -> None:
  """Validate regex patterns.

  Args:
    patterns: List of regex patterns to validate.

  Raises:
    SystemExit: If any pattern is invalid.
  """
  for p in patterns:
    try:
      re.compile(p)
    except re.error as e:
      print(f"ERROR: Invalid regex pattern '{p}': {e}")
      sys.exit(1)


def print_summary(
    provider_name: str,
    package_name: str,
    patterns: list[str],
    with_schema: bool,
) -> None:
  """Print configuration summary.

  Args:
    provider_name: Name of the provider.
    package_name: Package name.
    patterns: List of model ID patterns.
    with_schema: Whether to include schema support.
  """
  print("\n" + "=" * 60)
  print("LANGEXTRACT PROVIDER PLUGIN GENERATOR")
  print("=" * 60)
  print(f"Provider Name: {provider_name}")
  print(f"Package Name: langextract-{package_name}")
  print(f"Model Patterns: {patterns}")
  print(f"Include Schema: {with_schema}")
  print("\nFor documentation, see:")
  print(
      "https://github.com/google/langextract/blob/main/langextract/providers/README.md"
  )


def _validate_generated_output_paths(
    base_dir: Path, package_name: str, with_schema: bool
) -> None:
  """Refuses file symlinks before regeneration modifies any files."""
  package_dir = base_dir / f"langextract_{package_name}"
  output_paths = [
      base_dir / "pyproject.toml",
      package_dir / "provider.py",
      package_dir / "__init__.py",
      base_dir / "test_plugin.py",
      base_dir / "README.md",
      base_dir / ".gitignore",
      base_dir / "LICENSE",
  ]
  if with_schema:
    output_paths.append(package_dir / "schema.py")
  for path in output_paths:
    _ensure_safe_output_path(path)


def create_plugin(
    args: argparse.Namespace, package_name: str, patterns: list[str]
) -> Path:
  """Create the plugin with all necessary files.

  Args:
    args: Parsed command line arguments.
    package_name: Package name.
    patterns: List of model ID patterns.

  Returns:
    Path to the created plugin directory.
  """
  base_dir = create_directory_structure(package_name, force=args.force)
  _validate_generated_output_paths(base_dir, package_name, args.with_schema)
  create_pyproject_toml(base_dir, args.provider_name, package_name)
  create_provider(
      base_dir, args.provider_name, package_name, patterns, args.with_schema
  )

  if args.with_schema:
    create_schema(base_dir, args.provider_name, package_name)
  else:
    _retire_stale_schema(base_dir, package_name)

  create_test_script(
      base_dir, args.provider_name, package_name, patterns, args.with_schema
  )
  create_readme(base_dir, args.provider_name, package_name, patterns)
  create_gitignore(base_dir)
  create_license(base_dir)

  return base_dir


def print_completion_summary(with_schema: bool) -> None:
  """Print completion summary.

  Args:
    with_schema: Whether schema support was included.
  """
  print("\n" + "=" * 60)
  print("SUMMARY: Steps 1-6 Completed")
  print("=" * 60)
  print("✅ Package structure created")
  print("✅ Entry point configured")
  print("✅ Provider implemented")
  if with_schema:
    print("✅ Schema support added")
  print("✅ Tests created")
  print("✅ Documentation generated")


def main():
  """Main entry point for the provider plugin generator."""
  args = parse_arguments()

  package_name = args.package_name or args.provider_name.lower()
  patterns = args.patterns if args.patterns else [f"^{package_name}"]

  validate_names(args.provider_name, package_name)
  validate_patterns(patterns)
  print_summary(args.provider_name, package_name, patterns, args.with_schema)

  base_dir = create_plugin(args, package_name, patterns)
  print_completion_summary(args.with_schema)

  if not args.no_install:
    success = install_and_test(base_dir)
    if success:
      print("\n✅ Plugin created, installed, and tested successfully!")
      print(f"\nYour plugin is ready at: {base_dir.absolute()}")
      print("\nNext steps:")
      print("  1. Replace mock inference with actual API calls")
      print("  2. Update documentation with real examples")
      print("  3. Build package: python -m build")
      print("  4. Publish to PyPI: twine upload dist/*")
    else:
      print(
          "\n⚠️ Plugin created, but automatic setup did not complete."
          " See the error above."
      )
      sys.exit(1)
  else:
    print(f"\nPlugin created at: {base_dir.absolute()}")
    print("\nTo install and test:")
    print(f"  cd {base_dir}")
    print("  pip install -e .")
    print("  python test_plugin.py")


if __name__ == "__main__":
  main()
