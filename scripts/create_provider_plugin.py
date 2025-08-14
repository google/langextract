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
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap


def create_directory_structure(package_name: str, force: bool = False) -> Path:
  """Step 1: Setup Package Structure."""
  print("\n" + "=" * 60)
  print("STEP 1: Setup Package Structure")
  print("=" * 60)

  base_dir = Path(f"langextract-{package_name}")
  package_dir = base_dir / f"langextract_{package_name}"

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
        requires = ["setuptools>=61.0", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "langextract-{package_name}"
        version = "0.1.0"
        description = "LangExtract provider plugin for {provider_name}"
        readme = "README.md"
        requires-python = ">=3.10"
        license = {{text = "Apache-2.0"}}
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

  (base_dir / "pyproject.toml").write_text(content, encoding="utf-8")
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

  patterns_str = ", ".join(f"r'{p}'" for p in patterns)
  env_var_safe = re.sub(r"[^A-Z0-9]+", "_", package_name.upper()) + "_API_KEY"

  schema_imports = (
      f"""
from langextract_{package_name}.schema import {provider_name}Schema"""
      if with_schema
      else ""
  )

  schema_init = (
      """
                self.response_schema = kwargs.get('response_schema')
                self.structured_output = kwargs.get('structured_output', False)"""
      if with_schema
      else ""
  )

  schema_methods = f"""

            @classmethod
            def get_schema_class(cls):
                \"\"\"Tell LangExtract about our schema support.\"\"\"
                from langextract_{package_name}.schema import {provider_name}Schema
                return {provider_name}Schema

            def apply_schema(self, schema_instance):
                \"\"\"Apply or clear schema configuration.\"\"\"
                super().apply_schema(schema_instance)
                if schema_instance:
                    config = schema_instance.to_provider_config()
                    self.response_schema = config.get('response_schema')
                    self.structured_output = config.get('structured_output', False)
                else:
                    self.response_schema = None
                    self.structured_output = False""" if with_schema else ""

  schema_infer = (
      """
                    api_params = {}
                    if self.response_schema:
                        api_params['response_schema'] = self.response_schema
                    # result = self.client.generate(prompt, **api_params)"""
      if with_schema
      else """
                    # result = self.client.generate(prompt, **kwargs)"""
  )

  provider_content = textwrap.dedent(f'''\
        """Provider implementation for {provider_name}."""

        import os
        import langextract as lx{schema_imports}


        @lx.providers.registry.register({patterns_str}, priority=10)
        class {provider_name}LanguageModel(lx.inference.BaseLanguageModel):
            """LangExtract provider for {provider_name}.

            This provider handles model IDs matching: {patterns}
            """

            def __init__(self, model_id: str, api_key: str = None, **kwargs):
                """Initialize the {provider_name} provider.

                Args:
                    model_id: The model identifier.
                    api_key: API key for authentication.
                    **kwargs: Additional provider-specific parameters.
                """
                super().__init__()
                self.model_id = model_id
                self.api_key = api_key or os.environ.get('{env_var_safe}'){schema_init}

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
                    yield [lx.inference.ScoredOutput(score=1.0, output=result)]
    ''')

  (package_dir / "provider.py").write_text(provider_content, encoding="utf-8")
  print("✓ Created provider.py with mock implementation")

  # Create __init__.py
  init_content = textwrap.dedent(f'''\
        """LangExtract provider plugin for {provider_name}."""

        from langextract_{package_name}.provider import {provider_name}LanguageModel

        __all__ = ['{provider_name}LanguageModel']
        __version__ = "0.1.0"
    ''')

  (package_dir / "__init__.py").write_text(init_content, encoding="utf-8")
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

        import langextract as lx
        from langextract import schema


        class {provider_name}Schema(lx.schema.BaseSchema):
            """Schema implementation for {provider_name} structured output."""

            def __init__(self, schema_dict: dict):
                """Initialize the schema with a dictionary."""
                self._schema_dict = schema_dict

            @property
            def schema_dict(self) -> dict:
                """Return the schema dictionary."""
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
                extraction_types = {{}}
                for example in examples_data:
                    for extraction in example.extractions:
                        class_name = extraction.extraction_class
                        if class_name not in extraction_types:
                            extraction_types[class_name] = set()
                        if extraction.attributes:
                            extraction_types[class_name].update(extraction.attributes.keys())

                schema_dict = {{
                    "type": "object",
                    "properties": {{
                        "extractions": {{
                            "type": "array",
                            "items": {{"type": "object"}}
                        }}
                    }},
                    "required": ["extractions"]
                }}

                return cls(schema_dict)

            def to_provider_config(self) -> dict:
                """Convert to provider-specific configuration.

                Returns:
                    Dictionary of provider-specific configuration.
                """
                return {{
                    "response_schema": self._schema_dict,
                    "structured_output": True
                }}

            @property
            def supports_strict_mode(self) -> bool:
                """Whether this schema guarantees valid structured output.

                Returns:
                    True if the provider enforces valid JSON output.
                """
                return False  # Set to True only if your provider guarantees valid JSON
    ''')

  (package_dir / "schema.py").write_text(schema_content, encoding="utf-8")
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

  patterns_literal = "[" + ", ".join(repr(p) for p in patterns) + "]"
  provider_cls_name = f"{provider_name}LanguageModel"

  test_content = textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """Test script for {provider_name} provider (Step 5 checklist)."""

        import re
        import sys
        import langextract as lx
        from langextract.providers import registry

        try:
            from langextract_{package_name} import {provider_cls_name}
        except ImportError:
            print("ERROR: Plugin not installed. Run: pip install -e .")
            sys.exit(1)

        lx.providers.load_plugins_once()

        PROVIDER_CLS_NAME = "{provider_cls_name}"
        PATTERNS = {patterns_literal}

        def _example_id(pattern: str) -> str:
            \"\"\"Generate test model ID from pattern.\"\"\"
            base = re.sub(r'^\\^', '', pattern)
            m = re.match(r"[A-Za-z0-9._-]+", base)
            base = m.group(0) if m else (base or "model")
            return f"{{base}}-test"

        sample_ids = [_example_id(p) for p in PATTERNS]
        sample_ids.append("unknown-model")

        print("Testing {provider_name} Provider - Step 5 Checklist:")
        print("-" * 50)

        # 1 & 2. Provider registration + pattern matching via resolve()
        print("1–2. Provider registration & pattern matching")
        for model_id in sample_ids:
            try:
                provider_class = registry.resolve(model_id)
                ok = provider_class.__name__ == PROVIDER_CLS_NAME
                status = "✓" if (ok or model_id == "unknown-model") else "✗"
                note = "expected" if ok else ("expected (no provider)" if model_id == "unknown-model" else "unexpected provider")
                print(f"   {{status}} {{model_id}} -> {{provider_class.__name__ if ok else 'resolved'}} {{note}}")
            except Exception as e:
                if model_id == "unknown-model":
                    print(f"   ✓ {{model_id}}: No provider found (expected)")
                else:
                    print(f"   ✗ {{model_id}}: resolve() failed: {{e}}")

        # 3. Inference sanity check
        print("\\n3. Test inference with sample prompts")
        try:
            model_id = sample_ids[0] if sample_ids[0] != "unknown-model" else (_example_id(PATTERNS[0]) if PATTERNS else "test-model")
            provider = {provider_cls_name}(model_id=model_id)
            prompts = ["Test prompt 1", "Test prompt 2"]
            results = list(provider.infer(prompts))
            print(f"   ✓ Inference returned {{len(results)}} results")
            for i, result in enumerate(results):
                try:
                    out = result[0].output if result and result[0] else None
                    print(f"   ✓ Result {{i+1}}: {{(out or '')[:60]}}...")
                except Exception:
                    print(f"   ✗ Result {{i+1}}: Unexpected result shape: {{result}}")
        except Exception as e:
            print(f"   ✗ ERROR: {{e}}")
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
                            attributes={{"type": "example"}}
                        )
                    ]
                )
            ]

            schema = {provider_name}Schema.from_examples(examples)
            print(f"   ✓ Schema created (keys={{list(schema.schema_dict.keys())}})")

            schema_class = {provider_cls_name}.get_schema_class()
            print(f"   ✓ Provider schema class: {{schema_class.__name__}}")

            provider = {provider_cls_name}(model_id=_example_id(PATTERNS[0]) if PATTERNS else "test-model")
            provider.apply_schema(schema)
            print(f"   ✓ Schema applied: response_schema={{provider.response_schema is not None}} structured={{getattr(provider, 'structured_output', False)}}")
        except Exception as e:
            print(f"   ✗ ERROR: {{e}}")
        """)

  test_content += textwrap.dedent(f"""
        # 5. Test factory integration
        print("\\n5. Test factory integration")
        try:
            from langextract import factory
            config = factory.ModelConfig(
                model_id=_example_id(PATTERNS[0]) if PATTERNS else "test-model",
                provider="{provider_cls_name}"
            )
            model = factory.create_model(config)
            print(f"   ✓ Factory created: {{type(model).__name__}}")
        except Exception as e:
            print(f"   ✗ ERROR: {{e}}")

        print("\\n" + "-" * 50)
        print("✅ Testing complete!")
        """)

  (base_dir / "test_plugin.py").write_text(test_content, encoding="utf-8")
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

  readme_content = textwrap.dedent(f"""\
        # LangExtract {provider_name} Provider

A provider plugin for LangExtract that supports {provider_name} models.

## Installation

```bash
pip install -e .
```

## Supported Model IDs

{supported}

## Environment Variables

- `{env_var_safe}`: API key for authentication

## Usage

```python
import langextract as lx

result = lx.extract(
    text="Your document here",
    model_id="{_display(patterns[0]) if patterns else package_name}-model",
    prompt_description="Extract entities",
    examples=[...]
)
```

## Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`

## License

Apache License 2.0
    """)

  (base_dir / "README.md").write_text(readme_content, encoding="utf-8")
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

  (base_dir / ".gitignore").write_text(gitignore_content, encoding="utf-8")
  print("✓ Created .gitignore file with Python-specific entries")


def create_license(base_dir: Path) -> None:
  """Create LICENSE file."""
  license_content = textwrap.dedent("""\
        # LICENSE

        TODO: Add your license here.

        This is a placeholder license file for your provider plugin.
        Please replace this with your actual license before distribution.

        Common options include:
        - Apache License 2.0
        - MIT License
        - BSD License
        - GPL License
        - Proprietary/Commercial License
    """)

  (base_dir / "LICENSE").write_text(license_content, encoding="utf-8")
  print("✓ Created LICENSE file")
  print("✅ Step 6 complete: Documentation created")


def install_and_test(base_dir: Path) -> bool:
  """Install the plugin and run tests."""
  print("\n" + "=" * 60)
  print("Installing and testing the plugin...")
  print("=" * 60)

  os.chdir(base_dir)
  print("\nInstalling plugin...")
  result = subprocess.run(
      [sys.executable, "-m", "pip", "install", "-e", "."],
      capture_output=True,
      text=True,
      check=False,
  )
  if result.returncode:
    print(f"Installation failed: {result.stderr}")
    return False
  print("✓ Plugin installed successfully")

  print("\nRunning tests...")
  result = subprocess.run(
      [sys.executable, "test_plugin.py"],
      capture_output=True,
      text=True,
      check=False,
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
  create_pyproject_toml(base_dir, args.provider_name, package_name)
  create_provider(
      base_dir, args.provider_name, package_name, patterns, args.with_schema
  )

  if args.with_schema:
    create_schema(base_dir, args.provider_name, package_name)

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
          "\n⚠️ Plugin created but tests failed. Please check the"
          " implementation."
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
