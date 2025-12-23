import glob
import importlib
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_package_integrity(package_name: str) -> List[str]:
  """
  Recursively attempts to import all modules within a package.

  Args:
      package_name: The name of the root package to check.

  Returns:
      A list of error strings if imports fail, empty list otherwise.
  """
  logger.info("Checking Package Integrity: %s", package_name)
  errors: List[str] = []

  # Find package path
  try:
    package = importlib.import_module(package_name)
  except ImportError as e:
    logger.error("Critical: Cannot import package '%s': %s", package_name, e)
    return [f"Root import failed: {e}"]

  if not hasattr(package, "__path__"):
    logger.info(
        "Note: %s is a namespace package or single module.", package_name
    )
    return []

  path = list(package.__path__)[0]
  prefix = package_name + "."

  for root, _, files in os.walk(path):
    for file in files:
      if file.endswith(".py") and file != "__init__.py":
        # Convert file path to module name
        rel_path = os.path.relpath(os.path.join(root, file), path)
        module_name = prefix + rel_path.replace(os.path.sep, ".")[:-3]

        try:
          importlib.import_module(module_name)
        except Exception as e:
          logger.error("Failed to import %s: %s", module_name, e)
          errors.append(f"{module_name}: {e}")

  if not errors:
    logger.info("All modules in '%s' imported successfully.", package_name)
  else:
    logger.error("Found %d import errors.", len(errors))

  return errors


def check_examples_syntax(examples_dir: str) -> List[str]:
  """
  Checks python syntax of all scripts in the directory.

  Args:
      examples_dir: Path to directory containing example scripts.

  Returns:
      List of error strings.
  """
  logger.info("Checking Examples Syntax: %s", examples_dir)
  errors: List[str] = []

  py_files = glob.glob(os.path.join(examples_dir, "**", "*.py"), recursive=True)

  for file_path in py_files:
    try:
      with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
      compile(source, file_path, "exec")
    except SyntaxError as e:
      logger.error("Syntax error in %s: %s", file_path, e)
      errors.append(f"{file_path}: {e}")

  if not errors:
    logger.info("All %d example scripts have valid syntax.", len(py_files))
  else:
    logger.error("Found %d syntax errors in examples.", len(errors))
  return errors


def check_plugin_install(plugin_dir: str) -> List[str]:
  """
  Verifies that a plugin package is installable using uv.

  Args:
      plugin_dir: Path to the plugin source directory.

  Returns:
      List of error strings.
  """
  logger.info("Checking Plugin Installability: %s", plugin_dir)

  # Try to dry-run install with uv
  cmd = ["uv", "pip", "install", "--dry-run", "."]
  try:
    subprocess.run(
        cmd,
        cwd=plugin_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logger.info("Plugin at %s is compatible with uv.", plugin_dir)
    return []
  except subprocess.CalledProcessError as e:
    logger.error("Plugin install check failed: %s", e.stderr.strip())
    return [f"Plugin install failed: {e.stderr}"]


def main() -> None:
  base_dir = Path.cwd()
  errors: List[str] = []

  # 1. Check Source Integrity
  errors.extend(check_package_integrity("langextract"))

  # 2. Check Examples Syntax
  errors.extend(check_examples_syntax("examples"))

  # 3. Check Custom Plugin
  plugin_path = base_dir / "examples" / "custom_provider_plugin"
  if plugin_path.exists():
    errors.extend(check_plugin_install(str(plugin_path)))

  print("\n--- Verification Summary ---")
  if errors:
    logger.error("Checks failed with %d errors.", len(errors))
    sys.exit(1)
  else:
    logger.info("All integrity checks passed.")
    sys.exit(0)


if __name__ == "__main__":
  main()
