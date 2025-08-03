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

#!/bin/bash
# Kokoro test script for the langextract project.

# Exit immediately if a command exits with a non-zero status.
set -e
set -o pipefail

echo "========================================="
echo "Kokoro Test for langextract"
echo "========================================="

# Navigate to the root of the repository checkout.
cd "${KOKORO_ARTIFACTS_DIR}/git/repo"
echo "Current working directory: $(pwd)"

# Install Python 3.10-venv if it's not already installed.
echo "Installing python3-venv..."
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3.10-venv
else
  apt-get update -y
  apt-get install -y python3.10-venv
fi

echo "Setting up Python Virtual Environment..."
# Ensure Python 3.8+ is available from the Kokoro environment/Docker image
python3 -m venv /tmp/langextract_kokoro_venv
source /tmp/langextract_kokoro_venv/bin/activate
echo "Python version in venv: $(python --version)"

echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools --progress-bar off

echo "Installing dependencies and langextract package..."
if [ -f "pyproject.toml" ]; then
  echo "Found pyproject.toml. Installing langextract in editable mode with test dependencies..."
  pip install -e ".[test]" --progress-bar off
elif [ -f "python/requirements.txt" ]; then
  # Fallback to requirements.txt if pyproject.toml is not used for full dependency management
  echo "Found python/requirements.txt. Installing dependencies..."
  pip install -r python/requirements.txt --progress-bar off
else
  echo "Error: Neither pyproject.toml (at root) nor python/requirements.txt found."
  echo "Cannot determine how to install dependencies."
  exit 1
fi

# Verify pytest is installed
if ! pip show pytest > /dev/null 2>&1; then
    echo "Warning: pytest not found after installation steps."
    echo "Ensure pytest is listed in pyproject.toml [project.optional-dependencies.test] or requirements.txt."
fi

echo "Installing pytype for type checking..."
pip install pytype --progress-bar off

echo "Running type checking with pytype..."
# Note: pytype doesn't support Python 3.13+ yet
if python3 --version | grep -q "3.1[3-9]"; then
  echo "⚠️  Warning: pytype doesn't support Python 3.13+ yet, skipping type checking"
else
  echo "Running pytype on python/langextract/ directory..."
  if ! pytype python/langextract; then
    echo "❌ Type checking failed! Fix type errors before proceeding."
    deactivate
    exit 1
  fi
  echo "✅ Type checking passed!"
fi

echo "Running tests using pytest..."
export PYTHONPATH="$(pwd)/python:$PYTHONPATH"

# Define a subdirectory and define the XML output file path.
KOKORO_PYTEST_RESULTS_SUBDIR="pytest_results"
mkdir -p "${KOKORO_PYTEST_RESULTS_SUBDIR}"
TARGET_JUNIT_XML_FILE_PATH="${KOKORO_PYTEST_RESULTS_SUBDIR}/test.xml"

# Discover and run tests (e.g., python/*_test.py).
# Ensure XML_OUTPUT_FILE is set, otherwise default to a standard location.
XML_OUTPUT_FILE="${TARGET_JUNIT_XML_FILE_PATH}"

echo "Pytest will generate JUnit XML report at: $(pwd)/${TARGET_JUNIT_XML_FILE_PATH}"
python3 -m pytest python/ --junitxml="${TARGET_JUNIT_XML_FILE_PATH}"

echo "Deactivating virtual environment..."
deactivate

echo "========================================="
echo "Kokoro test script for langextract finished successfully."
echo "========================================="
