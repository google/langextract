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

"""Tests for verifying provider loading behavior in factory."""

from absl.testing import absltest

from langextract import exceptions
from langextract import factory
from langextract import providers
from langextract.providers import router


class FactoryProviderLoadingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Reset loading state so builtins are re-registered
    providers._reset_for_testing()  # pylint: disable=protected-access
    # Clear registry to ensure we are testing the loading mechanism
    router.clear()

  def test_create_model_with_schema_loads_builtins_explicit_provider(self):
    """Test that builtins are loaded when provider is explicitly set."""
    # This configuration caused a crash before the fix because
    # load_builtins_once() was skipped.
    config = factory.ModelConfig(
        model_id="gpt-4o",
        provider="openai",
        provider_kwargs={"api_key": "dummy-key"},
    )

    try:
      # This calls the method that had the bug
      factory._create_model_with_schema(config)
    except exceptions.InferenceConfigError as e:
      # If the bug exists, the error message starts with "No provider found matching"
      if "No provider found matching" in str(e):
        self.fail(f"Bug reproduced: Builtin provider not loaded! Error: {e}")

      # If we get "API key not provided" or similar, it means the class WAS found!
      # This is SUCCESS.
    except ImportError:
      # Also success - means we tried to import dependencies (so we found the class)
      pass
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.fail(f"Unexpected exception raised: {type(e).__name__}: {e}")


if __name__ == "__main__":
  absltest.main()
