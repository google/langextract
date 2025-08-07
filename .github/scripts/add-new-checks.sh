#!/bin/bash
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

# Script to add new required status checks to an existing branch protection rule.
# This preserves all your current settings and just adds the new checks

echo "Adding new PR validation checks to existing branch protection..."

# Add the new checks to existing ones
echo "Adding new checks: enforce, size, and protect-infrastructure..."
gh api repos/:owner/:repo/branches/main/protection/required_status_checks/contexts \
  --method POST \
  --input - <<< '["enforce", "size", "protect-infrastructure"]'

echo ""
echo "✓ New checks added!"
echo ""
echo "Updated required status checks will include:"
echo "- test (3.10)                    [existing]"
echo "- test (3.11)                    [existing]"
echo "- test (3.12)                    [existing]"
echo "- Validate PR Template           [existing]"
echo "- live-api-tests                 [existing]"
echo "- ollama-integration-test        [existing]"
echo "- enforce                        [NEW - linked issue validation]"
echo "- size                           [NEW - PR size limit]"
echo "- protect-infrastructure         [NEW - infrastructure file protection]"
