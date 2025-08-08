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

# Revalidate all open PRs

echo "Fetching all open PRs..."
PR_NUMBERS=$(gh pr list --limit 50 --json number --jq '.[].number')
TOTAL=$(echo "$PR_NUMBERS" | wc -w | tr -d ' ')

echo "Found $TOTAL open PRs"
echo "Starting revalidation..."
echo ""

COUNT=0
for pr in $PR_NUMBERS; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Triggering revalidation for PR #$pr..."
    gh workflow run revalidate-pr.yml -f pr_number=$pr

    # Small delay to avoid rate limiting
    sleep 2
done

echo ""
echo "All workflows triggered!"
echo ""
echo "To monitor progress:"
echo "  gh run list --workflow=revalidate-pr.yml --limit=$TOTAL"
echo ""
echo "To see results, check comments on each PR"
