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

# Add size labels to PRs based on their change count

echo "Adding size labels to PRs..."

# Get all open PRs with their additions and deletions
gh pr list --limit 50 --json number,additions,deletions --jq '.[]' | while read -r pr_data; do
    pr_number=$(echo "$pr_data" | jq -r '.number')
    additions=$(echo "$pr_data" | jq -r '.additions')
    deletions=$(echo "$pr_data" | jq -r '.deletions')
    total_changes=$((additions + deletions))

    # Determine size label
    if [ $total_changes -lt 50 ]; then
        size_label="size/XS"
    elif [ $total_changes -lt 150 ]; then
        size_label="size/S"
    elif [ $total_changes -lt 600 ]; then
        size_label="size/M"
    elif [ $total_changes -lt 1000 ]; then
        size_label="size/L"
    else
        size_label="size/XL"
    fi

    echo "PR #$pr_number: $total_changes lines -> $size_label"

    # Remove any existing size labels first
    existing_labels=$(gh pr view $pr_number --json labels --jq '.labels[].name' | grep "^size/" || true)
    if [ ! -z "$existing_labels" ]; then
        echo "  Removing existing label: $existing_labels"
        gh pr edit $pr_number --remove-label "$existing_labels"
    fi

    # Add the new size label
    gh pr edit $pr_number --add-label "$size_label"

    sleep 1  # Avoid rate limiting
done

echo "Done adding size labels!"
