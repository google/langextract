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

# -*- protobuffer -*-
# proto-file: devtools/kokoro/config/proto/build.proto
# proto-message: BuildConfig

# Location of the build script
build_file: "kokoro/test.sh"

# Specify a Docker image that has Python
container_properties {
  docker_image: "us-central1-docker.pkg.dev/kokoro-container-bakery/kokoro/ubuntu/ubuntu2204/full:current"
}

# Define the structured test results
xunit_test_results {
  target_name: "pytest_results"
  result_xml_path: "git/repo/pytest_results/test.xml"
}
