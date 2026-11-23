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

import importlib.util
import pathlib
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts"
    / "check_mock_patch_spec.py"
)


def _load_checker():
  spec = importlib.util.spec_from_file_location(
      "check_mock_patch_spec", _SCRIPT_PATH
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


checker = _load_checker()


class FindUnspeccedPatchesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="string_path_patch",
          source='mock.patch("pkg.func")',
      ),
      dict(
          testcase_name="patch_object",
          source='mock.patch.object(pkg, "func")',
      ),
      dict(
          testcase_name="imported_patch_name",
          source='patch("pkg.func")',
      ),
      dict(
          testcase_name="return_value_only",
          source='mock.patch("pkg.func", return_value=1)',
      ),
      dict(
          testcase_name="unittest_mock_patch_object",
          source='unittest.mock.patch.object(pkg, "func")',
      ),
  )
  def test_flags_call_without_replacement(self, source):
    self.assertEqual(checker.find_unspecced_patches(source), [1])

  @parameterized.named_parameters(
      dict(
          testcase_name="autospec",
          source='mock.patch("pkg.func", autospec=True)',
      ),
      dict(
          testcase_name="spec",
          source='mock.patch.object(pkg, "func", spec=pkg.Real)',
      ),
      dict(
          testcase_name="new_keyword_none",
          source='mock.patch.object(pkg, "HTML", new=None)',
      ),
      dict(
          testcase_name="new_positional",
          source='mock.patch("pkg.LIMIT", 3)',
      ),
      dict(
          testcase_name="new_positional_on_object",
          source='mock.patch.object(pkg, "LIMIT", 3)',
      ),
      dict(
          testcase_name="new_callable",
          source='mock.patch("pkg.prop", new_callable=mock.PropertyMock)',
      ),
      dict(
          testcase_name="patch_dict_is_out_of_scope",
          source='mock.patch.dict(os.environ, {"KEY": "value"})',
      ),
      dict(
          testcase_name="unrelated_patch_method",
          source="requests.patch(url)",
      ),
      dict(
          testcase_name="unrelated_object_method",
          source='registry.object(pkg, "func")',
      ),
  )
  def test_accepts_call_with_replacement(self, source):
    self.assertEmpty(checker.find_unspecced_patches(source))


class CheckPathsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    self.tests_dir = pathlib.Path(tmp_dir) / "tests"
    self.tests_dir.mkdir()
    self.test_file = self.tests_dir / "sample_test.py"
    self.test_file.write_text(
        'mock.patch("pkg.one")\nmock.patch("pkg.two")\n', encoding="utf-8"
    )

  def test_new_file_with_unspecced_call_fails_with_fix_and_lines(self):
    errors = checker.check_paths([self.tests_dir], legacy={})

    self.assertLen(errors, 1)
    self.assertIn("Add autospec=True", errors[0])
    self.assertIn("lines: 1, 2", errors[0])

  def test_legacy_file_at_its_recorded_count_passes(self):
    errors = checker.check_paths(
        [self.tests_dir], legacy={"tests/sample_test.py": 2}
    )

    self.assertEmpty(errors)

  def test_legacy_file_above_its_recorded_count_fails(self):
    errors = checker.check_paths(
        [self.tests_dir], legacy={"tests/sample_test.py": 1}
    )

    self.assertLen(errors, 1)
    self.assertIn("1 of these predate the check", errors[0])

  def test_single_file_uses_the_same_legacy_key_as_its_directory(self):
    errors = checker.check_paths(
        [self.test_file], legacy={"tests/sample_test.py": 2}
    )

    self.assertEmpty(errors)

  def test_same_file_name_in_subdirectory_gets_no_legacy_allowance(self):
    nested = self.tests_dir / "nested" / "sample_test.py"
    nested.parent.mkdir()
    nested.write_text('mock.patch("pkg.three")\n', encoding="utf-8")

    errors = checker.check_paths(
        [self.tests_dir], legacy={"tests/sample_test.py": 2}
    )

    self.assertLen(errors, 1)
    self.assertIn("nested", errors[0])


if __name__ == "__main__":
  absltest.main()
