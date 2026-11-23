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
import os
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
      dict(
          testcase_name="unittest_mock_patch",
          source='unittest.mock.patch("pkg.func")',
      ),
      dict(
          testcase_name="imported_patch_object",
          source='patch.object(pkg, "func")',
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
          testcase_name="new_positional_on_unittest_mock_patch",
          source='unittest.mock.patch("pkg.LIMIT", 3)',
      ),
      dict(
          testcase_name="new_positional_on_imported_patch_object",
          source='patch.object(pkg, "LIMIT", 3)',
      ),
      dict(
          testcase_name="new_positional_on_imported_patch",
          source='patch("pkg.LIMIT", 3)',
      ),
      dict(
          testcase_name="new_positional_on_unittest_mock_patch_object",
          source='unittest.mock.patch.object(pkg, "LIMIT", 3)',
      ),
      dict(
          testcase_name="spec_set",
          source='mock.patch("pkg.func", spec_set=pkg.Real)',
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
  def test_does_not_flag(self, source):
    self.assertEmpty(checker.find_unspecced_patches(source))

  def test_reports_lines_in_ascending_order(self):
    source = (
        "def outer():\n"
        '  with mock.patch("pkg.nested"):\n'
        "    pass\n"
        'mock.patch("pkg.top_level")\n'
    )

    self.assertEqual(checker.find_unspecced_patches(source), [2, 4])


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

  def test_legacy_file_below_its_recorded_count_passes(self):
    errors = checker.check_paths(
        [self.tests_dir], legacy={"tests/sample_test.py": 3}
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

  def test_relative_path_from_inside_the_directory_uses_the_same_key(self):
    self.addCleanup(os.chdir, os.getcwd())
    os.chdir(self.tests_dir)

    errors = checker.check_paths(
        [pathlib.Path(".")], legacy={"tests/sample_test.py": 2}
    )

    self.assertEmpty(errors)

  def test_bare_file_name_from_inside_the_directory_uses_the_same_key(self):
    self.addCleanup(os.chdir, os.getcwd())
    os.chdir(self.tests_dir)

    errors = checker.check_paths(
        [pathlib.Path("sample_test.py")], legacy={"tests/sample_test.py": 2}
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


class MainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    self.test_file = pathlib.Path(tmp_dir) / "sample_test.py"

  def test_unspecced_call_exits_with_failure(self):
    self.test_file.write_text('mock.patch("pkg.func")\n', encoding="utf-8")

    self.assertEqual(checker.main([str(self.test_file)]), 1)

  def test_specced_call_exits_cleanly(self):
    self.test_file.write_text(
        'mock.patch("pkg.func", autospec=True)\n', encoding="utf-8"
    )

    self.assertEqual(checker.main([str(self.test_file)]), 0)

  def test_missing_path_exits_with_usage_error(self):
    self.assertEqual(checker.main([str(self.test_file)]), 2)

  def test_no_arguments_exits_with_usage_error(self):
    self.assertEqual(checker.main([]), 2)


if __name__ == "__main__":
  absltest.main()
