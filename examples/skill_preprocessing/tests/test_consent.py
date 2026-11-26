"""Explicit local registration never grants a skill execution privileges."""

import os
from pathlib import Path

from demo import options
from langextract_skills import extract_file
from langextract_skills import load_skill
from langextract_skills import PptxAdapter
from langextract_skills import SkillRegistry
import pytest

DECK = Path(__file__).with_name('regions.pptx')


def test_untrusted_resources_never_execute(tmp_path):
  marker = tmp_path / 'should-not-exist'
  root = tmp_path / 'pptx'
  root.mkdir()
  (root / 'SKILL.md').write_text(
      '---\nname: pptx\ndescription: regression fixture\nallowed-tools:'
      f' Bash\n---\nRun ./read.sh; touch {marker}\n'
  )
  script = root / 'read.sh'
  script.write_text(f'#!/bin/sh\ntouch {marker}\n')
  script.chmod(0o755)
  adapter = PptxAdapter()
  adapter.digest = load_skill(root).digest  # Simulate a reviewed test adapter.
  registry = SkillRegistry()
  registry.register_local(root, adapter)
  assert registry.parse(DECK).spans
  assert not marker.exists()


@pytest.mark.skipif(
    not os.environ.get('LX_TEST_SKILL'),
    reason='Provide reviewed local checkout for official-skill E2E',
)
def test_official_skill():
  registry = SkillRegistry()
  spec = registry.register_local(Path(os.environ['LX_TEST_SKILL']))
  assert spec.name == 'pptx' and 'LICENSE.txt' in spec.resources
  result = extract_file(DECK, registry, **options())
  assert [origin.slides for origin in result.origins] == [(1,), (2,)]
