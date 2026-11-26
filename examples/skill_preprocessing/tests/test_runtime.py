"""Permission boundary, source grounding, and optional real-model acceptance."""

import os
from pathlib import Path

from demo import options
from langextract_skills import load_skill
from langextract_skills.runtime import extract_with_skill
from langextract_skills.runtime import PlanningExecutor
import pytest

BASE = Path(__file__).resolve().parents[1]
SKILL = BASE / 'sample_skills' / 'confirmed-orders'
DATA = BASE / 'tests' / 'orders.txt'


@pytest.mark.parametrize(
    'reply',
    [
        'no',
        '[]',
        '{}',
        '[1]',
        '[{"tool":"shell"}]',
        '[{"tool":"read_text","path":"other"}]',
        '[{"tool":"keep_lines","contains":"x"}]',
        '[{"tool":"read_text"},{"tool":"read_text"}]',
        '[{"tool":"read_text"},{"tool":"keep_lines","contains":"x"}]',
        'x' * 4097,
    ],
)
def test_invalid_plan_rejected_before_read(reply):
  executor = PlanningExecutor(lambda _: reply, allowed_tools=['read_text'])
  with pytest.raises(ValueError):
    executor.prepare(load_skill(SKILL), Path('absent'), 'read')


def test_instruction_and_document_boundaries():
  def planner(prompt):
    assert 'case-sensitive status label `CONFIRMED:`' in prompt
    assert 'PO-240' not in prompt
    return (
        '[{"tool":"read_text"},{"tool":"keep_lines","contains":"CONFIRMED:"}]'
    )

  doc = PlanningExecutor(
      planner, allowed_tools=['read_text', 'keep_lines']
  ).prepare(load_skill(SKILL), DATA, 'prepare orders')
  assert doc.text == 'CONFIRMED: PO-240' and not doc.spans


def test_no_implicit_grant_or_model():
  with pytest.raises(ValueError):
    PlanningExecutor(str, allowed_tools=['shell'])
  with pytest.raises(ValueError, match='explicitly'):
    extract_with_skill(
        DATA, skill_dir=SKILL, executor=None, prompt_description='read'
    )


def test_public_model_id_and_permission_boundary(monkeypatch):
  from unittest.mock import Mock

  from langextract import factory
  from langextract.core import types

  planner = Mock()
  planner.infer.return_value = iter([[
      types.ScoredOutput(
          score=1,
          output='[{"tool":"read_text"},{"tool":"keep_lines","contains":"CONFIRMED:"}]',
      )
  ]])
  create = Mock(return_value=planner)
  monkeypatch.setattr(factory, 'create_model_from_id', create)
  with pytest.raises(ValueError, match='Unauthorized'):
    extract_with_skill(
        'missing-file',
        skill=str(SKILL),
        allow_tools=['read_text'],
        prompt_description='read',
        model_id='test-model',
    )
  create.assert_called_once_with('test-model')


def test_runtime_deck_alignment():
  executor = PlanningExecutor(
      lambda _: '[{"tool":"markitdown_pptx"}]',
      allowed_tools=['markitdown_pptx'],
  )
  result = extract_with_skill(
      BASE / 'tests' / 'regions.pptx',
      skill_dir=SKILL,
      executor=executor,
      **options(),
  )
  assert [x.slides for x in result.origins] == [(1,), (2,)]


@pytest.mark.skipif(
    not (os.getenv('LX_LOCAL_MODEL') and os.getenv('LX_TEST_SKILL')),
    reason='Requires local MLX weights and pinned official skill',
)
def test_real_model_swap_without_adapter_code(tmp_path):
  from run_skill import _local_model

  import langextract as lx

  model = _local_model(Path(os.environ['LX_LOCAL_MODEL']))
  # Exercise the public convenience API with implicit read-only grants.

  def run(skill, source, task, kind, example):
    return extract_with_skill(
        source,
        skill=skill,
        prompt_description=task,
        model=model,
        fence_output=False,
        use_schema_constraints=False,
        examples=[
            lx.data.ExampleData(
                text=example,
                extractions=[
                    lx.data.Extraction(
                        extraction_class=kind, extraction_text=example
                    )
                ],
            )
        ],
    )

  result = run(
      Path(os.environ['LX_TEST_SKILL']),
      BASE / 'tests' / 'regions.pptx',
      'Read presentation content. Extract each regional revenue statement'
      ' verbatim in order.',
      'revenue',
      'West: $3M',
  )
  assert result.skill_trace.executed_tools == ('markitdown_pptx',)
  assert [o.slides for o in result.origins] == [(1,), (2,)]
  assert [e.extraction_text for e in result.annotated.extractions] == [
      'North: $12M',
      'South: $12M',
  ]
  # Identical name, description, source, task, executor, grants, model and code.
  # Only the natural-language instructions change between these two runs.
  skill = tmp_path / SKILL.name
  skill.mkdir()
  for label, expected in [('CONFIRMED:', 'PO-240'), ('PENDING:', 'PO-108')]:
    manifest = (
        (SKILL / 'SKILL.md').read_text().replace('`CONFIRMED:`', f'`{label}`')
    )
    (skill / 'SKILL.md').write_text(manifest)
    result = run(
        skill, DATA, 'Extract each order ID verbatim.', 'order_id', 'PO-123'
    )
    assert result.skill_trace.executed_tools == ('read_text', 'keep_lines')
    assert result.skill_trace.manifest_sha256 == load_skill(skill).digest
    assert result.skill_trace.name == skill.name
    assert result.parsed.text == f'{label} {expected}'
    assert [e.extraction_text for e in result.annotated.extractions] == [
        expected
    ]
    assert result.origins[0].status == 'unmapped'
