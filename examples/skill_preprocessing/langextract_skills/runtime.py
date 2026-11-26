"""Interpret a selected local skill at run time. A plan is data, not code."""

from collections.abc import Callable, Collection
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Protocol

from langextract import factory

from . import _extract_parsed
from . import load_skill
from . import ParsedDocument
from . import PptxAdapter
from . import Result
from . import SkillSpec
from . import SkillTrace


class SkillExecutor(Protocol):
  """Trusted application component. Implementations own their permission policy."""

  def prepare(
      self, skill: SkillSpec, source: Path, task: str
  ) -> ParsedDocument:
    ...


TOOLS = {
    'read_text': (
        'Read the caller-selected UTF-8 plain text file. No arguments.'
    ),
    'markitdown_pptx': (
        'Read the caller-selected PowerPoint file with MarkItDown; preserve'
        ' slide boundaries. No arguments.'
    ),
    'keep_lines': (
        'Keep only complete lines containing a case-sensitive literal'
        ' substring. Requires string argument "contains".'
    ),
}


def _text(source):
  if source.is_symlink() or not source.is_file():
    raise ValueError('Expected regular local source')
  with source.open('rb') as stream:
    raw = stream.read(1024 * 1024 + 1)
  if len(raw) > 1024 * 1024 or b'\x00' in raw:
    raise ValueError('Expected at most 1 MiB of UTF-8 text')
  return ParsedDocument(
      raw.decode(), hashlib.sha256(raw).hexdigest(), source.name, (), ()
  )


class PlanningExecutor:
  """Compile prose to a short allowlisted read-only plan with a caller's model."""

  def __init__(
      self, planner: Callable[[str], str], *, allowed_tools: Collection[str]
  ):
    self._planner = planner
    self._allowed = frozenset(allowed_tools)
    if not self._allowed or self._allowed - TOOLS.keys():
      raise ValueError('Select known, non-empty tool grants')

  def prepare(
      self, skill: SkillSpec, source: Path, task: str
  ) -> ParsedDocument:
    prompt = (
        'Act as a document preprocessing planner. Read the supplied skill'
        ' instructions and implement ONLY the read/preparation workflow needed'
        ' for the user task. Do not perform the final extraction. Return one'
        ' JSON array of tool calls, no markdown. Each call is {"tool":"name"}'
        ' with only the documented arguments as additional keys. Start with one'
        ' reader, then apply any required filters. The task describes what a'
        ' separate extractor will do later; it does not need a tool here. For'
        ' example, instructions to read a text file and retain rows mentioning'
        ' London compile to'
        ' [{"tool":"read_text"},{"tool":"keep_lines","contains":"London"}]. For'
        ' a deck requiring only text conversion use'
        ' [{"tool":"markitdown_pptx"}]. Only return [] when the requested'
        ' preprocessing itself cannot be done. Do not approximate unsupported'
        ' operations. No shell, URLs, scripts or other resources are'
        ' available.\n'
        + json.dumps(
            {
                'available_tools': {k: TOOLS[k] for k in sorted(self._allowed)},
                'source_extension': source.suffix.lower(),
                'task': task,
                'skill_description': skill.description,
                'skill_instructions': skill.instructions,
            },
            ensure_ascii=False,
        )
        + '\nEnd of skill. Select ONLY the reading workflow for the given task;'
        ' ignore workflows for creating, editing, rendering or validating'
        ' deliverables unless requested. The next stage handles extraction.'
        ' Emit the JSON tool calls now.'
    )
    reply = self._planner(prompt)
    if not isinstance(reply, str) or len(reply) > 4096:
      raise ValueError('Invalid skill plan response')
    try:
      plan = json.loads(reply)
    except (ValueError, RecursionError) as exc:
      raise ValueError('Expected a JSON tool plan') from exc
    if not isinstance(plan, list) or not 1 <= len(plan) <= 8:
      raise ValueError('Unsupported workflow or invalid plan length')
    for index, call in enumerate(plan):
      if (
          not isinstance(call, dict)
          or not isinstance(call.get('tool'), str)
          or call['tool'] not in self._allowed
      ):
        raise ValueError('Unauthorized tool')
      name = call['tool']
      if index == 0:
        if name not in ('read_text', 'markitdown_pptx') or set(call) != {
            'tool'
        }:
          raise ValueError('Expected one reader without arguments')
        if name == 'markitdown_pptx' and source.suffix.lower() != '.pptx':
          raise ValueError('Expected PPTX source')
      elif (
          name != 'keep_lines'
          or set(call) != {'tool', 'contains'}
          or not isinstance(call['contains'], str)
          or not 1 <= len(call['contains']) <= 256
      ):
        raise ValueError('Invalid filter operation')
    parsed = (
        _text(source)
        if plan[0]['tool'] == 'read_text'
        else PptxAdapter().parse(source)
    )
    executed = [plan[0]['tool']]
    for call in plan[1:]:
      filtered = '\n'.join(
          line for line in parsed.text.splitlines() if call['contains'] in line
      )
      parsed = replace(
          parsed,
          text=filtered,
          spans=(),
          warnings=parsed.warnings
          + ('Filtered text: document-level source only',),
      )
      executed.append('keep_lines')
    if not parsed.text.strip():
      raise ValueError('Skill produced no readable text')
    return replace(parsed, executed_tools=tuple(executed))


def extract_with_skill(
    source: str | Path,
    *,
    skill: str | Path | None = None,
    skill_dir: Path | None = None,
    executor: SkillExecutor | None = None,
    allow_tools: Collection[str] | None = None,
    prompt_description: str,
    **kwargs,
) -> Result:
  """Load the selected manifest for this invocation; no per-skill registration."""
  if not any(
      kwargs.get(k) is not None
      for k in ('model', 'model_id', 'config', 'language_model_type')
  ):
    raise ValueError('Select an extraction model explicitly')
  if (skill is None) == (skill_dir is None):
    raise ValueError('Select exactly one local skill directory')
  spec = load_skill(Path(skill if skill is not None else skill_dir))
  if executor is not None and allow_tools is not None:
    raise ValueError('Configure permissions on the custom executor instead')
  if executor is None:
    model = kwargs.get('model')
    if model is None:
      if not kwargs.get('model_id') or any(
          kwargs.get(k) is not None for k in ('config', 'language_model_type')
      ):
        raise ValueError('Default planner requires model or model_id')
      model = factory.create_model_from_id(kwargs['model_id'])
    executor = PlanningExecutor(
        lambda prompt: next(iter(model.infer([prompt])))[0].output,
        allowed_tools=('read_text', 'markitdown_pptx', 'keep_lines')
        if allow_tools is None
        else allow_tools,
    )
  parsed = executor.prepare(spec, Path(source), prompt_description)
  result = _extract_parsed(
      parsed, dict(kwargs, prompt_description=prompt_description)
  )
  return replace(
      result,
      skill_trace=SkillTrace(spec.name, spec.digest, parsed.executed_tools),
  )
