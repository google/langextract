"""Opt-in preprocessing; local skills are data, adapters are trusted code."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Protocol

import yaml

import langextract as lx

from .pptx import ParsedDocument
from .pptx import PptxAdapter


@dataclass(frozen=True)
class SkillSpec:
  name: str
  description: str
  digest: str
  resources: tuple[str, ...]
  instructions: str = ''


def load_skill(root: Path) -> SkillSpec:
  """Inspect only local regular files. Never follow resource links."""
  if root.is_symlink() or not root.is_dir():
    raise ValueError('Expected a local skill directory')
  resources = []
  pending = [root]
  count = 0
  while pending:
    for path in pending.pop().iterdir():
      count += 1
      if count > 1024 or path.is_symlink():
        raise ValueError('Unsafe skill tree')
      if path.is_dir():
        pending.append(path)
      elif path.is_file():
        resources.append(path.relative_to(root).as_posix())
      else:
        raise ValueError('Unsupported resource')
  with (root / 'SKILL.md').open('rb') as stream:
    raw = stream.read(65537)
  if len(raw) > 65536:
    raise ValueError('Oversized manifest')
  lines = raw.decode().splitlines()
  if not lines or lines[0] != '---' or '---' not in lines[1:]:
    raise ValueError('Missing YAML frontmatter')
  header = '\n'.join(lines[1 : lines.index('---', 1)])
  try:
    if any(
        isinstance(token, (yaml.AliasToken, yaml.AnchorToken))
        for token in yaml.scan(header)
    ):
      raise ValueError('YAML aliases are not supported')
    meta = yaml.safe_load(header)
  except (yaml.YAMLError, RecursionError) as exc:
    raise ValueError('Invalid skill YAML') from exc
  if not isinstance(meta, dict):
    raise ValueError('Expected skill metadata')
  name, desc = meta.get('name'), meta.get('description')
  if (
      name != root.name
      or len(name) > 64
      or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name)
  ):
    raise ValueError('Invalid skill name')
  if not isinstance(desc, str) or not desc.strip() or len(desc) > 1024:
    raise ValueError('Invalid skill description')
  return SkillSpec(
      name,
      desc,
      hashlib.sha256(raw).hexdigest(),
      tuple(sorted(resources)),
      '\n'.join(lines[lines.index('---', 1) + 1 :]),
  )


class Adapter(Protocol):
  suffix: str
  digest: str

  def parse(self, source: Path) -> ParsedDocument:
    ...


class SkillRegistry:

  def __init__(self):
    self._skills = {}

  def register_local(self, root: Path, adapter: Adapter | None = None):
    spec = load_skill(root)
    adapter = PptxAdapter() if adapter is None else adapter
    if spec.digest != adapter.digest:
      raise ValueError('Unreviewed skill manifest for this adapter')
    if spec.name in self._skills:
      raise ValueError('Duplicate skill')
    self._skills[spec.name] = adapter
    return spec

  def parse(self, source: Path, skill: str | None = None):
    matches = [
        a
        for n, a in self._skills.items()
        if (skill is None or n == skill) and a.suffix == source.suffix.lower()
    ]
    if len(matches) != 1:
      raise ValueError('Expected one matching registered skill')
    return matches[0].parse(source)


@dataclass(frozen=True)
class Origin:
  status: str
  slides: tuple[int, ...] = ()


def origin(parsed, extraction):
  interval = extraction.char_interval
  if interval is None:
    return Origin('unmapped')
  a, b = interval.start_pos, interval.end_pos
  if not (
      isinstance(a, int)
      and isinstance(b, int)
      and 0 <= a < b <= len(parsed.text)
  ):
    return Origin('unmapped')
  slides = tuple(n for n, start, end in parsed.spans if start < b and a < end)
  if len(slides) > 1:
    return Origin('cross-slide', slides)
  if any(start <= a < b <= end for _, start, end in parsed.spans):
    if (
        parsed.text[a:b] != extraction.extraction_text
        or extraction.alignment_status != lx.data.AlignmentStatus.MATCH_EXACT
    ):
      return Origin('approximate', slides)
    return Origin('mapped', slides)
  return Origin('unmapped')


@dataclass(frozen=True)
class SkillTrace:
  name: str
  manifest_sha256: str
  executed_tools: tuple[str, ...] | None


@dataclass(frozen=True)
class Result:
  parsed: ParsedDocument
  annotated: lx.data.AnnotatedDocument
  origins: tuple[Origin, ...]
  skill_trace: SkillTrace | None = None


def extract_file(
    source: Path, registry: SkillRegistry, *, skill=None, **kwargs
) -> Result:
  if not any(
      kwargs.get(key) is not None
      for key in ('model', 'config', 'model_id', 'language_model_type')
  ):
    raise ValueError(
        'Select a model explicitly, or use registry.parse for local text'
    )
  parsed = registry.parse(source, skill)
  return _extract_parsed(parsed, kwargs)


def _extract_parsed(parsed: ParsedDocument, kwargs: dict) -> Result:
  doc = lx.data.Document(text=parsed.text, document_id=parsed.document_id)
  (annotated,) = lx.extract(text_or_documents=[doc], **kwargs)
  if (
      annotated.text != parsed.text
      or annotated.document_id != parsed.document_id
  ):
    raise ValueError('Provider result does not match source')
  return Result(
      parsed,
      annotated,
      tuple(origin(parsed, item) for item in annotated.extractions or ()),
  )
