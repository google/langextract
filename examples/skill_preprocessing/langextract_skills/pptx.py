"""Run only our reviewed converter worker, never code in the skill tree."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class ParsedDocument:
  text: str
  document_id: str
  source: str
  spans: tuple[tuple[int, int, int], ...]
  warnings: tuple[str, ...]
  executed_tools: tuple[str, ...] | None = None


class PptxAdapter:
  suffix = '.pptx'
  digest = 'b60bd282c6e01297f0eefe71594d1756dbf60a584f524c04a803ec6950b7fba2'
  timeout = 30

  def parse(self, source: Path) -> ParsedDocument:
    if os.name != 'posix':
      raise ValueError('Resource-limited worker requires POSIX')
    if source.is_symlink() or not source.is_file():
      raise ValueError('Expected regular local presentation')
    with source.open('rb') as stream:
      data = stream.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
      raise ValueError('Oversized presentation')
    worker = Path(__file__).with_name('_worker.py').resolve()
    with (
        tempfile.TemporaryDirectory() as temp,
        tempfile.TemporaryFile() as output,
    ):
      try:
        subprocess.run(
            [sys.executable, '-I', str(worker)],
            input=data,
            stdout=output,
            stderr=subprocess.DEVNULL,
            cwd=temp,
            env={'HOME': temp, 'TMPDIR': temp, 'OPENBLAS_NUM_THREADS': '1'},
            timeout=self.timeout,
            check=True,
        )
      except subprocess.TimeoutExpired as exc:
        raise ValueError('Presentation conversion timed out') from exc
      except subprocess.CalledProcessError as exc:
        raise ValueError(
            'Invalid, unsupported or resource-limited presentation'
        ) from exc
      output.seek(0)
      text, spans, warnings = json.load(output)
    if not text.strip() or (not spans and not warnings):
      raise ValueError('No readable slide text')
    return ParsedDocument(
        text,
        hashlib.sha256(data).hexdigest(),
        source.name,
        tuple(map(tuple, spans)),
        tuple(warnings),
    )
