"""Private POSIX worker. No plugin discovery, script execution or URL inputs."""

import io
import json
import re
import resource
import sys
import zipfile


def convert(data):
  # Apply worker resource limits before importing conversion dependencies.
  # pylint: disable=import-outside-toplevel
  from defusedxml import ElementTree
  from markitdown import StreamInfo
  from markitdown.converters import PptxConverter
  from pptx import Presentation

  with zipfile.ZipFile(io.BytesIO(data)) as archive:
    entries = archive.infolist()
    if len(entries) > 2048 or sum(e.file_size for e in entries) > 64 << 20:
      raise ValueError('Archive limit')
    names = set()
    for entry in entries:
      name = entry.filename.rstrip('/')
      if (
          name in names
          or re.search(r'[:\\]', name)
          or any(p in ('', '.', '..') for p in name.split('/'))
      ):
        raise ValueError('Unsafe package path')
      if (
          entry.compress_type not in (0, 8)
          or entry.file_size > 16 << 20
          or entry.flag_bits & 1
      ):
        raise ValueError('Unsafe package part')
      names.add(name)
      if name.lower().endswith(('.xml', '.rels')):
        ElementTree.fromstring(archive.read(entry), forbid_dtd=True)
  count = len(Presentation(io.BytesIO(data)).slides)
  markdown = (
      PptxConverter()
      .convert(io.BytesIO(data), StreamInfo(extension='.pptx'))
      .markdown
  )
  markers = list(
      re.finditer(r'^<!-- Slide number: ([0-9]+) -->$', markdown, re.M)
  )
  if not count:
    raise ValueError('No slides')
  if [m[1] for m in markers] != list(map(str, range(1, count + 1))):
    return (
        markdown,
        [],
        ['Unverifiable slide boundaries; document-level text only'],
    )
  blocks, spans, warnings = [], [], []
  offset = 0
  for i, marker in enumerate(markers):
    block = markdown[
        marker.end() : markers[i + 1].start() if i + 1 < count else None
    ].strip()
    if not re.sub(
        r'^!\[.*?\]\(.*\)$|^### Notes:$', '', block, flags=re.M
    ).strip():
      warnings.append(f'Slide {i+1}: no readable text (no OCR)')
      continue
    if blocks:
      offset += 2
    spans.append((i + 1, offset, offset + len(block)))
    blocks.append(block)
    offset += len(block)
  return '\n\n'.join(blocks), spans, warnings


if __name__ == '__main__':
  resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
  resource.setrlimit(resource.RLIMIT_FSIZE, (4 << 20, 4 << 20))
  if sys.platform.startswith('linux'):
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
  json.dump(convert(sys.stdin.buffer.read()), sys.stdout, ensure_ascii=False)
