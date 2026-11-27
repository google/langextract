"""Compare both pipelines on the same fixture with identical canned inference."""

import argparse
from pathlib import Path
import statistics
import time

from demo import options
from langextract_skills import extract_file
from langextract_skills import SkillRegistry
from markitdown import MarkItDown

import langextract as lx


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('skill', type=Path)
  args = parser.parse_args()
  registry = SkillRegistry()
  registry.register_local(args.skill)
  deck = Path(__file__).with_name('tests') / 'regions.pptx'
  converter = MarkItDown(enable_plugins=False)
  direct = []
  adapter = []
  for _ in range(5):
    start = time.perf_counter()
    raw = converter.convert_local(str(deck)).markdown
    baseline = lx.extract(raw, **options())
    direct.append(time.perf_counter() - start)
    start = time.perf_counter()
    result = extract_file(deck, registry, **options())
    adapter.append(time.perf_counter() - start)
    for _, a, b in result.parsed.spans:
      assert result.parsed.text[a:b] in raw
    assert [e.extraction_text for e in baseline.extractions] == [
        e.extraction_text for e in result.annotated.extractions
    ]
  print(
      f'Median of 5 end-to-end runs: direct {statistics.median(direct):.3f}s,'
      f' adapter {statistics.median(adapter):.3f}s'
  )


if __name__ == '__main__':
  main()
