"""Offline fixture demo. --model-id explicitly opts into a configured provider."""

import argparse
import dataclasses
import json
from pathlib import Path

from langextract_skills import extract_file
from langextract_skills import SkillRegistry

import langextract as lx
from langextract.core import base_model
from langextract.core import types


class FixtureModel(base_model.BaseLanguageModel):
  """Canned response for the bundled deck only, not a real model."""

  def infer(self, batch_prompts, **kwargs):
    for _ in batch_prompts:
      yield [
          types.ScoredOutput(
              score=1,
              output=(
                  '{"extractions":[{"revenue":"North: $12M"},{"revenue":"South:'
                  ' $12M"}]}'
              ),
          )
      ]


def options(model_id=None):
  example = lx.data.ExampleData(
      text='West: $3M',
      extractions=[
          lx.data.Extraction(
              extraction_class='revenue', extraction_text='West: $3M'
          )
      ],
  )
  return {
      'prompt_description': (
          'Extract each regional revenue statement verbatim, in order.'
      ),
      'examples': [example],
      **(
          {'model_id': model_id}
          if model_id
          else {
              'model': FixtureModel(),
              'fence_output': False,
              'use_schema_constraints': False,
          }
      ),
  }


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('skill', type=Path)
  parser.add_argument(
      '--source',
      type=Path,
      default=Path(__file__).with_name('tests') / 'regions.pptx',
  )
  parser.add_argument('--model-id')
  args = parser.parse_args()
  registry = SkillRegistry()
  registry.register_local(args.skill)
  result = extract_file(args.source, registry, **options(args.model_id))
  print(
      json.dumps(
          {
              'document_id': result.parsed.document_id,
              'source': result.parsed.source,
              'warnings': result.parsed.warnings,
              'extractions': [
                  {
                      'extraction_class': item.extraction_class,
                      'extraction_text': item.extraction_text,
                      'char_interval': (
                          dataclasses.asdict(item.char_interval)
                          if item.char_interval
                          else None
                      ),
                      'origin': dataclasses.asdict(where),
                  }
                  for item, where in zip(
                      result.annotated.extractions or (), result.origins
                  )
              ],
          },
          ensure_ascii=False,
          indent=2,
      )
  )
