"""Run a local skill with explicit model and tool grants. See RUNTIME.md."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from langextract_skills.runtime import extract_with_skill
from langextract_skills.runtime import TOOLS

from langextract import factory
import langextract as lx
from langextract.core import base_model
from langextract.core import types


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('skill', type=Path)
  parser.add_argument('source', type=Path)
  parser.add_argument(
      '--allow',
      action='append',
      choices=sorted(TOOLS),
      help='Narrow the default local read-only capabilities',
  )
  model_group = parser.add_mutually_exclusive_group(required=True)
  model_group.add_argument(
      '--model-id',
      help=(
          'Configured LangExtract provider; may send skill and document text'
          ' externally'
      ),
  )
  model_group.add_argument(
      '--mlx-local',
      type=Path,
      help='Optional local MLX language model directory (Apple Silicon)',
  )
  parser.add_argument('--task', required=True)
  parser.add_argument('--example-class', required=True)
  parser.add_argument('--example-text', required=True)
  args = parser.parse_args()
  model = (
      _local_model(args.mlx_local)
      if args.mlx_local
      else factory.create_model_from_id(args.model_id)
  )

  result = extract_with_skill(
      args.source,
      skill=args.skill,
      allow_tools=args.allow,
      prompt_description=args.task,
      examples=[
          lx.data.ExampleData(
              text=args.example_text,
              extractions=[
                  lx.data.Extraction(
                      extraction_class=args.example_class,
                      extraction_text=args.example_text,
                  )
              ],
          )
      ],
      model=model,
      fence_output=False,
      use_schema_constraints=False,
  )
  print(
      json.dumps(
          {
              'skill_trace': asdict(result.skill_trace),
              'source': result.parsed.source,
              'document_id': result.parsed.document_id,
              'prepared_text': result.parsed.text,
              'warnings': result.parsed.warnings,
              'extractions': [
                  {
                      'text': e.extraction_text,
                      'interval': (
                          asdict(e.char_interval) if e.char_interval else None
                      ),
                      'origin': asdict(o),
                  }
                  for e, o in zip(
                      result.annotated.extractions or (), result.origins
                  )
              ],
          },
          ensure_ascii=False,
          indent=2,
      )
  )


def _local_model(path):
  # Optional smoke-test backend; no MLX imports on the normal provider path.
  from mlx_lm import generate  # pylint: disable=import-outside-toplevel
  from mlx_lm import load  # pylint: disable=import-outside-toplevel

  if not path.is_dir():
    raise ValueError('Expected an installed local model directory')
  weights, tokenizer = load(str(path.resolve()))

  class LocalModel(base_model.BaseLanguageModel):

    def infer(self, batch_prompts, **kwargs):
      for prompt in batch_prompts:
        formatted = tokenizer.apply_chat_template(
            [
                {
                    'role': 'system',
                    'content': (
                        'Return only valid JSON as requested. No Markdown'
                        ' fences.'
                    ),
                },
                {'role': 'user', 'content': prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        yield [
            types.ScoredOutput(
                score=1,
                output=generate(
                    weights,
                    tokenizer,
                    prompt=formatted,
                    max_tokens=512,
                    verbose=False,
                ),
            )
        ]

  return LocalModel()


if __name__ == '__main__':
  main()
