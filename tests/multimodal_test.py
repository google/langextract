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

"""Tests for multimodal (text + images) extraction plumbing."""

from absl.testing import absltest

from langextract import exceptions
from langextract import extraction as extraction_lib
from langextract.core import base_model
from langextract.core import data
from langextract.core import types


class _FakeMultimodalModel(base_model.BaseLanguageModel):

  def __init__(self):
    super().__init__()
    self.seen_images = []

  @property
  def supports_images(self) -> bool:
    return True

  def infer(self, batch_prompts, **kwargs):
    self.seen_images.append(kwargs.get("images"))
    for _ in batch_prompts:
      yield [types.ScoredOutput(score=1.0, output='{"extractions":[{"person":"Bob"}]}')]


class _FakeTextOnlyModel(base_model.BaseLanguageModel):

  def infer(self, batch_prompts, **kwargs):
    for _ in batch_prompts:
      yield [types.ScoredOutput(score=1.0, output='{"extractions":[{"person":"Bob"}]}')]


class MultimodalPlumbingTest(absltest.TestCase):

  def _examples(self):
    return [
        data.ExampleData(
            text="Bob went home.",
            extractions=[data.Extraction("person", "Bob")],
        )
    ]

  def test_extract_passes_global_images_to_model(self):
    model = _FakeMultimodalModel()
    extraction_lib.extract(
        "Bob went home.",
        prompt_description="Extract people.",
        examples=self._examples(),
        model=model,
        fence_output=False,
        images=[b"img-bytes"],
        show_progress=False,
    )

    self.assertLen(model.seen_images, 1)
    batch_images = model.seen_images[0]
    self.assertLen(batch_images, 1)
    self.assertLen(batch_images[0], 1)
    self.assertIsInstance(batch_images[0][0], data.Image)

  def test_extract_raises_for_text_only_provider_when_images_present(self):
    with self.assertRaises(exceptions.InferenceConfigError):
      extraction_lib.extract(
          "Bob went home.",
          prompt_description="Extract people.",
          examples=self._examples(),
          model=_FakeTextOnlyModel(),
          fence_output=False,
          images=[b"img-bytes"],
          show_progress=False,
      )

  def test_document_images_merge_with_global_images(self):
    model = _FakeMultimodalModel()
    doc_img = data.Image(data=b"doc", mime_type="image/png")
    global_img = data.Image(data=b"global", mime_type="image/png")
    docs = [
        data.Document("Bob went home.", images=[doc_img]),
        data.Document("Bob went home.", images=None),
    ]

    extraction_lib.extract(
        docs,
        prompt_description="Extract people.",
        examples=self._examples(),
        model=model,
        fence_output=False,
        images=[global_img],
        batch_length=10,
        show_progress=False,
    )

    self.assertLen(model.seen_images, 1)
    batch_images = model.seen_images[0]
    self.assertLen(batch_images, 2)
    self.assertEqual(batch_images[0], [doc_img, global_img])
    self.assertEqual(batch_images[1], [global_img])


if __name__ == "__main__":
  absltest.main()

