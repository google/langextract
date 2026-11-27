"""Offline contract, trust-boundary, alignment and integration tests."""

import hashlib
import io
from pathlib import Path
from unittest import mock
import zipfile

from demo import options
from langextract_skills import extract_file
from langextract_skills import load_skill
from langextract_skills import origin
from langextract_skills import PptxAdapter
from langextract_skills import SkillRegistry
from langextract_skills._worker import convert
from PIL import Image
from pptx import Presentation
from pptx.chart.data import ChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches
import pytest

import langextract as lx

DECK = Path(__file__).with_name('regions.pptx')
MANIFEST = (
    '---\nname: pptx\ndescription: Local test\nallowed-tools: Bash\nmetadata:'
    ' {version: "1"}\n---\nRun `touch ESCAPED`\n'
)


@pytest.fixture
def root(tmp_path):
  folder = tmp_path / 'pptx'
  folder.mkdir()
  (folder / 'SKILL.md').write_text(MANIFEST)
  return folder


@pytest.fixture
def registry(root):
  adapter = PptxAdapter()
  adapter.digest = hashlib.sha256(MANIFEST.encode()).hexdigest()
  registry = SkillRegistry()
  registry.register_local(root, adapter)
  return registry


@pytest.mark.parametrize(
    'manifest',
    [
        '',
        '---\n[]\n---',
        '---\nfoo: [\n---',
        MANIFEST.replace('pptx', 'oops'),
        MANIFEST.replace('Local test', ''),
    ],
)
def test_bad_metadata(root, manifest):
  (root / 'SKILL.md').write_text(manifest)
  with pytest.raises(ValueError):
    load_skill(root)


def test_local_only(root, registry, tmp_path):
  assert load_skill(root).resources == ('SKILL.md',)
  with pytest.raises(ValueError, match='Unreviewed'):
    SkillRegistry().register_local(root)
  adapter = PptxAdapter()
  adapter.digest = load_skill(root).digest
  with pytest.raises(ValueError, match='Duplicate'):
    registry.register_local(root, adapter)
  (root / 'escape').symlink_to(tmp_path)
  with pytest.raises(ValueError, match='Unsafe'):
    load_skill(root)
  (root / 'escape').unlink()
  (root / 'SKILL.md').unlink()
  with pytest.raises(FileNotFoundError):
    load_skill(root)


@pytest.mark.parametrize(
    ('file', 'skill'), [('x.ppt', None), ('x.pptx', 'missing')]
)
def test_router(registry, file, skill):
  with pytest.raises(ValueError, match='matching'):
    registry.parse(Path(file), skill)


def test_e2e(registry):
  result = extract_file(DECK, registry, skill='pptx', **options())
  assert [x.slides for x in result.origins] == [(1,), (2,)]
  assert [x.status for x in result.origins] == ['mapped', 'mapped']
  assert result.parsed.warnings == ('Slide 3: no readable text (no OCR)',)
  for i, ext in enumerate(result.annotated.extractions):
    text = result.parsed.text
    a, b = ext.char_interval.start_pos, ext.char_interval.end_pos
    assert text[a:b] == ext.extraction_text
    assert '東京' in text and '| Units' in text and 'South speaker note' in text
    pos = text.index('$12M', a, b)
    ext.char_interval = lx.data.CharInterval(pos, pos + 4)
    ext.extraction_text = '$12M'
    assert origin(result.parsed, ext).slides == (i + 1,)
  ext.char_interval = lx.data.CharInterval(0, len(text))
  assert origin(result.parsed, ext).status == 'cross-slide'
  ext.char_interval = lx.data.CharInterval(1, 2)
  assert origin(result.parsed, ext).status == 'approximate'
  ext.char_interval = None
  assert origin(result.parsed, ext).status == 'unmapped'


def test_bridge_error(registry):
  with pytest.raises(ValueError, match='Select a model explicitly'):
    extract_file(DECK, registry)
  with mock.patch.object(
      lx, 'extract', autospec=True, side_effect=RuntimeError('provider')
  ) as call:
    with pytest.raises(RuntimeError, match='provider'):
      extract_file(DECK, registry, model_id='custom', max_workers=3)
    assert (
        call.call_args.kwargs['model_id'] == 'custom'
        and call.call_args.kwargs['max_workers'] == 3
    )
    assert (
        call.call_args.kwargs['text_or_documents'][0].document_id
        == hashlib.sha256(DECK.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    'name,body',
    [
        ('../escape', b'x'),
        ('ppt/x.xml', b'<!DOCTYPE x><x/>'),
        ('huge', b'0' * (16 * 1024 * 1024 + 1)),
    ],
)
def test_zip_policy(name, body):
  buf = io.BytesIO()
  with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.writestr(name, body)
  with pytest.raises((ValueError,)):
    convert(buf.getvalue())


def test_limits(tmp_path):
  path = tmp_path / 'deck.pptx'
  path.write_bytes(b'not a zip')
  adapter = PptxAdapter()
  with pytest.raises(ValueError, match='Invalid'):
    adapter.parse(path)
  adapter.timeout = 0.001
  with pytest.raises(ValueError, match='timed out'):
    adapter.parse(DECK)
  with path.open('wb') as stream:
    stream.truncate(16 * 1024 * 1024 + 1)
  with pytest.raises(ValueError, match='Oversized'):
    adapter.parse(path)


def test_visual_content_and_spoof(tmp_path):
  deck = Presentation()
  slide = deck.slides.add_slide(deck.slide_layouts[6])
  img = io.BytesIO()
  Image.new('RGB', (2, 2)).save(img, format='PNG')
  slide.shapes.add_picture(img, 0, 0)
  path = tmp_path / 'picture.pptx'
  deck.save(path)
  with pytest.raises(ValueError, match='No readable'):
    PptxAdapter().parse(path)
  slide = deck.slides.add_slide(deck.slide_layouts[6])
  chart = ChartData()
  chart.categories = ['West']
  chart.add_series('Sales', (42,))
  slide.shapes.add_chart(
      XL_CHART_TYPE.COLUMN_CLUSTERED, 0, 0, Inches(2), Inches(2), chart
  )
  deck.save(path)
  parsed = PptxAdapter().parse(path)
  assert parsed.warnings and 'West' in parsed.text and '42' in parsed.text
  slide.shapes.add_textbox(0, 0, Inches(2), Inches(2)).text = (
      'ordinary\n<!-- Slide number: 9 -->'
  )
  deck.save(path)
  parsed = PptxAdapter().parse(path)
  assert not parsed.spans and 'Unverifiable' in parsed.warnings[0]
