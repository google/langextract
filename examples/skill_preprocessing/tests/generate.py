"""Regenerate our small, synthetic fixture with python-pptx."""

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches


def generate(path):
  deck = Presentation()
  for i, region in enumerate(('North', 'South')):
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = f'Quarter {i+1} – 東京'
    slide.shapes.add_textbox(
        Inches(1), Inches(2), Inches(4), Inches(1)
    ).text = f'{region}: $12M'
    table = slide.shapes.add_table(
        2, 2, Inches(1), Inches(4), Inches(4), Inches(2)
    ).table
    for row, values in enumerate((('Region', 'Units'), (region, '42'))):
      for col, value in enumerate(values):
        table.cell(row, col).text = value
    slide.notes_slide.notes_text_frame.text = f'{region} speaker note'
  deck.slides.add_slide(deck.slide_layouts[6])
  deck.save(path)


if __name__ == '__main__':
  generate(Path(__file__).with_name('regions.pptx'))
