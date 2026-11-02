#!/bin/bash
set -e
cd "$(dirname "$0")"
cp ../results/tables/*.tex tables/
mv tables/numbers.tex numbers.tex
cp ../results/case_study/case_stats.tex tables/
cp ../results/case_study/{thumb_*,strip_*,test_groups_*,test_outcomes_*}.png ../results/case_study/labels_*.tex figures/
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex > build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "built $(pwd)/report.pdf"
latexmk -pdf -interaction=nonstopmode -halt-on-error flowchart_standalone.tex > build_flowchart.log 2>&1
python -c "
import pymupdf
page = pymupdf.open('flowchart_standalone.pdf')[0]
box = pymupdf.Rect()
for b in page.get_text('blocks') + [d['rect'] for d in page.get_drawings()] + [i['bbox'] for i in page.get_image_info()]:
    box |= pymupdf.Rect(b[:4]) if isinstance(b, tuple) else b
page.get_pixmap(dpi=160, clip=box + (-8, -8, 8, 8)).save('figures/flowchart.png')"
