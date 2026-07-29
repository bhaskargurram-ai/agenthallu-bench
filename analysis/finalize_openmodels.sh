#!/bin/bash
# Finalize: fold open-weight-model results into the canonical data files, then
# regenerate hop analysis, tables, figures, and recompile the IEEE manuscript.
# Idempotent w.r.t. backups: originals are saved once as *.closed.csv.
set -e
cd "$(dirname "$0")/.."
PY=./venv/bin/python
R=data/results

echo "### 1. merge open models -> *_v2.csv"
$PY analysis/merge_open_models.py

echo "### 2. back up closed-only canonical files (once) and swap in merged"
for f in p2_all_rejudged baseline_all_rejudged baseline_all; do
  [ -f "$R/$f.closed.csv" ] || cp "$R/$f.csv" "$R/$f.closed.csv"
done
cp "$R/p2_all_rejudged_v2.csv"        "$R/p2_all_rejudged.csv"
cp "$R/baseline_all_rejudged_v2.csv"  "$R/baseline_all_rejudged.csv"
cp "$R/baseline_all_v2.csv"           "$R/baseline_all.csv"

echo "### 3. regenerate hop analysis (feeds paper prose numbers)"
$PY analysis/hop_analysis.py >/dev/null 2>&1 || $PY analysis/hop_analysis.py

echo "### 4. regenerate tables + figures (closed + open)"
$PY paper_elsarticle/generate_tables.py 2>&1 | tail -2 || true
$PY paper_elsarticle/generate_figures.py 2>&1 | tail -2 || true

echo "### 5. copy regenerated assets into paper_ieee"
cp paper_elsarticle/tables/*.tex paper_ieee/tables/ 2>/dev/null || true
cp paper_elsarticle/figures/*.png paper_ieee/figures/ 2>/dev/null || true

echo "### 5b. widen wide tables to full column (table*) + scale to text width"
for t in table4_main table5_hop table7_interceptor; do
  f=paper_ieee/tables/$t.tex; [ -f "$f" ] || continue
  sed -i '' 's/\\begin{table}/\\begin{table*}/; s/\\end{table}/\\end{table*}/' "$f"
  # scale the tabular to the full text width so nothing overflows the columns
  perl -0pi -e 's/\\begin\{tabular\}/\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}/; s/\\end\{tabular\}/\\end{tabular}}/' "$f"
done

echo "### 6. recompile IEEE manuscript"
cd paper_ieee
rm -f main.aux main.bbl main.blg
pdflatex -interaction=nonstopmode main.tex >/tmp/f1.log 2>&1
bibtex main >/tmp/fb.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/tmp/f2.log 2>&1
pdflatex -interaction=nonstopmode main.tex >/tmp/f3.log 2>&1
if [ -f main.pdf ]; then
  echo "OK main.pdf pages=$(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}') size=$(ls -lh main.pdf|awk '{print $5}')"
else
  echo "COMPILE FAILED:"; grep -E '^! ' /tmp/f1.log | head
fi
