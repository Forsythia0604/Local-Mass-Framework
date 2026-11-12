#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"

latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build \
  -pdflatex='pdflatex -no-shell-escape %O %S' main.tex

cp build/main.pdf main.pdf.tmp
mv -f main.pdf.tmp main.pdf
printf 'Updated %s/main.pdf\n' "$PROJECT_DIR"
