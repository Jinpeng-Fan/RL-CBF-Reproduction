#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex > build/pass1.txt 2>&1
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex > build/pass2.txt 2>&1
cp build/main.pdf main.pdf
