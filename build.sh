#!/bin/bash
set -euxo pipefail
jupytext --to ipynb session*/*.py
jupyter nbconvert --inplace --clear-output --execute session*/*.ipynb 
jupyter book build --html --execute
mkdir -p _build/html/pyvista
cp session*/pyvista_*.html _build/html/pyvista

