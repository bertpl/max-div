#!/bin/bash
cd /Users/bertpluymers/git/owned/public/max-div
export PYTHONPATH=. MPLCONFIGDIR=/tmp/claude-mpl
mkdir -p /tmp/claude-mpl
uv run --group benchmarks --python 3.14 python scripts/generate_concept_images.py > generated/regen_concept.log 2>&1
uv run --group benchmarks --python 3.14 python scripts/generate_guide_images.py --reuse-solution > generated/regen_guide.log 2>&1
make docs > generated/regen_docs.log 2>&1
echo DONE > generated/regen.done
