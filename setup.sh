#!/usr/bin/env bash
# Session 0 environment bootstrap — ABLkit + PyEDCR track
set -euo pipefail

VENV=".venv"
PYTHON="${PYTHON:-python3}"

echo "=== [1/5] Creating virtual environment at $VENV ==="
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "=== [2/5] Upgrading pip ==="
pip install --upgrade pip

echo "=== [3/5] Installing requirements.txt ==="
pip install -r requirements.txt

echo "=== [4/5] Installing PyEDCR from source (CIKM-2024 f-EDR branch) ==="
pip install "git+https://github.com/lab-v2/PyEDCR.git"

echo "=== [5/5] Registering Jupyter kernel ==="
python -m ipykernel install --user --name neurosymbolic --display-name "neurosymbolic (py3)"

echo ""
echo "──────────────────────────────────────────────────────"
echo "✓  Setup complete."
echo ""
echo "  Activate :  source $VENV/bin/activate"
echo "  Notebooks:  jupyter lab"
echo "  Capstone :  python week4/demo.py --help"
echo "──────────────────────────────────────────────────────"
echo ""
echo "NOTE: ABLkit requires SWI-Prolog on PATH."
echo "  Ubuntu/Debian: sudo apt install swi-prolog"
echo "  macOS:         brew install swi-prolog"
