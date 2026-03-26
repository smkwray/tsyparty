#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m tsyparty.cli show-plan
python -m tsyparty.cli example --out outputs/sample
