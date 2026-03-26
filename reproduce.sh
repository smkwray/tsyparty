#!/usr/bin/env bash
set -euo pipefail

python -m tsyparty.cli show-plan
python -m tsyparty.cli registry --public-only
python -m tsyparty.cli example --out outputs/sample

echo "Seed repo bootstrapped. Next step: run scripts/download_all_public.py"
