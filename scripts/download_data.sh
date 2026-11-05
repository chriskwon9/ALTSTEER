#!/usr/bin/env bash
# Download the benchmarks from their public sources into data/inputs/.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python altsteer/prepare_data.py
