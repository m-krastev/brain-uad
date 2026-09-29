#!/usr/bin/env bash
# HD-BET v2 skull stripping of all IXI T2 scans (GPU, with test-time mirroring).
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/hd-bet -i data/raw/IXI-T2 -o data/ixi_bet -device cuda
