#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate rag_py312

python "${SCRIPT_DIR}/../run_sig_HI_rag_top1.py"