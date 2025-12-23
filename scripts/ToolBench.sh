#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate rag_py312
# 1 is G1 query; 2 is G2 query; 3 is G3 query
python "${SCRIPT_DIR}/../tool_bench_hi_rag.py" 1