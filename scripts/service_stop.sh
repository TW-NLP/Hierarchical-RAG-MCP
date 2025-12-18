#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 执行上层目录的 service.py
python3 "${SCRIPT_DIR}/../service.py" stop