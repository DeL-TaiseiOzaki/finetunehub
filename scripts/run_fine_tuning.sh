#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
MAIN_SCRIPT="${SCRIPT_DIR}/main.py"

#マルチの設定がある場合、torchrunを使用
USE_DDP=$(grep 'ddp: true' ${SCRIPT_DIR}/../configs/config.yaml)

if [ -n "$USE_DDP" ]; then
  CMD="torchrun --nproc_per_node=$(nproc --all) ${MAIN_SCRIPT}"
else
  CMD="python3 ${MAIN_SCRIPT}"
fi

# 実行
echo "実行するコマンド:"
echo "${CMD}"

eval "${CMD}"
