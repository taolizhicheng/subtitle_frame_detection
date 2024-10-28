#!/bin/bash

set -e

# 判断OUTPUT_DIR变量是否存在
if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR 变量未设置"
    exit 1
fi

RUNS_DIR="${OUTPUT_DIR}/runs"

rm -rf ${RUNS_DIR}