#!/bin/bash

set -e

THIS_DIR=$(cd $(dirname $0); pwd)

python ${THIS_DIR}/inference.py $@
