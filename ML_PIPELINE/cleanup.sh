#!/bin/bash
python utils/cleanup.py \
  --ml-models-path ./models \
  --datasets-path ./datasets \
  --experiments-path ./experiments \
  "$@"
