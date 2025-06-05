#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <HF_MODEL_DIR> <ONNX_OUTPUT_DIR>"
  echo "  Example: $0 ./models/model/hf_pretrained ./model_onnx"
  exit 1
fi

HF_MODEL_DIR="$1"
ONNX_DIR="$2"

echo "=> Creating directory for ONNX (if not exists): ${ONNX_DIR}"
mkdir -p "${ONNX_DIR}"

echo "=> Converting model to ONNX..."
python -m transformers.onnx \
    --model "${HF_MODEL_DIR}" \
    --feature multiple-choice \
    --preprocessor tokenizer \
    --opset 17 \
    "${ONNX_DIR}"

echo "✅ ONNX model saved to: ${ONNX_DIR}/model.onnx"
