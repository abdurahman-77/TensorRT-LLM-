#!/usr/bin/env bash
# Usage: ./test_infer.sh <BASE_URL> ["prompt"]
# BASE_URL is printed by `cerebrium deploy`, e.g. https://api.cerebrium.ai/v4/p-xxxxxxxx/tensorrt-triton-demo
set -euo pipefail
BASE_URL="${1:?base url required}"
PROMPT="${2:-What is machine learning?}"

curl -sS -X POST "${BASE_URL}/v2/models/llama3_2/infer" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys;print(json.dumps({"inputs":[{"name":"text_input","shape":[1,1],"datatype":"BYTES","data":[sys.argv[1]]}],"outputs":[{"name":"text_output"}]}))' "$PROMPT")"
echo
