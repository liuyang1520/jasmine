#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
model_file="$project_dir/models/Qwen3.5-0.8B-Q4_K_M.gguf"
expected_sha=f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec
image=ghcr.io/ggml-org/llama.cpp@sha256:8fdfad183be053cdb72d4b6a5930c3a2475730f932855ee9260193422c2564ed

if [ ! -f "$model_file" ]; then
  echo "Model missing: $model_file" >&2
  exit 1
fi
actual_sha=$(shasum -a 256 "$model_file" | cut -d ' ' -f 1)
if [ "$actual_sha" != "$expected_sha" ]; then
  echo "Model SHA-256 mismatch" >&2
  exit 1
fi

container run -d --name jasmine-llama \
  --read-only --network none --cap-drop ALL --uid 65534 --gid 65534 \
  --cpus 4 --memory 4G \
  --mount "type=bind,source=$project_dir,target=/work,readonly" \
  "$image" \
  --model /work/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 --ctx-size 2048 --threads 4 --parallel 1

for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if container exec jasmine-llama curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "Isolated model ready. Run scripts/ask-container.sh < request.json"
    exit 0
  fi
  sleep 1
done
container logs jasmine-llama >&2
exit 1
