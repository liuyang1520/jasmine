# Jasmine Local

A small local implementation of Jev's **state → typed decisions** interface. It is an independent adapter around a local Qwen3.5 0.8B model, not Jev or a reproduction of its undisclosed model architecture or calibrated training method.

## How it works

The caller supplies a `state` and named `choice`, `score`, or `noul` questions. Each question is rendered against the same state, independently. The model sees at most ten options, labelled `A` through `J`. llama.cpp constrains the answer to one letter. We verify that each letter is a single token, add the same large logit bias to each allowed letter, then request the post-sampling probabilities. Equal bias preserves the letters' relative logits while making all of them appear in the returned top-token list; the code rejects incomplete distributions. Our Python code maps letters back to caller-defined values and computes the score as a probability-weighted mean. It never parses model-generated JSON or executes a model-selected action.

To reduce answer-letter bias, each option appears in every letter position once. We average the resulting distributions by option meaning. Choice keys are sorted before these rotations, so changing only the request object's insertion order produces the same probabilities. This costs **N model passes for N options**. It removes first-order position bias, but cannot remove all prompt-order interactions or make the probabilities calibrated.

The passes run sequentially, unlike Jev's parallel question sampler. The returned probabilities are averages of model probabilities over the allowed letters. They have **not been calibrated** against held-out data. `confidence` is only normalized concentration of the largest probability: `(max(p) - 1/n) / (1 - 1/n)`. It is not a measured probability of correctness. The caller should validate answers on its own data and set review thresholds there.

The base model is [Qwen3.5 0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), Apache 2.0 licensed. We use [LM Studio Community's Q4_K_M GGUF conversion](https://huggingface.co/lmstudio-community/Qwen3.5-0.8B-GGUF/blob/main/Qwen3.5-0.8B-Q4_K_M.gguf), 528 MB, because GGUF is a data file and llama.cpp can run it without loading Python model code or pickle. The model and official llama.cpp container image are pinned by hash. We reviewed the [llama.cpp endpoints](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) used here (`/apply-template`, `/completion`, grammar, and probability output).

## Run on macOS with Apple container

Requires Apple's `container` CLI. Its containers are Linux VMs, so this isolated mode uses CPU rather than Metal.

```sh
mkdir -p models
curl -fL -o models/Qwen3.5-0.8B-Q4_K_M.gguf \
  https://huggingface.co/lmstudio-community/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf
shasum -a 256 models/Qwen3.5-0.8B-Q4_K_M.gguf
# expected: f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec

container image pull ghcr.io/ggml-org/llama.cpp@sha256:8fdfad183be053cdb72d4b6a5930c3a2475730f932855ee9260193422c2564ed
scripts/start-container.sh
scripts/ask-container.sh < examples/support.json
scripts/stop-container.sh
```

The start script verifies the model hash, then starts the pinned image with a read-only root, a read-only project mount, no network, no Linux capabilities, and an unprivileged user. The model server listens only within the container. `ask-container.sh` sends JSON over stdin with `container exec`; neither request data nor model output needs a host TCP port. The model file is ignored by Git.

For an existing **local** llama.cpp server on `127.0.0.1:8080`, the same Python implementation can also run directly:

```sh
python3 jasmine_local.py ask < examples/support.json
python3 jasmine_local.py serve --port 8765
```

The second command exposes `POST http://127.0.0.1:8765/v1/systemone` on loopback. The server requires no third-party Python packages. It accepts `jev-latest` as an input alias for SDK-style requests, but returns the actual local model name.

## Example request

See [examples/support.json](examples/support.json). The response contains `model`, `answers`, and `usage`. `choice` includes the chosen key and all probabilities; `score` includes an interpolated numeric score, legend, and probabilities; `noul` includes the probability assigned to yes.

Limits: 16 questions per call, 2–10 options per Choice, 2–10 levels per Score, 16 KB of state. Requests beyond these limits return 422. The Python HTTP server binds loopback only. Inputs are treated as data in the prompt, but a small model may still be influenced by malicious or confusing state text; constrained output limits the result to the listed values.

## Validation and observed limits

Run the checks inside the offline model container:

```sh
container exec jasmine-llama sh -lc 'cd /work && python3 -m unittest discover -s tests -p test_local.py -v'
container exec jasmine-llama sh -lc 'cd /work && PYTHONPATH=/work python3 tests/live_check.py'
container exec jasmine-llama sh -lc 'cd /work && PYTHONPATH=/work python3 tests/http_check.py'
```

On this Mac, the five explicit billing/technical/sales cases passed. Reordering a Choice request changed no per-option probabilities in the smoke check. A three-option balanced Choice took about 0.9 seconds on four container CPU cores, compared with about 0.3 seconds for one pass; repeated prompts sometimes hit llama.cpp's cache. A ten-option request returned all ten probabilities summing to 1. The urgent vs. nonurgent examples had the expected ordering for both Noul and Score, but the balanced Noul assigned about 0.51 to yes even for “No deadline; I can wait until next week.” That is weak discrimination. The routing examples are a smoke check, not a benchmark or calibration study. Criterion wording changed one sales result from incorrect to correct, so production use needs representative labeled examples and deliberate option descriptions.

Apple container's host port forwarding reset connections on this machine; the `container exec` route above worked. For a network API, run the Python server with a native local llama.cpp process, or resolve that host forwarding issue first.

## References

- [TypeSafe's Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [TypeSafe System One API reference](https://docs.typesafe.ai/api)
- [Apple container GPU limitation](https://github.com/apple/containerization/issues/46)
