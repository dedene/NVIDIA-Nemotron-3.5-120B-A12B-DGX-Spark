# AGENTS.md

Serve NVIDIA Nemotron 3.5 Super VL 120B-A12B (NVFP4, 70 GiB) with the LiLiCorr drafter on one DGX Spark with vLLM via Docker Compose.

## Layout
- `compose.yaml`: `download` (one-shot `hf download` of the pinned revisions) -> `nemotron` (vLLM server, healthcheck).
  Knobs are `${VAR:-default}` plus `tools/serve.sh`. `.env.example` documents them.
- `tools/download.sh`, `tools/serve.sh`: fetch and launch. The image is the official vLLM nightly, so there is no Dockerfile.
- `tools/check.py`: functional checks, including the `local-ai` alias and an image (+ `--needle N`).

## Spark
- Same box as `~/Development/MiniCPM-35B-A3B-DGX-Spark` (`ssh zenjoy@10.0.0.158`). Checkout at `~/Development/NVIDIA-Nemotron-3.5-120B-A12B-DGX-Spark`.
- GB10, sm_121, aarch64, 128 GB unified (121.6 GiB visible), driver 580, CUDA 13, Compose v5.
- Port 8888. Stop the other model server first. Hermes calls model id `local-ai`; `ALIAS` covers it.

## Verify
- `docker compose config --quiet`, `shellcheck tools/*.sh`, `python3 -m py_compile tools/*.py`
- `docker compose up -d --wait`, then `tools/check.py` (all ok).

## Image
- `vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2` at index digest
  `sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd` (arm64 `sha256:fd33bec1…`), 25.6 GB.
- The card also names `v0.32.0`; on 2026-10-15 Docker Hub had no such tag (manifest unknown). Switch once it exists.
- Model revision `2e5cc3953bb1ea66b2fb7c831fee3aa317427e69`. Drafter revision `7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2`.

## Weights (config.json)
- `NemotronH_Omni_Reasoning_V3`, llm `nemotron_h`: 88 layers = 40 mamba, 40 moe, 8 attention. 512 experts, 22 per token,
  hidden 4096, 2 kv heads x 128. `max_position_embeddings` 1048576. One MTP layer.
- LiLiCorr drafter: 5-layer qwen3-style, block size 8, sliding window 8192, W4A16 NVFP4, own LM head.

## Findings (2026-10-16)
- Cold start about 12 min: weights 398 s (70.03 GiB), engine init 263 s (graphs 42 s, FlashInfer autotune).
- KV pool 1,806,446 tokens at CONTEXT=1048576, GPU_UTIL 0.75, fp8 KV (vLLM picks it from the checkpoint). ~14 GiB free.
- `tools/check.py` all ok: ids `nemotron-3.5-super-vl` + `local-ai`, alias pong, thinking off `42`, thinking on `15`
  (235 chars), `create_event` Sprint Review / 2027-03-12, streaming Brussels, vision blue, yellow, red.
- Log warning "no KV cache group could be identified as the draft model's" with dflash: harmless, checks pass.
- Decode, 1 stream, temp 0, thinking off: code 51.1, English 32.5, Dutch 23.6, tool call 78.3, weighted 34.6 tok/s.
  Prefill 1,684 tok/s at 1k, 2,373 at 32k.

## Findings (2026-10-16, agent benchmark)
- Private kit, label `nemotron-3.5-super-vl-nvfp4`, run on the Spark itself with `MAX_SEQS=4` (other knobs default).
  Numbers in the README section "As an agent".
- AgentDojo hijacks 37/200: slack 12/29, banking 10/23, travel 6/35, workspace 9/113. Banking hijacks were only
  data-leaking transfers and password changes (injection tasks 1, 2, 3, 7, 8); 4, 5, 6 (redirect, drain, $30k) never.
- BFCL multi-turn n=199: one sample hit `ValueError('zip() argument 2 is shorter')` in the scorer.

## Constraints
- Use the official NVFP4 checkpoint and the card's Spark flags. Do not drop the mamba flags: they are the card's accuracy settings.
- `--trust-remote-code` is required (the processor and config are custom code in the repo).
- Leave thinking on the template default. `enable_thinking: false` is how a request turns it off.
- The wait loop `while pgrep -f "docker pull vllm"` matches its own shell. Poll `docker image inspect` instead.
- Never run `hf cache prune` on the Spark. `hf download --revision <sha>` writes no `refs/main`, so prune deletes every
  pinned model (it took this one, its drafter and three others on 2026-10-16). Remove repos with `hf cache rm` only.
