#!/usr/bin/env bash
# Start vLLM with the model card's DGX Spark settings. SPEC picks the drafter: lilicorr (default, 7 tokens),
# mtp (the model's own head, 5 tokens) or none. Extra arguments are passed on to `vllm serve`.
set -euo pipefail

snapshot() { echo "/root/.cache/huggingface/hub/models--${1//\//--}/snapshots/$2"; }

case ${SPEC:-lilicorr} in
  lilicorr)
    spec=(--speculative-config "{\"method\": \"dflash\", \"model\": \"$(snapshot "$DRAFT_ID" "$DRAFT_REVISION")\", \"num_speculative_tokens\": ${SPEC_TOKENS:-7}}") ;;
  mtp)
    spec=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS:-5}}") ;;
  none)
    spec=() ;;
  *)
    echo "SPEC must be lilicorr, mtp or none, not '$SPEC'" >&2
    exit 1 ;;
esac

names=("$SERVED_NAME")
[[ -n ${ALIAS:-} ]] && names+=("$ALIAS")
auth=()
[[ -n ${API_KEY:-} ]] && auth=(--api-key "$API_KEY")

exec vllm serve "$(snapshot "$MODEL_ID" "$MODEL_REVISION")" \
  --served-model-name "${names[@]}" \
  --host "$HOST" --port "$PORT" \
  --trust-remote-code \
  --max-model-len "$CONTEXT" \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  "${spec[@]}" \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --enable-mamba-shared-prefix-checkpoint \
  --video-pruning-rate 0.5 \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
  --block-size 64 \
  --prefix-match-unit 64 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  "${auth[@]}" \
  "$@"
