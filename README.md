# NVIDIA Nemotron 3.5 Super VL 120B-A12B on a single DGX Spark (vLLM, Docker Compose)

Serve [NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) on one NVIDIA DGX Spark with vLLM and NVIDIA's [LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr) speculative drafter. One command downloads the pinned checkpoint and drafter, and starts an OpenAI-compatible API with text, images, reasoning and tool calls.

Nemotron 3.5 Super VL is a 120B mixture-of-experts model with 12B active per token: 40 Mamba-2 layers, 40 MoE layers (512 experts, 22 per token) and only 8 attention layers, plus a C-RADIO vision encoder. The NVFP4 checkpoint takes 70 GiB on the GPU. Because only the 8 attention layers keep a KV cache, the native 1M-token window fits on the Spark too: the default settings leave a pool of 1.8M tokens.

> **Day-one recipe.** The model and drafter came out on 15 October 2026. The model card names `vllm/vllm-openai:v0.32.0`, but that tag was not on Docker Hub yet, so this pins the nightly the card also names (`nightly-3f1bcfd8`) by digest. Expect the image and numbers to change.

**Hardware:** DGX Spark (GB10, sm_121, aarch64), driver 580, CUDA 13, Docker with the NVIDIA Container Toolkit and Compose v2. About 100 GB free disk for the weights, drafter and the vLLM image.

## Quickstart

```bash
git clone https://github.com/dedene/NVIDIA-Nemotron-3.5-120B-A12B-DGX-Spark.git
cd NVIDIA-Nemotron-3.5-120B-A12B-DGX-Spark
docker compose up -d --wait     # download 71 GB (first run only), load, wait until healthy
tools/check.py                  # model ids, thinking switch, tool call, streaming, an image
docker compose down
```

`up --wait` returns once `/health` answers. A cold start takes about 12 minutes: 7 minutes of weight loading, then CUDA graph capture and FlashInfer autotuning. `docker compose logs -f nemotron` shows progress. The torch.compile, Triton and FlashInfer caches live in `~/.cache/nemotron-3.5-super-vl-dgx-spark`, so later starts skip most of the compilation.

The API is at `http://<spark-ip>:8888/v1`, model `nemotron-3.5-super-vl`. The server also answers to `local-ai` (set `ALIAS=` to drop that).

```bash
curl http://<spark-ip>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nemotron-3.5-super-vl",
  "messages": [{"role": "user", "content": "Explain in three sentences why a Mamba layer needs no KV cache."}],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Thinking is on by default, as in the chat template. Per request, `chat_template_kwargs.enable_thinking: false` turns it off, and `{"enable_thinking": true, "medium_effort": true}` asks for medium effort. The reasoning comes back in `message.reasoning`, the answer in `message.content`. NVIDIA recommends temperature 1.0 and top_p 0.95 for text, 0.6 and 0.95 for images.

Images go in as an `image_url` part. A data URL works:

```bash
printf '{"model": "nemotron-3.5-super-vl", "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,%s"}},
    {"type": "text", "text": "What is in this photo?"}]}]}' "$(base64 < photo.jpg | tr -d '\n')" |
  curl http://<spark-ip>:8888/v1/chat/completions -H 'Content-Type: application/json' -d @-
```

## How it is put together

- **Image:** `vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2`, pinned by digest. It is the official multi-arch vLLM nightly, with the `nemotron_h_omni` architecture, the LiLiCorr (`dflash`) drafter path and the `nemotron_v3` reasoning parser. No Dockerfile.
- **`tools/download.sh`:** `hf download` of the pinned model revision, plus the drafter when `SPEC=lilicorr`. Resumable, and a no-op once complete.
- **`tools/serve.sh`:** `vllm serve` with the flags from the model card's DGX Spark quick start: FlashInfer Mamba backend, fp16 SSM cache with stochastic rounding, shared-prefix Mamba checkpoints, block size 64, the `qwen3_coder` tool parser and the `nemotron_v3` reasoning parser. `SPEC` picks the drafter.
- **`compose.yaml`:** the `download` service runs to completion, then `nemotron` serves offline from your normal Hugging Face cache. A healthcheck hits `/health`.
- Nothing is installed on the host.

## Configuration

Copy `.env.example` to `.env` and uncomment what you want to change, then `docker compose up -d --wait` again (Compose recreates the container when settings change).

| Variable | Default | |
|---|---|---|
| `SPEC` | `lilicorr` | `lilicorr` (NVIDIA's Spark drafter, 1.3 GB), `mtp` (the model's own MTP head) or `none`. |
| `SPEC_TOKENS` | 7 for lilicorr, 5 for mtp | Draft length. |
| `CONTEXT` | 1048576 | Tokens per sequence. The checkpoint's native window. |
| `MAX_SEQS` | 1 | Concurrent sequences. The model card's Spark recipe uses 1. |
| `GPU_UTIL` | 0.75 | Share of unified memory vLLM claims. |
| `BATCHED_TOKENS` | 8192 | Chunked-prefill size. |
| `PORT`, `HOST`, `SERVED_NAME`, `ALIAS` | 8888, 0.0.0.0, nemotron-3.5-super-vl, local-ai | |
| `API_KEY` | (empty) | Bearer token, when set. |
| `VLLM_IMAGE` | nightly `3f1bcfd8` digest | Switch to `vllm/vllm-openai:v0.32.0` once it is published. Re-run `tools/check.py` after a bump. |

For any other vLLM flag, append it to the `nemotron` service: `tools/serve.sh` passes extra arguments on to `vllm serve`.

## Notes

- **Memory.** Weights 70.0 GiB, drafter about 1.3 GiB, CUDA graphs 1.9 GiB, and 17.3 GiB of KV cache at `GPU_UTIL=0.75`. About 14 GiB stays free for the OS. vLLM stores the KV cache in fp8, as the checkpoint's quantization config asks.
- **Video.** The serve flags include the card's video settings (2 fps, up to 256 frames, pruning 0.5). The card lists the Spark recipe as text and image; `tools/check.py` tests an image, not video.
- **Hermes.** Hermes on the same Spark calls `http://127.0.0.1:8888/v1` with model id `local-ai`. vLLM rejects ids it does not serve, which is why `ALIAS` defaults to `local-ai`.
- **The API is open on the LAN.** Set `API_KEY` to require a bearer token.
- **Port 8888** is the Spark's usual model port. Stop the other model server first.

## Credits

- [NVIDIA](https://huggingface.co/nvidia) for Nemotron 3.5 Super VL, the NVFP4 checkpoint, the LiLiCorr drafter and the DGX Spark serving recipe on the model card (OpenMDW-1.1). The weights are not in this repository.
- The [vLLM](https://github.com/vllm-project/vllm) project and its nightly images.
- The shape of this repository follows [Kolibri-1-DGX-Spark](https://github.com/dedene/Kolibri-1-DGX-Spark): one Compose project, a pinned download, and a small check.

## License

MIT, see [LICENSE](LICENSE). The model weights are licensed by NVIDIA under the [OpenMDW License 1.1](https://openmdw.ai/license/1-1/).
