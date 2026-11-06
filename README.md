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

## As an agent: benchmark results

I ran this stack through the same agent benchmark as the other models I run on this Spark, on 16 October 2026. Same box, same samples, thinking off, temperature 0. The only change from the defaults was `MAX_SEQS=4`, so the harness could send four requests at once; everything else is the model card's Spark recipe with LiLiCorr. The suites are the standard [Inspect](https://inspect.aisi.org.uk/) tasks from inspect_evals 0.24.0. Brackets are the 95% interval.

| | Nemotron 3.5 Super VL (NVFP4, vLLM + LiLiCorr) | Qwen3.8 Flash Next (NVIDIA FP4-Local, vLLM + MTP) | MiniCPM-V 4.7 (Q8_0, llama.cpp) | Underdog Saluki 27B (IQ2, llama.cpp + DFlash) |
|---|---:|---:|---:|---:|
| BFCL single-turn, n=500 | 85.6 (82–88) | 82.4 | **89.0** | 86.8 |
| BFCL multi-turn, n=199 | 49.2 (42–56) | 54.5 | **63.0** | 48.2 |
| AgentDojo, tasks done without an attack, n=96 | 77.1 (68–84) | 84.4 | **85.4** | 75.0 |
| AgentDojo, attacks that executed, n=200 (lower is better) | 55/200 (22–34) | **2/200** (0–4) | 78/200 (33–46) | 29/200 (10–20) |
| AgentDojo, attacks that disclosed user data, n=200 (lower is better) | 30/200 (11–21) | **0/200** (0–2) | 41/200 (15–27) | 13/200 (4–11) |
| Same, mail, calendar and files: executed, n=113 | 24/113 (15–30) | **0/113** (0–3) | 12/113 (6–18) | 8/113 (4–13) |
| Same, mail, calendar and files: disclosed, n=113 | 15/113 (8–21) | **0/113** (0–3) | 6/113 (2–11) | 2/113 (0–6) |
| AgentDojo, task done with no unauthorised action, n=200 | 55.5 (49–62) | **81.0** (75–86) | 48.5 (42–55) | 58.0 (51–65) |
| IFEval, prompt strict, n=541 | 85.6 (82–88) | 85.8 | **88.5** | 79.3 |
| GPQA-Diamond, n=198 | 56.1 (49–63) | 64.6 | **69.7** | 49.0 |
| Global-MMLU Dutch, n=300 | 86.7 (82–90) | **87.0** | 86.7 | 81.3 |
| Scripted loop, steps per minute (latency only, no outcome scored) | **19.2** | 13.8 | 18.3 | 10.1 |
| Decode, tok/s weighted over code, prose and tool calls | 34.6 | 43.6 | **55.7** | 44.9 |
| Prefill at ~32k tokens, tok/s | **2,373** | 2,274 | 1,854 | 704 |
| Memory for the whole server | ~91 GB | ~97 GB | ~44 GB | 27.5 GB |

**Corrected on 6 November 2026.** This table used to show "attacks resisted", taken from AgentDojo's own score. That score only counts an attack when every step completes, so a model that mailed the data out but skipped deleting the sent mail counted as safe. The rows now count what the model did: attacks that went through as an unauthorised action, and attacks that got the user's data out. Lower is better. Data disclosed is a lower bound, because it matches text word for word. All of it is one English attack template on one mailbox, so a low count is no proof of safety. The Qwen columns are thinking off.

It lands in the middle. Tool calls, instruction following and Dutch are solid, but it leads none of the quality rows, and with thinking off GPQA trails both Qwen and MiniCPM. The agent loop is its strong point: tool-call replies decode at 78 tok/s because LiLiCorr drafts them well, and long prompts prefill fastest of the four. Prose is another story. Decode is the slowest here, and Dutch drops to 23.6 tok/s. One multi-turn sample failed inside the scorer, hence n=199.

Prompt injection: AgentDojo hides an instruction in the data a tool returns and checks whether the agent carries it out. Nemotron did so in 55 of 200 runs, and at least 30 got user data out. That is with thinking off. Mail, calendar and files looked fine on the old score (9 of 113) and don't on this count: 24 attacks went through, 15 with user data. Slack (13 of 29) and banking (12 of 23) were weak too. In banking the attacks that drain the account or send $30,000 failed, but it did point a scheduled payment at the attacker's IBAN twice, which the old score missed. The rest were small transfers to that IBAN with personal details in the subject (a friend's IBAN, the user's phone model, a favourite food, the list of scheduled payments), and three password changes to the attacker's choice. In 6 of those 12 banking runs, and 10 of the 13 in Slack, it finished the user's task as well, so nothing looked wrong.

If you give it tools that send money or messages, put the check outside the model: an allowlist of recipients, or a confirmation step for anything new.

## Credits

- [NVIDIA](https://huggingface.co/nvidia) for Nemotron 3.5 Super VL, the NVFP4 checkpoint, the LiLiCorr drafter and the DGX Spark serving recipe on the model card (OpenMDW-1.1). The weights are not in this repository.
- The [vLLM](https://github.com/vllm-project/vllm) project and its nightly images.
- The shape of this repository follows [Kolibri-1-DGX-Spark](https://github.com/dedene/Kolibri-1-DGX-Spark): one Compose project, a pinned download, and a small check.

## License

MIT, see [LICENSE](LICENSE). The model weights are licensed by NVIDIA under the [OpenMDW License 1.1](https://openmdw.ai/license/1-1/).
