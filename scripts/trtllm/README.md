# TensorRT-LLM inference backend (DGX Spark / GB10)

Runs the agent node's chat and embedding endpoints on TensorRT-LLM instead of
Ollama or `llama-server`, to get two things llama.cpp does not currently give
us on this box:

- **NVFP4 weights on the Blackwell FP4 tensor cores.** GB10's fifth-generation
  tensor cores execute FP4 GEMMs natively with FP8/FP16 accumulation. An NVFP4
  checkpoint (`nvidia/*-NVFP4`) is *already* 4-bit with per-block scales, so it
  loads straight into those kernels - no dequant-to-BF16 step, and the weights
  take roughly a quarter of the unified memory that BF16 would, which leaves
  the rest for KV cache.
- **PagedAttention with block reuse.** The KV cache is a pool of fixed-size
  blocks that requests borrow as they grow, so concurrent requests don't each
  reserve `max_seq_len` up front, and a request whose prefix matches an earlier
  one skips that part of prefill entirely. For the RAG loop (same system
  prompt, repeatedly retrieved chunks) that is where most of the win is.

Both servers speak the OpenAI API, so `scripts/server-chromadb-test.py` and
anything else pointed at `llama-server` keeps working unchanged.

## Requirements

- DGX Spark (GB10, `sm_121`, 128 GB unified memory), driver 580.x / CUDA 13.
- Docker with the NVIDIA container runtime.
- `HF_TOKEN` exported for gated repos.
- ~30 GB for the container image plus the weights (~18 GB for the default
  30B-A3B NVFP4 checkpoint).

## Usage

```bash
# terminal 1 - chat, OpenAI API on 127.0.0.1:8080
scripts/trtllm/serve.sh chat

# terminal 2 - embeddings, OpenAI API on 127.0.0.1:8081
scripts/trtllm/serve.sh embed

# terminal 3 - end-to-end check through ChromaDB
python3 scripts/server-chromadb-test.py
```

First start pulls the image and the weights and then builds/warms the engine;
expect several minutes cold and ~1-2 minutes warm.

Everything is environment-overridable, and trailing arguments are passed
through to `trtllm-serve`:

| Variable | Default | Notes |
| --- | --- | --- |
| `TRTLLM_IMAGE` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | aarch64 tags lag x86; check NGC before bumping |
| `CHAT_MODEL` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | any NVFP4 checkpoint, e.g. `nvidia/Qwen3-30B-A3B-NVFP4` |
| `EMBED_MODEL` | `BAAI/bge-m3` | encoder-only model; the embeddings server has no KV cache |
| `CHAT_PORT` / `EMBED_PORT` | `8080` / `8081` | published on loopback only |
| `MAX_BATCH_SIZE` | `32` | keep `cuda_graph_config.max_batch_size` in the YAML in step with it |
| `MAX_SEQ_LEN` | `32768` | raise for long-context work, it costs KV pool |
| `KV_CACHE_FRACTION` | `0.7` | share of post-weights memory given to the KV pool |
| `LLM_API_CONFIG` | `scripts/trtllm/llm-api-config.yml` | chat server only |

```bash
# example: bigger batches, FP8 KV cache to fit more concurrent sequences
MAX_BATCH_SIZE=64 scripts/trtllm/serve.sh chat --kv_cache_dtype fp8
```

`llm-api-config.yml` carries the paged-KV settings (`tokens_per_block`,
`enable_block_reuse`, `enable_partial_reuse`) and the decode CUDA graph config.
Reuse is on by default in TensorRT-LLM; it is spelled out there because it is
the reason we are running this engine.

## Checking that it is doing what we think

```bash
curl -s localhost:8080/v1/models                 # served model id
curl -s localhost:8080/metrics                   # request/latency counters
curl -s localhost:8080/kv_cache_events           # block store/reuse events
```

Send the same prompt prefix twice and watch time-to-first-token drop on the
second call - that is block reuse. If it doesn't, the prefix is probably not
block-aligned (`tokens_per_block: 32`) or reuse got disabled.

For throughput numbers use `trtllm-bench` inside the container, and compare
against the `llama-bench` / `llama-batched-bench` baselines already recorded in
`benches/dgx-spark/dgx-spark.md` (gpt-oss-20b MXFP4: ~4.5k t/s prefill, ~83 t/s
single-stream decode). Compare like for like: same prompt/output lengths, same
concurrency, and note the quantization, since the two engines are not running
the same weights.

## Caveats

- NVFP4 support on GB10 is newer than on datacenter Blackwell and is
  per-architecture: some checkpoints that load on B200 still fail on `sm_121`
  with weight-scale or tokenizer errors. If a model won't load, try another
  NVFP4 checkpoint or a `1.3.0rc*` bump before assuming the setup is wrong.
- FP4 is not free accuracy-wise. Spot-check output quality on our own prompts
  before treating a swap as a pure win.
- gpt-oss checkpoints are MXFP4, not NVFP4 - different scale format, so the
  comparison against the llama.cpp numbers above is engine *and* weights.
- The container is single-model-per-process, same as `llama-server`, which is
  why chat and embeddings are two commands.
- `llama-server` stays the fallback: the test script talks to either engine, so
  a broken container upgrade means changing two ports, not the pipeline.
