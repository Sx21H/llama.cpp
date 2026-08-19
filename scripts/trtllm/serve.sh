#!/usr/bin/env bash
#
# Start the TensorRT-LLM OpenAI-compatible servers the agent node talks to.
#
#   ./serve.sh chat     # /v1/chat/completions on $CHAT_PORT  (NVFP4 weights)
#   ./serve.sh embed    # /v1/embeddings      on $EMBED_PORT
#
# Both run in the foreground inside the NGC TensorRT-LLM container, so use two
# terminals (or two `docker run -d` wrappers) when you want both up. Ports are
# published on 127.0.0.1 only.
#
# Everything below is overridable from the environment, e.g.
#   CHAT_MODEL=nvidia/Qwen3-30B-A3B-NVFP4 MAX_BATCH_SIZE=64 ./serve.sh chat
# Trailing arguments are appended to the trtllm-serve command line:
#   ./serve.sh chat --kv_cache_dtype fp8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Container. 1.3.0rc* is where the DGX Spark (GB10, sm_121) kernels landed;
# check the NGC tag list before bumping, aarch64 images lag x86 ones.
TRTLLM_IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# NVFP4 checkpoint: the weights are already 4-bit, so they load straight onto
# the Blackwell FP4 tensor cores with no quantization step at startup.
CHAT_MODEL="${CHAT_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
CHAT_PORT="${CHAT_PORT:-8080}"

# Encoder-only model; the embeddings server runs the encode() path, no KV cache.
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"
EMBED_PORT="${EMBED_PORT:-8081}"

MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
KV_CACHE_FRACTION="${KV_CACHE_FRACTION:-0.7}"
LLM_API_CONFIG="${LLM_API_CONFIG:-$SCRIPT_DIR/llm-api-config.yml}"

usage() {
    echo "usage: $(basename "$0") chat|embed [extra trtllm-serve args...]" >&2
    exit 2
}

[ $# -ge 1 ] || usage
MODE="$1"
shift

case "$MODE" in
    chat)
        PORT="$CHAT_PORT"
        NAME="trtllm-chat"
        MOUNT_CONFIG=(-v "$LLM_API_CONFIG:/workspace/llm-api-config.yml:ro")
        SERVE_CMD=(
            trtllm-serve "$CHAT_MODEL"
            --host 0.0.0.0 --port "$PORT"
            --max_batch_size "$MAX_BATCH_SIZE"
            --max_seq_len "$MAX_SEQ_LEN"
            --kv_cache_free_gpu_memory_fraction "$KV_CACHE_FRACTION"
            --extra_llm_api_options /workspace/llm-api-config.yml
        )
        ;;
    embed)
        PORT="$EMBED_PORT"
        NAME="trtllm-embed"
        # The encode path allocates no KV cache, so the chat config (which is
        # all KV cache and decode CUDA graphs) does not apply here.
        MOUNT_CONFIG=()
        SERVE_CMD=(
            trtllm-serve embeddings "$EMBED_MODEL"
            --host 0.0.0.0 --port "$PORT"
            --max_batch_size "$MAX_BATCH_SIZE"
        )
        ;;
    *)
        usage
        ;;
esac

# -t only when there is a terminal, so this also works from a service unit
TTY_FLAGS=(-i)
if [ -t 1 ]; then
    TTY_FLAGS=(-it)
fi

DOCKER_CMD=(
    docker run --rm "${TTY_FLAGS[@]}"
    --name "$NAME"
    --gpus all --ipc=host --shm-size=8g
    -v "$HF_CACHE:/root/.cache/huggingface"
    "${MOUNT_CONFIG[@]}"
    -p "127.0.0.1:$PORT:$PORT"
)
# gated repos need the token; pass it through only when it is actually set
if [ -n "${HF_TOKEN:-}" ]; then
    DOCKER_CMD+=(-e HF_TOKEN)
fi
DOCKER_CMD+=("$TRTLLM_IMAGE" "${SERVE_CMD[@]}" "$@")

printf '+ %q ' "${DOCKER_CMD[@]}" >&2
printf '\n' >&2
exec "${DOCKER_CMD[@]}"
