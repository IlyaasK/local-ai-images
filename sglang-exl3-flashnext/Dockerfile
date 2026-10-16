# ghcr.io/<owner>/sglang-exl3-flashnext: Qwen3.8-Flash-Next EXL3 on every RTX 30-series card (sm_86), with two-tier
# offload: routed experts in pinned host RAM behind a GPU expert cache, and the 32.6 GB EXL3 n-gram table on NVMe
# (buffered reads, page cache as elastic L2) behind a RAM row cache. The serving code is trellis-serve
# (github.com/sybil-solutions/trellis-serve, branch rtx30-offload) at 2e27422: the Phase D build (b1e74af) plus
#   - rtx30-12g-3bpw (b6db01f): chunked expert-cache arena with elastic flex chunks (SGLANG_EXL3_OFFLOAD_ELASTIC_GB:
#     handed back to torch for each prefill, re-taken at decode), and the AVX2 CPU expert tier for K=3 decode misses
#     (SGLANG_EXL3_OFFLOAD_CPU_TIER=1), so 3.05bpw reaches 25+ tok/s on 12 GB cards over PCIe3;
#   - rtx30-2bpw-k2 (45724ba): K=2 experts in the grouped Marlin MoE and offload kernels, so the 2.05bpw quant
#     offloads on 8-10 GB cards (the CPU tier stays K=3 only and switches itself off for K=2 stores).
# Earlier tags of this image (digest ce47df25..., trellis-serve 1e72b25) stay published for the recipes pinned to them.
#
# Base is ghcr.io/0xsero/sglang-exl3 v0.7.0-ampere by digest: lmsysorg/sglang v0.5.20 (CUDA 13.0, torch 2.13, nvcc
# present) + exllamav3 1.5.1 (6b84a21, = v1.5.1 plus an MSVC-only fix) built for sm_86, shim entrypoint, openssh.
# Built by this repo's release-image workflow (run 36210770340, attested). The campaign's dev image had the same
# SGLang build (commit 94602c9) and exllamav3 v1.5.1 on the same lmsysorg base.
#
# This image adds trellis-serve at the pinned commit and compiles its kernels AOT for sm_86:
#   - trellis_exl3_kernels (dense Marlin-template EXL3 GEMMs, MUL1 codebook, K = 3/4/5/6)
#   - trellis_exl3_moe_kernels (grouped MoE + GPU expert cache, K = 2/3/4, fp16-accumulate prefill variants)
#   - trellis_offload_kernels, trellis_cache_chunked, trellis_cpu_tier_dev (the offload runtime's JIT extensions,
#     pre-built into the JIT cache)
# Two host-side modules are compiled with -march=native, so they are deliberately NOT pre-built here: the n-gram NVMe
# row store (csrc/offload/ngram_nvme.cu, ~1 min) and the CPU tier's AVX2 pool (cpu_tier_host.cpp, C++ only, only when
# SGLANG_EXL3_OFFLOAD_CPU_TIER=1). They JIT-compile on the serving host at first start (nvcc + ninja are in the image)
# and are cached in SGLANG_EXL3_JIT_DIR for the life of the container.
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance, an SBOM, and a GitHub build
# attestation on the digest.
FROM ghcr.io/0xsero/sglang-exl3@sha256:c8922bd7256caf1b273af7ac49221e652be9e6072c1cdec4b959f506fd415d79

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="sglang"
LABEL ai.omarchy.engine.version="0.5.20"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="rtx-30-series"
LABEL ai.omarchy.cuda.archs="8.6"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-Flash-Next-exl3"
LABEL ai.omarchy.model.revision="69e33439ae950f17bcbe95c98f117d80f759ab6d"
# the 2.05bpw quant (K=2 experts) for 8-10 GB cards
LABEL ai.omarchy.model.revision.2bpw="65c895314393431c09050b2e04e250836b3a6eb4"

ARG TRELLIS_SERVE_REPO=https://github.com/sybil-solutions/trellis-serve
ARG TRELLIS_SERVE_COMMIT=2e2742258fefe2ff4fbe9e52faea58e9f9bcd212
LABEL ai.omarchy.trellis-serve.commit="2e2742258fefe2ff4fbe9e52faea58e9f9bcd212"
ENV TORCH_CUDA_ARCH_LIST=8.6
# Phase D server defaults (INT058b): 5-bit exllamav3-quantized QSA KV (9,216 B/token, 209,984 tokens at 204,800
# context) and an 8.26 GB reserve for the expert-cache auto-fit (11.08 GB of experts on a 24 GB card). Recipes also
# pass these explicitly, with --chunked-prefill-size 12288.
ENV SGLANG_EXL3_KV_BITS=5 \
    SGLANG_EXL3_EXPERT_CACHE_RESERVE_GB=8.26

# Source at the pinned commit; fail closed if the checkout is anything else.
RUN git init -q /opt/trellis-serve \
 && cd /opt/trellis-serve \
 && git fetch -q --depth 1 "${TRELLIS_SERVE_REPO}" "${TRELLIS_SERVE_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${TRELLIS_SERVE_COMMIT}" \
 && echo "${TRELLIS_SERVE_COMMIT}" > /opt/trellis-serve/COMMIT \
 && rm -rf .git

# The trellis-serve `sglang-exl3` distribution replaces the base image's plugin of the same name (same entry point,
# `sglang.srt.plugins: exl3`); exllamav3 stays (the n-gram and CPU-worker paths import it).
RUN pip install -q --no-deps --no-build-isolation -e /opt/trellis-serve/core -e /opt/trellis-serve/cuda \
 && python3 -c "from importlib.metadata import entry_points as e, distribution as d; \
assert any(x.value == 'sglang_exl3.plugin:activate' for x in e(group='sglang.srt.plugins')); \
print('sglang-exl3 dist at', d('sglang-exl3').read_text('direct_url.json'))"

# Kernels, exactly the campaign's two build.sh invocations (dense, then MoE only), MUL1 codebook. The MoE build's
# default TRELLIS_MOE_BITS is 2,3,4 at this commit; it is passed explicitly so a default change cannot drop K=2.
# build.sh never exits non-zero; its log ends in BUILD_OK or BUILD_FAILED, so check the log.
RUN cd /opt/trellis-serve/cuda/csrc \
 && export MAX_JOBS=4 \
 && TRELLIS_CODEBOOKS=2 TRELLIS_MOE=0 ./build.sh \
 && { grep -q BUILD_OK build.log || { tail -60 build.log; exit 1; }; } \
 && TRELLIS_MOE=only TRELLIS_CODEBOOKS=2 TRELLIS_MOE_BITS=2,3,4 ./build.sh \
 && { grep -q BUILD_OK build.log || { tail -60 build.log; exit 1; }; } \
 && rm -rf build/temp.* build/gen build.log \
 && ls -la build/lib

ENV PYTHONPATH=/opt/trellis-serve/cuda/csrc/build/lib \
    SGLANG_EXL3_JIT_DIR=/opt/trellis-serve/cuda/csrc/build/jit

# Pre-build the offload runtime's JIT extensions (no host-specific flags): offload kernels, the chunked cache arena
# and the CPU tier's split/combine kernels, then prove every AOT module imports (CPU only: no GPU is needed to build
# or load the extensions).
RUN python3 -c "import torch, trellis_exl3_kernels, trellis_exl3_moe_kernels, importlib; \
importlib.import_module('sglang_exl3.offload._ext').load(); \
importlib.import_module('sglang_exl3.kernels.offload_moe')._cmod(); \
importlib.import_module('sglang_exl3.offload.cpu_tier').load_dev(); print('trellis-serve kernels ok')" \
 && ls /opt/trellis-serve/cuda/csrc/build/jit

# Entrypoint (/opt/entrypoint.sh shim), ports 30000/22 and openssh are inherited from the base.
