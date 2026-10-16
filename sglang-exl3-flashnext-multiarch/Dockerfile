# ghcr.io/<owner>/sglang-exl3-flashnext-multiarch: Qwen3.8-Flash-Next EXL3 with two-tier expert offload on RTX 30-, 40-
# and 50-series cards (sm_86, sm_89, sm_120). The same serving code as sglang-exl3-flashnext (trellis-serve
# rtx30-offload @ 2e27422: elastic GPU expert cache over pinned host RAM, the AVX2 CPU tier for K=3 decode misses, K=2
# offload kernels for the 2.05bpw quant, the n-gram table on NVMe behind a RAM row cache), compiled for three archs.
# sglang-exl3-flashnext (sm_86 only, digest 9d137b00...) stays published and unchanged for the 30-series recipes.
#
# Base is ghcr.io/sybil-solutions/sglang-exl3-multiarch v0.7.0-multiarch-ci by digest (this repo's release-image
# workflow, attested): lmsysorg/sglang v0.5.20 (CUDA 13.0, torch 2.13, nvcc present) with exllamav3 v1.5.1's
# exllamav3_ext recompiled for sm_86/sm_89/sm_120. exllamav3_ext is used at runtime (QSA KV quantization, n-gram and
# CPU-worker paths), which is why the sm_86-only base of sglang-exl3-flashnext cannot simply be rebuilt with more
# archs. Turing (sm_75) and older are out: exllamav3 v1.5.1 has no sm_75 path.
#
# This image adds trellis-serve at the pinned commit and compiles, for every arch in TORCH_CUDA_ARCH_LIST:
#   - trellis_exl3_kernels (dense Marlin-template EXL3 GEMMs, MUL1 codebook, K = 3/4/5/6)
#   - trellis_exl3_moe_kernels (grouped MoE + GPU expert cache, K = 2/3/4, fp16-accumulate prefill variants)
#   - trellis_offload_kernels, trellis_cache_chunked, trellis_cpu_tier_dev (offload runtime JIT extensions, pre-built
#     into the JIT cache)
# The two -march=native host modules (n-gram NVMe row store, CPU tier AVX2 pool) JIT-compile on the serving host at
# first start, as in sglang-exl3-flashnext. The build fails unless every .so carries SASS for all three archs.
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance, an SBOM, and a GitHub build
# attestation on the digest.
FROM ghcr.io/sybil-solutions/sglang-exl3-multiarch@sha256:7736acc974e8e5d40b8edc968d6836a28f5fb0827bd655c2169b526710367ae8

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="sglang"
LABEL ai.omarchy.engine.version="0.5.20"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="rtx-30-40-50-series"
LABEL ai.omarchy.cuda.archs="8.6;8.9;12.0"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-Flash-Next-exl3"
LABEL ai.omarchy.model.revision="69e33439ae950f17bcbe95c98f117d80f759ab6d"
LABEL ai.omarchy.model.revision.2bpw="65c895314393431c09050b2e04e250836b3a6eb4"

ARG TRELLIS_SERVE_REPO=https://github.com/sybil-solutions/trellis-serve
ARG TRELLIS_SERVE_COMMIT=2e2742258fefe2ff4fbe9e52faea58e9f9bcd212
LABEL ai.omarchy.trellis-serve.commit="2e2742258fefe2ff4fbe9e52faea58e9f9bcd212"
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"
ENV SGLANG_EXL3_KV_BITS=5 \
    SGLANG_EXL3_EXPERT_CACHE_RESERVE_GB=8.26

RUN git init -q /opt/trellis-serve \
 && cd /opt/trellis-serve \
 && git fetch -q --depth 1 "${TRELLIS_SERVE_REPO}" "${TRELLIS_SERVE_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${TRELLIS_SERVE_COMMIT}" \
 && echo "${TRELLIS_SERVE_COMMIT}" > /opt/trellis-serve/COMMIT \
 && rm -rf .git

RUN pip install -q --no-deps --no-build-isolation -e /opt/trellis-serve/core -e /opt/trellis-serve/cuda \
 && python3 -c "from importlib.metadata import entry_points as e, distribution as d; \
assert any(x.value == 'sglang_exl3.plugin:activate' for x in e(group='sglang.srt.plugins')); \
print('sglang-exl3 dist at', d('sglang-exl3').read_text('direct_url.json'))"

# Same two build.sh invocations as sglang-exl3-flashnext; build.sh keeps TORCH_CUDA_ARCH_LIST from the environment.
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

RUN python3 -c "import torch, exllamav3_ext, trellis_exl3_kernels, trellis_exl3_moe_kernels, importlib; \
importlib.import_module('sglang_exl3.offload._ext').load(); \
importlib.import_module('sglang_exl3.kernels.offload_moe')._cmod(); \
importlib.import_module('sglang_exl3.offload.cpu_tier').load_dev(); print('trellis-serve kernels ok')" \
 && ls /opt/trellis-serve/cuda/csrc/build/jit

# Fail closed unless every compiled extension (AOT and pre-built JIT) carries SASS for sm_86, sm_89 and sm_120.
RUN set -e; n=0; \
 for f in $(find /opt/trellis-serve/cuda/csrc/build -name '*.so') \
          $(python3 -c "import torch, exllamav3_ext; print(exllamav3_ext.__file__)"); do \
   archs=$(cuobjdump --list-elf "$f" 2>/dev/null | grep -oE 'sm_[0-9]+a?' | sort -u | tr '\n' ' '); \
   echo "$(basename "$f"): $archs"; \
   for sm in sm_86 sm_89 sm_120; do case " $archs" in *" $sm "*) ;; *) echo "missing $sm in $f"; exit 1;; esac; done; \
   n=$((n+1)); \
 done; test "$n" -ge 6

# Entrypoint (/opt/entrypoint.sh shim), ports 30000/22 and openssh are inherited from the base.
