# ghcr.io/sybil-solutions/sglang-exl3-multiarch: SGLang 0.5.20 + EXL3 for Ampere, Ada and consumer Blackwell
# (sm_86, sm_89, sm_120). The CI-built, attested replacement for the hand-built
# ghcr.io/0xsero/sglang-exl3:v0.7.0-multiarch@sha256:9d95829ab41a9307e9b25c86e4af1ec52c6a02f7888f0e13a1ce93c0a7ff3814,
# which was exactly this recipe (sglang-exl3 docker/Dockerfile.multiarch) run by hand on 2026-09-27.
#
# Base is ghcr.io/0xsero/sglang-exl3 v0.7.0-ampere by digest: lmsysorg/sglang v0.5.20 (CUDA 13.0, nvcc present), the
# sglang-exl3 plugin (vendored @ a25f837), exllamav3 1.5.1 for sm_86, token maps under /opt/sglang-exl3/tokenmaps,
# the /opt/entrypoint.sh shim, ports 30000/22 and openssh. Built by this repo's release-image workflow (run
# 36210770340, attested). Everything the registry recipes reference (/opt/entrypoint.sh, /opt/sglang/bin/python,
# /opt/sglang-exl3/tokenmaps/*.pt) is inherited unchanged.
#
# This image recompiles the two native parts for the three archs:
#   - exllamav3_ext from turboderp-org/exllamav3 v1.5.1 (958ec93, the tag the hand build cloned)
#   - aikido_exl3_kernels + aikido_exl3_moe_kernels from csrc/, vendored from sybil-solutions/sglang-exl3 (private)
#     at the commit in ./VERSION by ./sync-src.sh. Its csrc tree (6fd60ad) is byte-identical to the base's a25f837
#     snapshot and to the cd0c72c commit that added Dockerfile.multiarch.
# The hand build used MAX_JOBS=44; the 4-vCPU runner uses 4 (parallelism only, same compiler inputs).
FROM ghcr.io/0xsero/sglang-exl3@sha256:c8922bd7256caf1b273af7ac49221e652be9e6072c1cdec4b959f506fd415d79

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="sglang"
LABEL ai.omarchy.engine.version="0.5.20"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.cuda.archs="8.6;8.9;12.0"
LABEL ai.omarchy.sglang-exl3.commit="926f5dd6a9d8ec134eb646466325b1f14466e581"
LABEL ai.omarchy.exllamav3.commit="958ec933361b24eb8426ec7222e5b0062a679dcd"

ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"

# ExLlamaV3 v1.5.1 at its tag commit; fail closed if the checkout is anything else.
ARG EXLLAMAV3_COMMIT=958ec933361b24eb8426ec7222e5b0062a679dcd
RUN rm -rf /opt/exllamav3-src \
 && git init -q /opt/exllamav3-src && cd /opt/exllamav3-src \
 && git fetch -q --depth 1 https://github.com/turboderp-org/exllamav3 "${EXLLAMAV3_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${EXLLAMAV3_COMMIT}" \
 && MAX_JOBS=4 pip install --no-build-isolation --no-deps --force-reinstall -v . 2>&1 | tail -3 \
 && cd / && rm -rf /opt/exllamav3-src

# The Marlin-template EXL3 dense + MoE kernels, rebuilt in place (PYTHONPATH=/opt/sglang-exl3/csrc/build/lib is set
# by the base).
COPY VERSION /opt/sglang-exl3/csrc.VERSION
COPY csrc /opt/sglang-exl3/csrc
RUN cd /opt/sglang-exl3/csrc && rm -rf build \
 && { MAX_JOBS=4 python3 setup.py build_ext --build-lib build/lib > build.log 2>&1 || { tail -60 build.log; exit 1; }; } \
 && rm -rf build/temp.* build/gen build.log \
 && ls -la build/lib

# Prove every extension imports (CPU only) and the recipe paths are still there.
RUN python3 -c "import torch, exllamav3_ext, aikido_exl3_kernels, aikido_exl3_moe_kernels; print('kernels import ok')" \
 && test -x /opt/entrypoint.sh \
 && test -s /opt/sglang-exl3/tokenmaps/qwen38_hot32k_v2.pt \
 && test -s /opt/sglang-exl3/tokenmaps/qwen36_hot32k_v2.pt

# Entrypoint (/opt/entrypoint.sh shim), ports 30000/22 and openssh are inherited from the base.
