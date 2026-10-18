# ghcr.io/sybil-solutions/deepseek-v4.1-flash-exl3-spark: the serving image of sudoingX/dgx-spark-deepseek-v41
# (dsv41-serve:v4, release v1.1) built by CI from pinned sources, for DeepSeek-V4.1-Flash EXL3 checkpoints in that
# layout (0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw, sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw) served with vLLM
# tensor parallel 2 across TWO NVIDIA DGX Spark (GB10, sm_121, arm64) over RoCE v2. OpenAI-compatible API on :8000.
#
# linux/arm64 ONLY; every compiled op targets GB10 (12.1a). Weights are NOT in the image.
#
# Stack (all pinned; sources fetched as GitHub archives with sha256 checksums):
#   base       vllm/vllm-openai v0.31.0 by digest (arm64 used): torch 2.13.0+cu130, py3.12, nvcc 13.0. Build
#              environment for the wheels and runtime base; its vLLM is replaced by the wheel built here.
#   vLLM       vllm-project/vllm main @ 6e517b15 (Apache-2.0), compiled for TORCH_CUDA_ARCH_LIST=12.1a with the
#              recipe's two source fixes (CUDA>=13 arch lists += 12.1, Hopper-only FA3 extension skipped)
#   exllamav3  turboderp-org/exllamav3 v1.6.0 @ 151539c7 (MIT), built for 12.1
#   recipe     sudoingX/dgx-spark-deepseek-v41 @ cc59c360 (Apache-2.0): patches 0001-0008, 0010-0021 applied to the
#              installed vllm package, and the vllm_exl3_moe plugin (exl3_mrow / exl3_smoe kernels, JIT-built for
#              sm_121 on first use into the mounted cache; TensorFold's RDMA ring + L2 prefetch, Apache-2.0)
# This Dockerfile reproduces the recipe's install.sh steps (scripts/build/*, scripts/image/Dockerfile{,.quant,.serve})
# in one multi-stage build; the build scripts are run unmodified from the pinned recipe archive.
FROM vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3 AS wheels
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG MAX_JOBS=4
ADD --checksum=sha256:7143390d5616bd708c21b0e868bf33449ec443a4446076aba81c017e61a45bd1 https://github.com/vllm-project/vllm/archive/6e517b15c1833cf72a7f557ee32524d98682e617.tar.gz /tmp/vllm.tgz
ADD --checksum=sha256:fb4c93f4622139a57f33c7a8b29261e86d0d3501d5470a664e1dcfef22c43c9e https://github.com/turboderp-org/exllamav3/archive/151539c77abc7ab7425d30da7a4e8e3c5c154e7b.tar.gz /tmp/x3.tgz
ADD --checksum=sha256:ff7cb8ae630a66484212ad91ee5c9ecccd00887229e1b431c526fe39899d3ef2 https://github.com/sudoingX/dgx-spark-deepseek-v41/archive/cc59c3609e8e671418f39eb92aa16b0e4f770b77.tar.gz /tmp/recipe.tgz
RUN for p in vllm x3 recipe; do mkdir -p /src/$p && tar -xzf /tmp/$p.tgz -C /src/$p --strip-components=1; done \
 && rm -f /tmp/*.tgz && mkdir -p /out /logs
# CPU-only compiles (no GPU on the runner); four nvcc jobs with 24 GB swap in the hosted release workflow.
RUN MAX_JOBS=${MAX_JOBS} NVCC_THREADS=1 VLLM_VERSION=0.31.1.dev0+g6e517b15.dsv41gb10 \
    bash /src/recipe/scripts/build/build_vllm_in_container.sh
RUN MAX_JOBS=${MAX_JOBS} bash /src/recipe/scripts/build/build_exllamav3_in_container.sh

FROM vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL org.opencontainers.image.description="DeepSeek-V4.1-Flash EXL3 (sudoingX dgx-spark-deepseek-v41 v1.1 stack): vLLM 6e517b15 + patches + vllm_exl3_moe, TP2 over 2x DGX Spark (sm_121, arm64)"
LABEL org.opencontainers.image.licenses="Apache-2.0 AND MIT"
LABEL ai.omarchy.engine="vllm"
LABEL ai.omarchy.engine.version="0.31.1.dev0+g6e517b15.dsv41gb10"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="dgx-spark"
LABEL ai.omarchy.cuda.archs="12.1a"
LABEL ai.omarchy.model.repo="0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw"
LABEL dsv41.vllm.commit="6e517b15c1833cf72a7f557ee32524d98682e617" \
      dsv41.exllamav3.commit="151539c77abc7ab7425d30da7a4e8e3c5c154e7b" \
      dsv41.recipe.repo="sudoingX/dgx-spark-deepseek-v41" dsv41.recipe.commit="cc59c3609e8e671418f39eb92aa16b0e4f770b77"
RUN command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq --no-install-recommends git && rm -rf /var/lib/apt/lists/*)
COPY --from=wheels /out/ /tmp/wheels/
COPY --from=wheels /src/recipe/ /opt/dsv41/recipe/
# scripts/image/Dockerfile: the vLLM wheel, then every numbered patch into site-packages, in order
RUN python3 -m pip install --no-deps --force-reinstall /tmp/wheels/vllm-*.whl \
 && mkdir -p /opt/dsv41/patches && cp /opt/dsv41/recipe/patches/[0-9][0-9][0-9][0-9]-*.patch /opt/dsv41/patches/ \
 && SITE=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") && cd "$SITE" \
 && for p in /opt/dsv41/patches/[0-9][0-9][0-9][0-9]-*.patch; do git apply --verbose --include='vllm/*' -p1 "$p"; done \
 && python3 -c "import vllm.models.deepseek_v41.common.engram as e; print('engram module ok')"
# build_image.sh's proof: every patch reverse-applies, in reverse order, on a scratch copy of the installed package
RUN SITE=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") \
 && mkdir -p /tmp/chk && cp -r "$SITE/vllm" /tmp/chk/ && cd /tmp/chk \
 && for p in $(ls /opt/dsv41/patches/*.patch | sort -r); do git apply -R --include="vllm/*" -p1 "$p" && echo "in image: $(basename "$p")"; done \
 && rm -rf /tmp/chk
# scripts/image/Dockerfile.quant: exllamav3 1.6.0 for sm_121 and the modules it imports
RUN python3 -m pip install --no-deps /tmp/wheels/exllamav3-1.6.0-*.whl \
 && missing=$(python3 -c "import importlib.util as u; print(' '.join(m for m in ('marisa_trie', 'llguidance', 'pydantic', 'rich', 'yaml') if u.find_spec(m) is None))") \
 && if [ -n "$missing" ]; then python3 -m pip install $(echo $missing | sed 's/yaml/pyyaml/; s/marisa_trie/marisa-trie/'); fi
# scripts/image/Dockerfile.serve: the vllm_exl3_moe plugin, --no-deps so pip never touches torch
RUN cp -r /opt/dsv41/recipe/patches/exl3_moe /tmp/plugin && cd /tmp/plugin \
 && python3 -m pip wheel --no-deps --no-build-isolation -w /tmp/wheels . >/dev/null \
 && python3 -m pip install --no-deps /tmp/wheels/vllm_exl3_moe-*.whl \
 && rm -rf /tmp/plugin /tmp/wheels /root/.cache/pip \
 && python3 -c "import importlib.metadata as m; eps = [e.name for e in m.entry_points(group='vllm.general_plugins')]; assert 'exl3_moe' in eps, eps; print('plugin entry point ok', eps)"
# The Engram disk reader and the plugin kernels JIT-build on first use into TORCH_EXTENSIONS_DIR; the launcher points
# it at the mounted cache (/root/.cache/torch_extensions), so one boot builds them and later boots reuse them.
ENV TORCH_CUDA_ARCH_LIST=12.1a
EXPOSE 8000
ENTRYPOINT ["vllm", "serve"]
