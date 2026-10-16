# exl3-rdna4: ExLlamaV3 with the RDNA4 patches + TabbyAPI for the RX 9070 / 9070 XT (gfx1201, 16 GB). Serves
# Qwen3.8-27B EXL3 (SC 3.0bpw H4) with the DFlash2 draft at 96k context (128k with CTX=128k); the models are mounted
# at /models, not baked in.
#
# Same build as the recipe's own container/Dockerfile, at a pinned recipe commit: the build stage runs the recipe's
# setup scripts on AMD's ROCm dev image, which pin ExLlamaV3 (turboderp-org/exllamav3 0f685aa9, plus
# patches/exllamav3-rdna4.patch), TabbyAPI (theroyallab/tabbyAPI a3fed43e) and torch 2.13.0+rocm7.2 (setup/versions.sh,
# setup/requirements*.txt). The runtime stage needs no ROCm: the extension resolves its ROCm libraries to the copies
# the PyTorch ROCm wheel loads first, which carry the same sonames. It keeps gcc and the Python headers because Triton
# compiles a small C helper at runtime. No GPU is needed to build.
#
# Recipe: https://github.com/parth-thakre/exl3-rdna4-recipe (container/README.md covers running it by hand).
FROM docker.io/rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054 AS build
ARG RECIPE_COMMIT=6ffe066e0958ba35a0f699192c159f516d31746b
# Two compile jobs fit a GitHub-hosted runner's RAM
ARG MAX_JOBS=2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ python3.12-venv python3.12-dev git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/recipe
RUN git init -q . && git remote add origin https://github.com/parth-thakre/exl3-rdna4-recipe.git \
    && git fetch -q --depth 1 origin "$RECIPE_COMMIT" && git -c advice.detachedHead=false checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$RECIPE_COMMIT"
# Only the ROCm -dev packages the build needs (setup/fetch_deps.sh lists them), not all of rocm-hip-sdk
RUN apt-get update \
    && pkgs=$(NO_FETCH=1 setup/fetch_deps.sh | sed -n 's/^  sudo apt install //p') \
    && { [ -z "$pkgs" ] || apt-get install -y --no-install-recommends $pkgs; } \
    && rm -rf /var/lib/apt/lists/*
RUN setup/fetch_deps.sh \
    && setup/make_venv.sh \
    && MAX_JOBS=$MAX_JOBS setup/build_exllamav3.sh \
    && setup/install_tabby.sh \
    && cp container/config.yml tabbyAPI/config.yml \
    && rm -rf exllamav3/build logs /root/.cache \
    && git rev-parse HEAD > container/RECIPE_COMMIT

FROM docker.io/library/ubuntu:24.04@sha256:f610ab94648195aa356059f5b41d6085c9d4d903c072430cdd1af7bdb646106b
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-dev gcc libc6-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/recipe /opt/recipe
WORKDIR /opt/recipe
LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
# Any uid can run it (the plugin may not run it as root): the log directories and Triton's kernel cache must be
# writable. TabbyAPI writes logs/{time}.log relative to its own working directory (tabbyAPI/), not /opt/recipe/logs,
# so that directory has to exist too - the API server exits at startup without it.
RUN mkdir -p logs tabbyAPI/logs && chmod 1777 logs tabbyAPI/logs
ENV TRITON_CACHE_DIR=/tmp/triton HOME=/tmp
# Models: /models (Qwen3.8-27B-EXL3-SC3.0bpw-H4 and draft/Qwen3.8-27B-DFlash2-EXL3-3.0bpw), read-only is fine
ENV EXL3_MODELS=/models
EXPOSE 8096
# Checks the models are mounted, then starts TabbyAPI (container/config.yml: port 8096, no API key)
ENTRYPOINT ["/opt/recipe/container/entrypoint.sh"]
