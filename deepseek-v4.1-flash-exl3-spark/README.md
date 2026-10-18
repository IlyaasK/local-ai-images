# deepseek-v4.1-flash-exl3-spark

`ghcr.io/sybil-solutions/deepseek-v4.1-flash-exl3-spark`: the serving stack of
[sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41) v1.1 (`dsv41-serve:v4`),
built by this repository's release workflow from pinned sources. It serves DeepSeek-V4.1-Flash EXL3 checkpoints in
that project's layout with vLLM tensor parallel 2 across **two NVIDIA DGX Spark** (GB10, `sm_121`, arm64) over RoCE v2:

- [`0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw`](https://huggingface.co/0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw)
  (DeepSeek-V4.1-Flash-1017; recipe: [sybil-solutions/deepseek-v4.1-flash-1017-exl3-spark](https://github.com/sybil-solutions/deepseek-v4.1-flash-1017-exl3-spark))
- [`sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw`](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw)
  (DeepSeek-V4.1-Flash `2cba9e42`)

OpenAI-compatible API on :8000 with `deepseek_v41` reasoning and tool-call parsing. **linux/arm64 only.** Weights are
not in the image.

## Stack

| component | pin | licence |
|---|---|---|
| base `vllm/vllm-openai` v0.31.0 (arm64 from the multi-arch index): torch 2.13.0+cu130, py3.12, nvcc 13.0 | `sha256:c1c9f6fd…7c3` | Apache-2.0; NVIDIA CUDA/NCCL runtime under NVIDIA's licences |
| vLLM main, compiled for `12.1a` | `6e517b15c1833cf72a7f557ee32524d98682e617` | Apache-2.0 |
| exllamav3 v1.6.0, compiled for `12.1` | `151539c77abc7ab7425d30da7a4e8e3c5c154e7b` | MIT (turboderp) |
| sudoingX/dgx-spark-deepseek-v41: patches 0001-0008 and 0010-0021, `vllm_exl3_moe` plugin | `cc59c3609e8e671418f39eb92aa16b0e4f770b77` | Apache-2.0 (includes TensorFold code, Apache-2.0) |

The Dockerfile does what the recipe's `install.sh` does on a Spark, in one multi-stage build:

1. The `wheels` stage unpacks the three source archives (`ADD --checksum=sha256:`) and runs the recipe's own
   `scripts/build/build_vllm_in_container.sh` and `build_exllamav3_in_container.sh` unmodified. The first makes the
   two GB10 source fixes: 12.1 added to the CUDA>=13 arch lists, and the Hopper-only FA3 extension skipped.
2. The final stage installs the vLLM wheel over the base's vLLM and applies every numbered patch to the installed
   package (`scripts/image/Dockerfile`). It then proves the series by reverse-applying it on a scratch copy
   (`build_image.sh`), installs exllamav3 (`Dockerfile.quant`), and builds and installs the plugin with its
   entry-point check (`Dockerfile.serve`).

The pinned recipe tree is kept at `/opt/dsv41/recipe` (launcher, bench scripts, `data/dspark/markov_top_tokens.json`).
The plugin's kernels and the Engram disk reader JIT-build for `sm_121` on first use into `TORCH_EXTENSIONS_DIR`. The
launcher mounts `~/.cache/dsv41-serve` at `/root/.cache` and points it there, so the first boot builds them and later
boots reuse them.

## Run

Use the recipe's launcher, which starts one container per Spark and wires NCCL to the RoCE HCA and GID that carry each
node's fabric IPv4:

```bash
git clone https://github.com/sudoingX/dgx-spark-deepseek-v41 ~/dsv41_recipe && git -C ~/dsv41_recipe checkout cc59c360
IMAGE=ghcr.io/sybil-solutions/deepseek-v4.1-flash-exl3-spark@sha256:<digest> \
HEAD=<rank-0 host> WORKER=<rank-1 host> MODEL_DIR=/same/path/on/both \
  ~/dsv41_recipe/scripts/launch/dsv41_tp2.sh up
```

The registry recipe `vllm-deepseek-v4.1-flash-1017-exl3-2.7bpw-spark-256k-2x` lists the same flags and environment as
one launch.

## Build

Run `release-image.yml` with `image=deepseek-v4.1-flash-exl3-spark`, a tag, and `platform=linux/arm64`. It builds on
`ubuntu-24.04-arm` with 24 GB of swap, `MAX_JOBS=4`, and a 6 h limit. No GPU is needed to compile. GPU model
acceptance is a separate gate: the registry recipe's run on two Sparks.

Check without weights, on a Spark:

    docker run --rm --gpus all --entrypoint python3 IMAGE -c "import vllm, exllamav3, vllm_exl3_moe; print('ok')"
