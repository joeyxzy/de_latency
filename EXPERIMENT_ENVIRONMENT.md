# de_latency Experimental Environment

Last updated: 2026-04-13
Workspace: `/home/joeyxzy/de_latency`
Git branch: `main`
Git commit: `a4057b24613773034f9f5784709109b9fa8c4b57`

## 1. Project positioning

This repository is a joint tracing tool for the vLLM service path. It combines three data paths:

1. Python monkey patch / `sitecustomize` events
2. CUPTI-based GPU events from `libtracer.so`
3. eBPF-based scheduler events from `ebpf_monitor`

The merged log is converted by `perfetto/log_to_trace.py` into a Perfetto-viewable trace.

Important project components:

- `build/controller`: starts the collector, injects `libtracer.so`, launches the target process, and starts `ebpf_monitor`
- `build/libtracer.so`: CUPTI tracer injected through `LD_PRELOAD`
- `build/ebpf_monitor`: eBPF-based scheduling monitor
- `src/FINAL/collector.py`: merges the three event streams
- `perfetto/log_to_trace.py`: converts merged logs into Perfetto traces
- `vllm_trace/start_server.py`: tracing-aware vLLM startup entry

## 2. Machine information

Host / server model:

- Vendor: `New H3C Technologies Co., Ltd.`
- Product: `H3C UniServer R5300 G6`
- Hostname: `dell-H3C-UniServer-R5300-G6`

Operating system:

- OS: `Ubuntu 22.04.5 LTS (Jammy Jellyfish)`
- Kernel: `6.8.0-65-generic`
- Architecture: `x86_64`

CPU:

- Model: `Intel(R) Xeon(R) Platinum 8558`
- Sockets: `2`
- Cores per socket: `48`
- Threads per core: `2`
- Total logical CPUs: `192`
- Max frequency reported by `lscpu`: `4000 MHz`
- NUMA nodes: `2`

Memory:

- Total RAM: `1.0 TiB`
- Available RAM at collection time: `976 GiB`
- Swap: `2.0 GiB`

Disk:

- Root filesystem: `/dev/sda2`
- Capacity: `439 GiB`
- Used: `285 GiB`
- Available: `132 GiB`

## 3. GPU / CUDA / CUPTI status

Physical GPU visibility:

- `lspci` can enumerate multiple NVIDIA devices on this host
- At least 5 NVIDIA PCIe functions are visible, plus an ASPEED management display device
- Exact GPU product names could not be confirmed from the runtime because the driver stack is not healthy enough for `nvidia-smi`

Observed driver/runtime status:

- `nvidia-smi` fails with: cannot communicate with the NVIDIA driver
- In the `vllm` conda environment, PyTorch reports:
  - `torch.__version__ = 2.8.0+cu128`
  - `torch.version.cuda = 12.8`
  - `torch.cuda.is_available() = False`
  - `torch.cuda.device_count() = 0`
- Conclusion: this host has CUDA toolkit files installed, but the NVIDIA driver / NVML runtime is currently unavailable or not functioning correctly in the current execution context

CUDA / CUPTI installation:

- Default CUDA symlink: `/usr/local/cuda -> /usr/local/cuda-12.8`
- `nvcc` version: `12.8.93`
- `cupti.h`: `/usr/local/cuda-12.8/targets/x86_64-linux/include/cupti.h`
- `libcupti.so`: `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so`

Note:

- The repository `Makefile` defaults to `CUDA_INSTALL_PATH=/usr/local/cuda-12.8`
- The repository README also documents CUDA 12.8 as the validated default

## 4. Python / Conda environment

System Python:

- Interpreter: `/usr/bin/python3`
- Version: `Python 3.10.12`
- `pip` version: `22.0.2`

Conda:

- Conda installation: `/home/joeyxzy/miniforge3/condabin/conda`
- Available environments:
  - `base`
  - `vllm`

Active experimental Python environment:

- Conda env name: `vllm`
- Python version: `3.12.12`

Key packages installed in `conda run -n vllm ...`:

- `vllm==0.11.0`
- `torch==2.8.0+cu128`
- `openai==2.30.0`
- `pyzmq==27.1.0`
- `numpy==2.2.6`
- `transformers==4.57.6`
- `tokenizers==0.22.2`
- `triton==3.4.0`
- `xformers==0.0.32.post1`
- `fastapi==0.135.2`
- `uvicorn==0.42.0`
- `pydantic==2.12.5`
- `aiohttp==3.13.3`
- `sentencepiece==0.2.1`
- `pandas` is not installed
- `datasets` is not installed

Important distinction:

- The README lists a validated package set for the project
- The current `vllm` environment is close to that set, but not identical
- Notable differences between README and current environment:
  - `openai`: README says `2.6.1`, current env is `2.30.0`
  - `transformers`: README says `4.57.1`, current env is `4.57.6`
  - `tokenizers`: README says `0.22.1`, current env is `0.22.2`
  - `fastapi`: README says `0.121.0`, current env is `0.135.2`
  - `uvicorn`: README says `0.38.0`, current env is `0.42.0`
  - `pydantic`: README says `2.12.3`, current env is `2.12.5`
  - `aiohttp`: README says `3.13.2`, current env is `3.13.3`

## 5. Build toolchain and native dependencies

Compiler / build tools:

- `gcc --version`: `12.3.0`
- `clang --version`: `14.0.0`
- `make --version`: `4.3`
- `bpftool version`: `v7.4.0`
- `bpftool` reports `using libbpf v1.4`

Native library versions observed from the system:

- `libzmq` via `pkg-config`: `4.3.6`
- `json-c` via `pkg-config`: `0.18.99`
- Runtime libraries visible via `ldconfig`:
  - `libbpf.so.0`
  - `libelf.so.1`
  - `libjson-c.so.5`

Package manager note:

- `pkg-config` can resolve `libzmq` and `json-c`
- `pkg-config` does not currently resolve `libbpf.pc`, even though `bpftool` and runtime `libbpf` are present
- In practice, the repository `Makefile` already has fallback search logic for `libbpf.so`

## 6. Repository build status

Built artifacts currently present in `build/`:

- `build/controller` (`31K`)
- `build/ebpf_monitor` (`62K`)
- `build/libtracer.so` (`1.2M`)

Artifact type summary:

- `build/controller`: ELF 64-bit PIE executable, dynamically linked, with debug info
- `build/ebpf_monitor`: ELF 64-bit PIE executable, dynamically linked, with debug info
- `build/libtracer.so`: ELF 64-bit shared object, dynamically linked, with debug info

This indicates the native components have already been built successfully at least once on this machine.

## 7. Project-defined recommended environment

According to the repository README, the validated baseline is:

- Ubuntu-like Linux environment
- Build packages:
  - `build-essential`
  - `make`
  - `gcc`
  - `g++`
  - `clang`
  - `bpftool`
  - `pkg-config`
  - `libzmq3-dev`
  - `libjson-c-dev`
  - `libbpf-dev`
  - `libelf-dev`
  - `zlib1g-dev`
- CUDA / CUPTI baseline:
  - `CUDA_INSTALL_PATH=/usr/local/cuda-12.8`
  - `CUPTI_INSTALL_PATH=$CUDA_INSTALL_PATH/extras/CUPTI`
- Recommended Python environment:
  - `conda create -n vllm python=3.12`
- Validated core Python package set in README:
  - `vllm==0.11.0`
  - `torch==2.8.0`
  - `openai==2.6.1`
  - `pyzmq==27.1.0`
  - `numpy==2.2.6`
  - `transformers==4.57.1`
  - `tokenizers==0.22.1`
  - `triton==3.4.0`
  - `xformers==0.0.32.post1`
  - `fastapi==0.121.0`
  - `uvicorn==0.38.0`
  - `pydantic==2.12.3`
  - `aiohttp==3.13.2`
  - `sentencepiece==0.2.1`

## 8. Current reproducibility notes

If this environment is presented externally, the following description is accurate:

- The project codebase and native binaries are already in place and built on the current host
- The Python serving environment is managed through Conda, specifically the `vllm` environment
- CUDA 12.8 and CUPTI files are installed locally
- The current blocker for end-to-end GPU execution is the NVIDIA driver / NVML state, not the absence of CUDA toolkit files

Recommended wording for external introduction:

"The experiments are conducted on an H3C UniServer R5300 G6 server running Ubuntu 22.04.5 LTS with dual Intel Xeon Platinum 8558 processors, 192 logical CPUs, and 1 TiB RAM. The tracing stack combines vLLM 0.11.0, PyTorch 2.8.0+cu128, CUDA 12.8, CUPTI-based GPU tracing, and eBPF-based scheduler tracing, with Perfetto used for trace visualization. The code and native tracing binaries are already built on the host. At the time of environment collection, the CUDA toolkit was installed but the NVIDIA driver runtime was not healthy enough for `nvidia-smi` or PyTorch CUDA enumeration."

## 9. Current repo state note

The working tree is not fully clean at collection time. There are modified, deleted, and untracked files under:

- `.gitignore`
- `perfetto/`
- `vllm_trace/__pycache__/`
- `.codex`

This does not prevent describing the environment, but it is worth recording if strict reproducibility is required.
