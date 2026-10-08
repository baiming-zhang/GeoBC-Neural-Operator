# Replaying the WSL tiny-cuda-nn build

The recorded environment is Python 3.12.3, PyTorch `2.14.1+cu130`, CUDA runtime
`13.0.96`, Ubuntu 24.04 and GCC 13.3. The official binding is tiny-cuda-nn 2.0,
built for RTX 5090 `sm_120` from commit
`3daa6e5b8b8e0fa17971a1a20acfbb3754494b38`.

Run from this repository inside WSL with the target venv's Python:

```bash
/home/research/.venvs/geobc/bin/python tools/install_tcnn_wsl.py --dry-run
```

Dry run is also the default. It reads package metadata and prints the full
build plan without installing packages, changing links or touching the
checkout. To build in a prepared venv where TCNN is absent:

```bash
/path/to/venv/bin/python tools/install_tcnn_wsl.py --install
```

The script verifies the recorded Torch/runtime before any installation. It
does not install Torch, change the runtime or install a GPU driver. An already
installed TCNN 2.0 skips all installation/build mutations. Use a separate venv
when recreating the build. Host prerequisites are `git`, `g++` and `ninja`.
`--source`, `--cuda-home`, `--python`, `--jobs` and `--log` can override the
locations, interpreter, CPU concurrency and log destination. Defaults infer
the owning directory from the target venv; the original locations were
`/home/research/src/tiny-cuda-nn` and `/home/research/.local/cuda-13.0-tcnn`.

`requirements-lock-wsl.txt` records the actual Python environment. TCNN's
local `file:///home/...` freeze entry is deliberately excluded: rebuild it
with this script, the pinned commit and the checked-in compatibility patch.
The installed Torch reports the CUDA build suffix even though pip metadata
freezes its public package version as `torch==2.14.1`.

## Compiler components and layout

The script replays these exact packages with `pip install --no-deps`:

| Package | Version |
| --- | --- |
| nvidia-cuda-nvcc | 13.0.88 |
| nvidia-nvvm | 13.0.88 |
| nvidia-cuda-crt | 13.0.88 |
| nvidia-cuda-cccl | 13.0.85 |
| setuptools | 80.9.0 |

The four NVIDIA compiler wheels total about 97.5 MiB. This uses the already
installed CUDA runtime; it does not download the full toolkit. The compiler
reports `Cuda compilation tools, release 13.0, V13.0.88`.

The NVIDIA wheels share `$VENV/lib/python3.12/site-packages/nvidia/cu13`.
`CUDA_HOME` supplies `bin`, `include` and `nvvm` links to that directory, plus
`lib64` links and the unversioned `libcudart.so`, `libnvrtc.so` aliases. Its
`libcuda.so` link uses WSL's existing `/usr/lib/wsl/lib/libcuda.so`.

The build environment is recorded in `tcnn_build.json`:

```text
TCNN_CUDA_ARCHITECTURES=120
TCNN_HALF_PRECISION=1
MAX_JOBS=2
PATH=$CUDA_HOME/bin:$PATH
CPATH=$VENV/lib/python3.12/site-packages/nvidia/cu13/include/cccl
LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/wsl/lib
LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/lib/wsl/lib
LDFLAGS=-Wl,-rpath,$VENV/lib/python3.12/site-packages/nvidia/cu13/lib
```

The generated extension contains that CUDA library RUNPATH. Replaying it
generates the path for the target venv rather than copying an old binary.

## Source and build compatibility

The script checks out the pinned official commit, initializes its pinned
submodules and applies `notes/tcnn_cpp20.patch`. The patch uses C++20 for the
PyTorch `.cpp` binding and retains official C++17 for native TCNN `.cu` files.
ATen in PyTorch 2.14 requires C++20; applying it to all native CUDA files
instead triggered an NVCC 13.0.88 compiler error. No TCNN runtime or network
code is modified. Setuptools 80.9 supplies `pkg_resources`, imported by the
official setup script. The final build uses no temporary fmt workaround.

The binding is installed with `--no-build-isolation --no-deps`. The successful
CPU compilation took 152.1 seconds with two workers. `tcnn_build.json` records
the source, submodules, actual compiler, environment and patch checksum;
`tcnn_build.log` contains the original build output.

Primary sources: [official TCNN setup](https://github.com/NVlabs/tiny-cuda-nn/blob/3daa6e5b8b8e0fa17971a1a20acfbb3754494b38/bindings/torch/setup.py),
[PyTorch ATen C++20 requirement](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/ATen.h),
[NVIDIA CUDA pip components](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-installation-guide-linux/index.html#pip-wheels).

## Real GPU validation

The installer did not run a GPU test. Root subsequently completed
`runs/slides_reproduction/tcnn_hash_single_exact/summary.json`: 1000 sampled
updates, official FullyFusedMLP, explicit seed 2026, and full evaluation of all
120777 original nodes from `506_533/ver`. Best and final metrics are finite.
Best was update 500 with physical stress relative L2 `0.004139400298830824`
(0.41394003%); total elapsed was 18.37 seconds. This is a real fitting run,
not a synthetic installer smoke test. Its configuration records assumptions
for the unavailable original learning rate and asinh constants.

`tcnn_build.json.root_run` links this independent GPU validation while
`GPU_smoke_executed` remains false for the installer. No GPU work is part of
`install_tcnn_wsl.py`.
