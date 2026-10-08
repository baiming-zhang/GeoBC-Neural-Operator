"""Replay the pinned TCNN build in WSL; defaults to a read-only dry run.

The target venv must already contain the recorded PyTorch/CUDA runtime.
Only compiler components and setuptools are installed, with --no-deps.
No driver installation, GPU probing, model construction or training occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "3daa6e5b8b8e0fa17971a1a20acfbb3754494b38"
PATCH_SHA256 = "7f9008e060a3bfb51366941b831cd1dd226beb6f6191bfb1dd71bfe44685410e"
COMPILER_PACKAGES = ["nvidia-cuda-nvcc==13.0.88", "nvidia-nvvm==13.0.88",
                     "nvidia-cuda-crt==13.0.88", "nvidia-cuda-cccl==13.0.85",
                     "setuptools==80.9.0"]


def inspect_runtime(python: str) -> dict:
    code = """import sys,json,torch,importlib.metadata as md
try: tcnn=md.version('tinycudann')
except md.PackageNotFoundError: tcnn=None
print(json.dumps({'prefix':sys.prefix,'torch':torch.__version__,
 'cuda':torch.version.cuda,'runtime':md.version('nvidia-cuda-runtime'),
 'tinycudann':tcnn,'python_version':list(sys.version_info[:3]),
 'CUDA_context_initialized':torch.cuda.is_initialized()}))
"""
    return json.loads(subprocess.check_output([python, "-c", code], text=True))


def toolchain_links(base: Path, prefix: Path) -> None:
    prefix.mkdir(parents=True, exist_ok=True)
    for name in ("bin", "include", "nvvm"):
        link = prefix / name
        if link.is_symlink():
            if link.resolve() != (base / name).resolve():
                raise ValueError(f"Unexpected existing toolchain link: {link}")
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(base / name, target_is_directory=True)
    library_dir = prefix / "lib64"
    library_dir.mkdir(exist_ok=True)
    targets = {library.name: library for library in (base / "lib").iterdir() if library.is_file()}
    targets.update({"libcudart.so": base / "lib/libcudart.so.13",
                    "libnvrtc.so": base / "lib/libnvrtc.so.13",
                    "libcuda.so": Path("/usr/lib/wsl/lib/libcuda.so")})
    for name, target in targets.items():
        if not target.exists():
            raise FileNotFoundError(target)
        link = library_dir / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"Unexpected existing library link: {link}")
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Target venv Python; run this script inside WSL")
    parser.add_argument("--source", type=Path, help="Isolated checkout; defaults to <venv-owner>/src/tiny-cuda-nn")
    parser.add_argument("--cuda-home", type=Path, help="Symlink layout; defaults to <venv-owner>/.local/cuda-13.0-tcnn")
    parser.add_argument("--jobs", type=int, default=2)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Print commands; the default")
    action.add_argument("--install", action="store_true", help="Install into a prepared venv; skip an already installed TCNN 2.0")
    parser.add_argument("--log", type=Path, default=ROOT / "notes/tcnn_replay_build.log")
    args = parser.parse_args()
    if os.name != "posix" or args.jobs < 1:
        parser.error("Run inside WSL/Linux with a positive --jobs value.")
    runtime = inspect_runtime(args.python)
    if (runtime["torch"], runtime["cuda"], runtime["runtime"]) != ("2.14.1+cu130", "13.0", "13.0.96"):
        raise RuntimeError(f"Prepare the recorded Torch/runtime first; this script will not replace them: {runtime}")
    if runtime["python_version"][:2] != [3, 12] or runtime["CUDA_context_initialized"]:
        raise RuntimeError(f"Expected Python 3.12 and a read-only CPU probe: {runtime}")
    venv = Path(runtime["prefix"])
    owner = venv.parent.parent if venv.parent.name == ".venvs" else Path.home()
    source = args.source or owner / "src/tiny-cuda-nn"
    prefix = args.cuda_home or owner / ".local/cuda-13.0-tcnn"
    base = next(venv.glob("lib/python3.12/site-packages/nvidia/cu13"))
    patch = ROOT / "notes/tcnn_cpp20.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != PATCH_SHA256:
        raise ValueError("The recorded compatibility patch has changed.")
    environment = {
        "CUDA_HOME": str(prefix), "TCNN_CUDA_ARCHITECTURES": "120",
        "TCNN_HALF_PRECISION": "1", "MAX_JOBS": str(args.jobs),
        "CPATH": str(base / "include/cccl"),
        "LIBRARY_PATH": str(prefix / "lib64") + ":/usr/lib/wsl/lib",
        "LD_LIBRARY_PATH": str(base / "lib") + ":/usr/lib/wsl/lib",
        "LDFLAGS": "-Wl,-rpath," + str(base / "lib"),
    }
    compiler_command = [args.python, "-m", "pip", "install", "--no-deps", *COMPILER_PACKAGES]
    clone_command = ["git", "clone", "--no-checkout", "https://github.com/NVlabs/tiny-cuda-nn", str(source)]
    fetch_command = ["git", "-C", str(source), "fetch", "--depth", "1", "origin", COMMIT]
    checkout_command = ["git", "-C", str(source), "checkout", "--detach", COMMIT]
    submodule_command = ["git", "-C", str(source), "submodule", "update", "--init", "--recursive"]
    patch_command = ["git", "-C", str(source), "apply", str(patch)]
    build_command = [args.python, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "--verbose", str(source / "bindings/torch")]
    print(json.dumps({"mode": "install" if args.install else "dry-run", "runtime": runtime,
                      "commit": COMMIT, "patch_sha256": PATCH_SHA256,
                      "source": str(source), "CUDA_HOME": str(prefix), "environment": environment,
                      "commands_for_fresh_environment": [compiler_command, clone_command, fetch_command,
                                                           checkout_command, submodule_command, patch_command, build_command],
                      "toolchain_operation": "symlink installed cu13 bin/include/nvvm and library aliases into CUDA_HOME",
                      "CUDA_standard": 17, "CPP_binding_standard": 20,
                      "GPU_execution": False}, indent=2), flush=True)
    if not args.install:
        return 0
    if runtime["tinycudann"] == "2.0":
        print("TCNN 2.0 is already installed; skipped all installation/build mutations.")
        return 0
    if runtime["tinycudann"] is not None:
        raise RuntimeError("A different TCNN version is installed; use a separate target venv.")
    for executable in ("git", "g++", "ninja"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Missing host prerequisite: {executable}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, **environment, "PATH": str(prefix / "bin") + ":" + os.environ["PATH"]}
    with args.log.open("w") as log:
        def run(command: list[str], env: dict | None = None) -> None:
            print(json.dumps(command), flush=True)
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        run(compiler_command)
        toolchain_links(base, prefix)
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            run(clone_command)
            run(fetch_command)
            run(checkout_command)
        elif subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
            raise RuntimeError("Existing checkout is a different commit; choose another --source.")
        run(submodule_command)
        if subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            run(patch_command)
        elif subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            raise RuntimeError("Existing setup.py differs from the pinned original and recorded patch.")
        run([str(prefix / "bin/nvcc"), "--version"], environment)
        run(build_command, environment)
    after = inspect_runtime(args.python)
    if (after["torch"], after["cuda"], after["runtime"], after["tinycudann"]) != ("2.14.1+cu130", "13.0", "13.0.96", "2.0"):
        raise RuntimeError(f"Unexpected installed versions: {after}")
    print("Compiled TCNN 2.0. GPU execution was not requested by this installer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
