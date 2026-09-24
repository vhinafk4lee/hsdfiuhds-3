"""Shared test helpers: repo paths, worker module import, CPU build of kernel.cu."""
import importlib.util
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "tests" / ".build"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_worker():
    spec = importlib.util.spec_from_file_location("unicred_worker", str(ROOT / "worker" / "worker.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_cpu_kernel():
    """Compile worker/kernel.cu as plain C++ (the #ifndef __CUDACC__ shims) into a shared lib."""
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if not cxx or os.name == "nt":
        raise unittest.SkipTest("no C++ compiler (g++/clang++) for the CPU build of kernel.cu")
    BUILD.mkdir(parents=True, exist_ok=True)
    src = ROOT / "worker" / "kernel.cu"
    lib = BUILD / "libkernel_cpu.so"
    if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        subprocess.check_call([cxx, "-x", "c++", "-O2", "-shared", "-fPIC", "-Wall", "-Wextra",
                               "-Wno-unknown-pragmas", "-Wno-unused-parameter", "-o", str(lib), str(src)])
    return str(lib)


def build_fake_libcuda():
    """tests/fake_libcuda.cpp: CUDA Driver API subset running kernel.cu on the CPU."""
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if not cxx or os.name == "nt":
        raise unittest.SkipTest("no C++ compiler for the fake libcuda")
    BUILD.mkdir(parents=True, exist_ok=True)
    src = ROOT / "tests" / "fake_libcuda.cpp"
    lib = BUILD / "libcuda_fake.so.1"
    newest = max(src.stat().st_mtime, (ROOT / "worker" / "kernel.cu").stat().st_mtime)
    if not lib.exists() or lib.stat().st_mtime < newest:
        subprocess.check_call([cxx, "-O2", "-shared", "-fPIC", "-Wno-unknown-pragmas", "-o", str(lib), str(src)])
    return str(lib)
