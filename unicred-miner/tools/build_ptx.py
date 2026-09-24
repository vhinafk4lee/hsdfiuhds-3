#!/usr/bin/env python3
"""Rebuild the PTX for worker/kernel.cu with NVRTC and embed it into worker/worker.py.

NVRTC comes from the pip wheel (no CUDA toolkit needed):

    pip download --no-deps nvidia-cuda-nvrtc-cu11==11.8.89 -d wheels
    python -m zipfile -e wheels/nvidia_cuda_nvrtc_cu11-*.whl wheels/nvrtc
    python tools/build_ptx.py --nvrtc wheels/nvrtc/nvidia/cuda_nvrtc/lib/libnvrtc.so.11.2

Optional check with ptxas (from nvidia-cuda-nvcc-cu12):

    python tools/build_ptx.py --nvrtc ... --ptxas path/to/ptxas --sm 61,75,86,89,90

Or simply let the script fetch both wheels into tools/.cache and do everything:

    python tools/build_ptx.py --download --check

Linux only (NVRTC is loaded via ctypes). The result is committed, so the
controller machine (Windows) never needs this script.
"""
import argparse
import base64
import ctypes
import os
import re
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "worker" / "kernel.cu"
WORKER = ROOT / "worker" / "worker.py"
PTX_OUT = ROOT / "worker" / "kernel.ptx"
CACHE = ROOT / "tools" / ".cache"
CACHE_NVRTC = CACHE / "nvidia" / "cuda_nvrtc" / "lib" / "libnvrtc.so.11.2"
CACHE_PTXAS = CACHE / "nvidia" / "cuda_nvcc" / "bin" / "ptxas"
WHEELS = ["nvidia-cuda-nvrtc-cu11==11.8.89", "nvidia-cuda-nvcc-cu12"]
BEGIN = "# ---- BEGIN EMBEDDED PTX (tools/build_ptx.py) ----"
END = "# ---- END EMBEDDED PTX ----"


def download():
    """pip download the NVRTC / nvcc wheels and unpack them into tools/.cache."""
    import zipfile
    CACHE.mkdir(parents=True, exist_ok=True)
    wheels = CACHE / "wheels"
    subprocess.check_call([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                           "--platform", "manylinux2014_x86_64", "-d", str(wheels)] + WHEELS)
    for whl in wheels.glob("*.whl"):
        with zipfile.ZipFile(str(whl)) as z:
            z.extractall(str(CACHE))
    if CACHE_PTXAS.exists():
        CACHE_PTXAS.chmod(0o755)


def find_ptxas(explicit=None):
    for cand in (explicit, os.environ.get("PTXAS"), str(CACHE_PTXAS) if CACHE_PTXAS.exists() else None):
        if cand:
            return cand
    import shutil
    return shutil.which("ptxas")


def find_nvrtc(explicit):
    if explicit:
        return explicit
    env = os.environ.get("NVRTC_LIB")
    if env:
        return env
    if CACHE_NVRTC.exists():
        return str(CACHE_NVRTC)
    try:
        import nvidia.cuda_nvrtc  # type: ignore
        lib_dir = Path(list(nvidia.cuda_nvrtc.__path__)[0]) / "lib"
        for cand in sorted(lib_dir.glob("libnvrtc.so*")):
            return str(cand)
    except ImportError:
        pass
    return "libnvrtc.so"


def compile_ptx(nvrtc_path, arch):
    lib_dir = Path(nvrtc_path).resolve().parent
    for builtins in sorted(lib_dir.glob("libnvrtc-builtins.so*")):
        ctypes.CDLL(str(builtins), mode=ctypes.RTLD_GLOBAL)
    nvrtc = ctypes.CDLL(nvrtc_path)
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p

    def check(res, what):
        if res != 0:
            raise RuntimeError(f"{what}: {nvrtc.nvrtcGetErrorString(res).decode()}")

    major, minor = ctypes.c_int(), ctypes.c_int()
    check(nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)), "nvrtcVersion")
    print(f"NVRTC {major.value}.{minor.value}")

    prog = ctypes.c_void_p()
    src = KERNEL.read_bytes()
    check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"kernel.cu", 0, None, None),
          "nvrtcCreateProgram")
    opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++11", b"-lineinfo"]
    opts = [o for o in opts if o != b"-lineinfo"]  # keep PTX small
    arr = (ctypes.c_char_p * len(opts))(*opts)
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    if log.value.strip():
        print(log.value.decode(errors="replace"))
    check(res, "nvrtcCompileProgram")
    size = ctypes.c_size_t()
    check(nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size)), "nvrtcGetPTXSize")
    buf = ctypes.create_string_buffer(size.value)
    check(nvrtc.nvrtcGetPTX(prog, buf), "nvrtcGetPTX")
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.value.decode()


def embed(ptx):
    text = WORKER.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit("PTX markers not found in worker.py")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    packed = base64.b64encode(zlib.compress(ptx.encode(), 9)).decode()
    lines = "\n".join(packed[i:i + 100] for i in range(0, len(packed), 100))
    block = f'{BEGIN}\nPTX_ZB64 = """\n{lines}\n"""\n{END}'
    WORKER.write_text(head + block + tail, encoding="utf-8", newline="\n")


def run_ptxas(ptxas, ptx_path, sms):
    """Assemble for each sm; return list of (sm, regs, spill_stores, spill_loads, stack)."""
    rows = []
    for sm in sms:
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run(
                [ptxas, "-v", f"-arch=sm_{sm}", str(ptx_path), "-o", os.path.join(tmp, "k.cubin")],
                capture_output=True, text=True)
        text = out.stdout + out.stderr
        if out.returncode != 0:
            raise SystemExit(f"ptxas sm_{sm} failed:\n{text}")
        # take the stats of unicred_search
        block = text.split("unicred_search", 1)[1] if "unicred_search" in text else text
        regs = int(re.search(r"Used (\d+) registers", block).group(1))
        spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", block)
        stack = re.search(r"(\d+) bytes stack frame", block)
        rows.append((sm, regs, int(spill.group(1)), int(spill.group(2)), int(stack.group(1))))
        print(f"  sm_{sm}: unicred_search {regs} registers, "
              f"spill stores {spill.group(1)} B, spill loads {spill.group(2)} B, "
              f"stack {stack.group(1)} B")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nvrtc", help="path to libnvrtc.so (11.8 recommended)")
    ap.add_argument("--arch", default="compute_52")
    ap.add_argument("--ptxas", help="path to ptxas for verification")
    ap.add_argument("--sm", default="61,75,86,89,90")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--download", action="store_true", help="fetch NVRTC/ptxas wheels into tools/.cache")
    ap.add_argument("--check", action="store_true", help="run ptxas (found automatically)")
    args = ap.parse_args()
    if args.download:
        download()
    if args.check and not args.ptxas:
        args.ptxas = find_ptxas()
        if not args.ptxas:
            raise SystemExit("ptxas not found (use --download or --ptxas)")

    ptx = compile_ptx(find_nvrtc(args.nvrtc), args.arch)
    version = re.search(r"^\.version\s+(\S+)", ptx, re.M).group(1)
    target = re.search(r"^\.target\s+(\S+)", ptx, re.M).group(1)
    print(f"PTX: {len(ptx)} bytes, ISA {version}, target {target}")
    PTX_OUT.write_text(ptx, encoding="utf-8", newline="\n")
    if not args.no_embed:
        embed(ptx)
        print(f"embedded into {WORKER.relative_to(ROOT)} (zlib+base64)")
    if args.ptxas:
        rows = run_ptxas(args.ptxas, PTX_OUT, [s.strip() for s in args.sm.split(",") if s.strip()])
        bad = [r for r in rows if r[2] or r[3]]
        if bad:
            raise SystemExit(f"register spill detected: {bad}")
        print("ptxas: OK, no spills")


if __name__ == "__main__":
    sys.exit(main())
