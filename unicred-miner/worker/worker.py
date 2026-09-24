#!/usr/bin/env python3
"""UNICRED GPU worker. Standard library only, no keys, no network.

Runs on a GPU server (vast.ai). Talks to the controller over stdin/stdout:

  controller -> worker
    JOB <id> <prefix128hex> <sender20hex> <target32hex> <noncePrefix16hex>
    IDLE
    PING <token>
  worker -> controller
    READY <ngpu> <gpu names json>
    HR <total_hps> <gpu0_hps,gpu1_hps,...>
    FOUND <id> <nonce32hex> <digest32hex>
    PONG <token>
    LOG <text>
    ERR <text>

EOF on stdin (or no line for --idle-timeout seconds) -> the worker exits, so
GPUs never keep burning stale work when the controller is gone.

Kernel: PTX embedded below (built from kernel.cu by tools/build_ptx.py),
loaded through the CUDA Driver API (libcuda.so.1) via ctypes; the driver
JIT-compiles it for the local GPU, so no nvcc is needed. Fallbacks: lower
PTX ISA version for old drivers, then nvcc if it exists.

Usage:
  python3 -u worker.py              serve (protocol above)
  python3 worker.py --selftest      find the test-vector nonce on every GPU
  python3 worker.py --bench 10      hashrate per GPU
  python3 worker.py --cpu-lib X.so  use a CPU build of kernel.cu (tests only)
"""
import argparse
import base64
import ctypes
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zlib

# ---------------------------------------------------------------------------
# Keccak-256 (pure Python, used for midstates and exact candidate checks)
# ---------------------------------------------------------------------------
M64 = (1 << 64) - 1
RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
ROT = [0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14]
PI = [0] * 25
for _x in range(5):
    for _y in range(5):
        PI[_x + 5 * _y] = _y + 5 * ((2 * _x + 3 * _y) % 5)


def keccak_f(a):
    """keccak-f[1600] on a list of 25 ints (lane index x + 5*y), in place."""
    for rc in RC:
        c = [a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] for x in range(5)]
        d = [c[(x + 4) % 5] ^ (((c[(x + 1) % 5] << 1) | (c[(x + 1) % 5] >> 63)) & M64) for x in range(5)]
        b = [0] * 25
        for i in range(25):
            v = a[i] ^ d[i % 5]
            r = ROT[i]
            b[PI[i]] = ((v << r) | (v >> (64 - r))) & M64 if r else v
        for y in range(0, 25, 5):
            for x in range(5):
                a[y + x] = b[y + x] ^ ((~b[y + (x + 1) % 5]) & b[y + (x + 2) % 5])
        a[0] ^= rc
    return a


def lanes_of(block):
    return [int.from_bytes(block[8 * i:8 * i + 8], "little") for i in range(len(block) // 8)]


def keccak256(data):
    data = bytearray(data)
    pad = 136 - len(data) % 136
    data += b"\x00" * pad
    data[len(data) - pad] ^= 0x01
    data[-1] ^= 0x80
    st = [0] * 25
    for off in range(0, len(data), 136):
        for i, v in enumerate(lanes_of(data[off:off + 136])):
            st[i] ^= v
        keccak_f(st)
    return b"".join(st[i].to_bytes(8, "little") for i in range(4))


def bswap64(v):
    return int.from_bytes(v.to_bytes(8, "big"), "little")


# ---------------------------------------------------------------------------
# Job layout
# ---------------------------------------------------------------------------
TYPEHASH = bytes.fromhex("d38d03374ffe0ab0595a37e26339a924c9bf9d87e54292dabb07b3bddb3cc04e")
CHAIN_ID = 130
UNICRED = bytes.fromhex("f60de24f228dc7ca6ff025958d2ee3a956ed88e5")

# Test vector: mint #636
TV_BLOCKHASH = bytes.fromhex("cc06b669726f684d0d4e2ab050dbe9abcd592e41741cdd280eb8609956618656")
TV_CHALLENGE = bytes.fromhex("0000000000062445aa544d049a58f68155e7c268a6334635ed9796e9f8cf4fdb")
TV_MINER = bytes.fromhex("67d09f31a4453b11f39007a982248bde00625902")
TV_NONCE = bytes.fromhex("877face3b45ddc27c83c163a1dc14e34e2fe5e2f92bf3c0032098b94396a2939")
TV_DIGEST = bytes.fromhex("000000000000edf23d93bdc90caa71781e8c9bd831b71e8d5b2ae17271bced8e")


def job_prefix(blockhash, challenge, chain_id=CHAIN_ID, contract=UNICRED):
    """First 128 bytes of the message: TYPEHASH | chainid | contract | inner."""
    inner = keccak256(blockhash + challenge)
    return TYPEHASH + chain_id.to_bytes(32, "big") + b"\x00" * 12 + contract + inner


class Job(object):
    def __init__(self, job_id, prefix, sender, target, nonce_prefix):
        if len(prefix) != 128 or len(sender) != 20 or len(nonce_prefix) != 16:
            raise ValueError("bad job field length")
        self.id = job_id
        self.prefix = prefix
        self.sender = sender
        self.target = target
        self.nonce_prefix = nonce_prefix
        self.t0 = (target >> 192) & M64 if target < (1 << 256) else M64
        self.t1 = (target >> 128) & M64 if target < (1 << 256) else M64
        st = lanes_of(prefix + b"\x00" * 8)
        st += [0] * (25 - len(st))
        self.midstate = keccak_f(st)

    def cx_for(self, gpu8):
        """c_x = midstate ^ block2 (lane 6 = 0, counter goes there on the GPU)."""
        block2 = bytearray(136)
        block2[4:24] = self.sender
        block2[24:40] = self.nonce_prefix
        block2[40:48] = gpu8
        block2[56] ^= 0x01
        block2[135] ^= 0x80
        b = lanes_of(bytes(block2))
        return [self.midstate[i] ^ (b[i] if i < 17 else 0) for i in range(25)]

    def nonce(self, gpu8, ctr):
        return self.nonce_prefix + gpu8 + ctr.to_bytes(8, "big")


def digest_from_cx(cx, ctr):
    st = list(cx)
    st[6] ^= bswap64(ctr)
    keccak_f(st)
    return b"".join(st[i].to_bytes(8, "little") for i in range(4))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
_out_lock = threading.Lock()


def emit(line):
    with _out_lock:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError, ValueError):
            os._exit(0)


def log(text):
    emit("LOG " + " ".join(str(text).split()))


# ---------------------------------------------------------------------------
# Embedded PTX
# ---------------------------------------------------------------------------
# ---- BEGIN EMBEDDED PTX (tools/build_ptx.py) ----
PTX_ZB64 = """
eNqUvd3OJDmSHXjd9RR5IQEr7FfZzn9SMxB2RwKEAbR7OTfCoJHdmWoVUF3VW1U9GkkvvxHBc4x0Gj3gBgwmq78wt2NO0kij/fnv
f//d73//6T9/++nbL19++/b10x//56f/95/+8T/94//9+Oef/p9P//Hnv/z1hx+//fIgetLxf376h7/98OPXT//4n/79p//4X74P
robQjvSi+NvXL5/+9CL78tsPP//06beff/7x149Pv3z78duXX799cu5z/fj0T89/Ptf2fOIfHn/++ulB+QIsn4/P7on23ed/+fbL
r08O5XP97vNvX37587ffPv36lz8k/93nL1+//vLt11//8OsP/+vbpxy/++53D0af//zjz3/88Xd/++mHP/3y7esffv325Zc//fcn
nx9+/eGPP3779PnbT7/98j8/nX//P7773ee/fvnly18+ff7y4w9//ulT/fT5j3Uh+sOL4g/Hf/Uu//PHeOJvOe4p3UwU/J7I3+EU
vvt3333+y5d//em3H75+8il/fHKP//vufz8e/eXbnz99/uvjiU+/+7d//fv6H/6Of/vjA/F3//aXvy+l5OmvD4jHX7/+fco1Pf78
GLS//Pwv8ud4fFy89YPDj1+7oC85X9QPKf7r/tX/+UTfRXGX5P6ft+z9FX140v/pX3778vm3n18T/uVHeeoB8nr2QfHrt9/++vnb
//f51xf+X1+/PIV4vsz/9fjfn/74y5dPv/s3/+UPf/iHfzj+UP4Og0Fxw4P+OeKf//Xv8AMFCx1jOyhPoV80/2fdvlWQ3+OeIA4G
e4IkBM7vKfKgyHuKMqQ4tgSVBM9f//XnX15LpMsXX29fXv+/7p5t4w0O/Xjqg9e5tN3z7pjecMMhdw6dkdvOwmutcQg0C+c7i87J
uS2LMVMpb8fIxQlEi9lHyPXF4uKWw5jLtuFQu5RglLYcprkOGxats+icXN6yGIvBFc3CQYrOyZUtizHlOW7Hyo859VFhpKMz7xh+
r1jTlB4bFl3xwclvp9QP5XRxw6KvCnDyfstirApXNQvflzU4+bBlMSa97Hc+PyY1eC1mX1G+r3+/nVM/zanbsICY4LSdU18Hi7Rh
0eHByW83Aj+WhWuaRehzBU6+9U39uf+Sg4tDedafoP6BXB+79uvvL53uv26fm7Tp9NxrmePpsnmuDRWan/Ovl8DTm1fwbujNCe/5
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
ES9Chs3yKbMG2ZIGA44IMUZnanDWXX1ILbxIiTEGC92DQIxRX7yoSklJidQH2wyspNR0iehFjziKlYCIFFGRoeLTtCXAljQQlI1I
sw5yNJY9gvqQ2oknNVwgjmJYkPuCoxhNSkryY71MQWxBtx50rJ2PR5DEulNHQBLMo+Bp2hJgSxqNsxwIF5iSgDqqD6kteFLDhWYz
QS96AUmMJiWlcegSzdvVVRNAFlE/IiDWPS2CgEDawi6gsCXAljSAgAglNXCltxwB1IfU8kkNF4giGERAjBcExOiS6Z4SdA/Cittj
V+IgvGE8whvWhS0Cb8BNnDZwlABb0kD8PCKwYl48oY7qQ2rhRQorHoQ3DMIbxgXecKECGIZu+HWMKxXAMMiCOaaSfgyr0C/c+TYC
w0TbNw7pxxj2x7lbc9VjRf7uWGqVe3t5h/RjsFK9xFtVPcWGcvSwFmcffIf0Y7DYvcS78z6GcPT02KRyahzSjzGmvLWQPYxV4XX0
WEGXUsMh/Rgs1y7xoe+zqpPO+ljBncXM4ZB+jPXHstt809NsKEeP1VvsfZqjh2fXKnI67zNtKNEzreBnW9ch/Zgs9C3xKe8zj7TP
6V2PYSsqh0qOnnJ766nkL9MKYKenhzv2ND3V0dP5R6tZOe9jaefq6OGkJrNUz4N52h/nbvNVD+s/IznvY0UdplrneTp6Mv+Ydptv
eooN5eixqqLpKY6exj/W3eabnr7P6V2PTSqn5uyOnilvLefBtPpAdvQku3RTT9LzYDLZv8Szvo/FTMVZH9JwWPphpuzosT+W3eab
nmZDOXrs+mPv0xw9g3/su803PXbZdd7Hoo9sepzzgGniJV70fVYO2NNjqU5OjRN2TatX2ltnPQ8sfdo8PZzUZnqc84C50CVenfex
QNFZH8aFFldOhzln2mY0cY0PJrOHw4kPpqUE+aWcDhPOZApsiWt8MJlWGU58MC2bVk2Pcx4wi7bEu/M+fZ/Tux6bVE6Nw1Qzy5S3
dmbX+nMspnH4bCbTSr/i7Cmt3nAWLPCUcPgYJ/MkJj4PT2mhFMWTs91trk08e2Nxj1qU4RA+TaZUTHyenhQPDBN3+m4m2Y2WeHHG
sgjPohKH8GkyH2Hi8/SkOJkmnpwV4u16iRdvLE4mo5jhED5NOx9M/PSkOJnrOHH2BC/iS7x4Y3EyLepxCJ8mKRyX+OlI2alj4tlx
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
VHytdgdbAgxJ+PXVWD0Y1VArr0J9SC33loYLREAMIiDGCwJiZMl0Nwm6OwHF7bH/cBDeMB7hDVbIGhF4A4qYtIGjBBiS+DO4PQIo
ZkEO6qg+pBZepIDiYecJ4Q3jAm+4tP8PQzf8xvvX9v9hkAVzTCX6GFZiyNkZgXBl2zcO0cco9se+W3PTM20o1bMq92apdpgPc2MT
z1n0kOijF+d9rOhtH3yH6GMwpb/Es76PlamTp8ewfpwah+hjkOhjvbUSpIxV4fX0cFLNfRyij8Fy7RIv+j5WnazO+lgGncXM4RB9
jPXHtNt801NsKEePVS35Pg7Rx2CVbolX5326DeXoMXyK6emOHl5hVn1Q38cy18XRY8Uq7qThEH2Mnu5vrYQvYxXAPD2c1GF6sqOH
X0oT7877WNrZWR/Se1iWenTnPOj2x77bfNMzbSjVY0Wdae/jnAcsYizxoe9DXo4+nfexeghndwznPGAdZIkPfR8rcnRPj1XEODXD
OQ9IwrHeWglshs384enhjrXMzXDOAyb7TbxPfZ9pWTlnfUi4sdIP0zkP7DVM/ND1YXp8nM77WLbU7tDTOQ+Y413ih/M+3YZy9Fi6
2PQ45wHTxCb+Z/NFz1w5YNUzV6oTUzMPPQ+mJSD51vM4VY9dWT09nNRserKjp/KPFE/O+1igmB09nNRsljZHj/2x7zbf9EwbSvVY
vq/Y++h5MJkCW+IaH0ymVYYTH0zbzowPpsNhM5lFW+IaH0xLkSVPj91hOTUOJ81kemG9tUO7OM9xj2mKN9bkEEOJmfLh8N5NyyAx
fpgOG+NkxmSJd0c9r+EmPqaz8W3WTfzwxuJutXjDIXmaTK4s8e5JNUpxuZwOnMlE2RI/vLEsluDqOiRPk5mJJT4cKeYdTHxMZ+l5
z17ipzcWd7zFMw7J07STwsSHJ8Wtb1eK6ewJXsmX+OmNxWli/DMckqdJAsclPjwp+pGJH47j8j67xE9pQ2Y3mtMdCc6myu7I5jR1
8E9jawi6jE/QgNP6SNQVWx+7tj521Ikp6rQ+kpmBok0aTFmLd1ofOxEVNEFbHztSMRR1Wh/BqbRMkdbHPrai082E+W8rm3xEZfwB
5FafW3XxJoL0K00Zwq0w9kTW1QR0LnV2jw7tNx/IelBUsc7AhC5ThqQXZzE3f+q0NkjiDCVGgCIiLrWHeyvZbd4jvZVAnRrEE+pD
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
jQjYv/hNopeK0CZvyCICM4gpgu7DaFZ04uhYokS18mtM0lqpoilSzCoCk6PghKpZ92H0TTpx1KrLfDpN94llq8ZKepP+zgISanox
1vvRg/1+be7D7LGBkL6NpynQfV+bp3zDsfCLzn457Fk1ovWbn8lQsiMK49uFofIyapD3+3XOk2fqNpDAiPdJvtkfcplv0BtfBtMs
jDRpyl9h3Lzcfh04RVsOnrMZNha7n0AWR/cbdjJJni4SaD8zwR1uWHKQrMzCLIDChILy5evAhekAFqXZBi6YF3CSbzZiJjL2dHNl
Y8cWI03amqcwu+CDeTPXfLHiuVJFNlSCQUSr6q4vxGO+9covelv+0KGfPobr5mcuUnW+MEWiMI/iQ7zPeirFtChJTLkUG1Mun53F
VCxAavGbzc5QjQUtk83nqAqzJybyHRWB3uTLB1LePd4lZaNUW0D6xZpSbZw4choHtnuqEeckJ2Wb9trlpDTVxFrXw7rwOkI1UGWC
ixqkb5ekksDwIWnekMAgGbpUNcmwMH/kM9A7dft5mMPwCQH/+S//Oo0DzQECsSFx2i/lCkOpbDjxp/SME06NT9bPrBR2j6s2jXVS
if/441///Pt//ive7s9jdeGZTxHu//Hzx9/+w1/+8Nu/+U//59///X/7b6+//0HYjQqeUcsEqTKx3A4ode0zH//67/+D7/Xz1j/T
/ZKxqVwMeGm2JrgbG6y8zw1k4cQ10x5xQkCmfoJs162nmfpSydvrfpypk3hqIvkjSZJMxVuv/vyXP/7TezH/+b3QPjR/jmOW8maW
/vDXf//X3//zn/79H/7wp9//4Z/+ySblvTf/7c+/x9/9+u2h0o2V8ib61HT8MC8b5tdAGJKRIj3NP2j/cf443pdDNL4D/ldK9kH9
WxD+5/Z3pPrB/o0h/m+/+Zs//bz+H/7yh3+dlscP37/9v/7tX/7xZ1T+/j/++Ie//ON///sPyd+Hv6MQRgvVfP97qITJUMd7JA7L
n/4KvD/Xk8EAH2NTr2MTJqnrj9R/+eNff/7y/33Ovs8b/+lvKPF//8N//Pff/P5//Mt//Ms//OmPv/39H//tr3/5f347//i//OZv
xiv/9vd/+NO//PO//bb99vf/0C4keN/X34aj/N2vk/49qBu6439OEn7zv/7m//2h+ssf//m3Y739vPt/Df/7f+HfoD3/9X0Nnf6K
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
/LcfP379dfmx/vXbn/73799//p8/vn37PwNcVoA=
"""
# ---- END EMBEDDED PTX ----


def embedded_ptx():
    data = "".join(PTX_ZB64.split())
    if not data:
        raise RuntimeError("no embedded PTX: run tools/build_ptx.py")
    return zlib.decompress(base64.b64decode(data)).decode()


# ---------------------------------------------------------------------------
# CUDA driver API via ctypes
# ---------------------------------------------------------------------------
class JobParams(ctypes.Structure):
    _fields_ = [("cx", ctypes.c_uint64 * 25), ("t0", ctypes.c_uint64), ("t1", ctypes.c_uint64)]


class CudaError(RuntimeError):
    def __init__(self, code, what, name=""):
        RuntimeError.__init__(self, "%s failed: %s (%d)" % (what, name, code))
        self.code = code


CU_CTX_SCHED_BLOCKING_SYNC = 0x4
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_CC_MAJOR = 75
CU_DEVICE_ATTRIBUTE_CC_MINOR = 76
CU_FUNC_ATTRIBUTE_NUM_REGS = 4
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
BAD_PTX_CODES = (218, 222, 209, 200)  # INVALID_PTX, UNSUPPORTED_PTX_VERSION, NO_BINARY_FOR_GPU, INVALID_IMAGE


class Cuda(object):
    def __init__(self):
        self.lib = ctypes.CDLL(os.environ.get("UNICRED_LIBCUDA") or "libcuda.so.1")
        self.lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.check(self.lib.cuInit(0), "cuInit")

    def check(self, res, what):
        if res != 0:
            name = ctypes.c_char_p()
            try:
                self.lib.cuGetErrorName(res, ctypes.byref(name))
            except Exception:
                pass
            raise CudaError(res, what, (name.value or b"?").decode())

    def driver_version(self):
        v = ctypes.c_int()
        self.check(self.lib.cuDriverGetVersion(ctypes.byref(v)), "cuDriverGetVersion")
        return v.value

    def device_count(self):
        n = ctypes.c_int()
        self.check(self.lib.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
        return n.value

    def device(self, index):
        dev = ctypes.c_int()
        self.check(self.lib.cuDeviceGet(ctypes.byref(dev), index), "cuDeviceGet")
        return dev

    def attr(self, dev, attr):
        v = ctypes.c_int()
        self.check(self.lib.cuDeviceGetAttribute(ctypes.byref(v), attr, dev), "cuDeviceGetAttribute")
        return v.value

    def name(self, dev):
        buf = ctypes.create_string_buffer(256)
        self.check(self.lib.cuDeviceGetName(buf, 256, dev), "cuDeviceGetName")
        return buf.value.decode(errors="replace")


def lower_ptx_version(ptx, version):
    lines = ptx.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(".version"):
            lines[i] = ".version " + version
            break
    return "\n".join(lines)


class CudaDevice(object):
    """One GPU: own context, module and buffers. Used from a single thread."""

    BLOCK = 256

    def __init__(self, cuda, index):
        self.cuda = cuda
        self.lib = cuda.lib
        self.index = index
        self.dev = cuda.device(index)
        self.name = cuda.name(self.dev)
        self.sm_count = cuda.attr(self.dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        self.cc = (cuda.attr(self.dev, CU_DEVICE_ATTRIBUTE_CC_MAJOR),
                   cuda.attr(self.dev, CU_DEVICE_ATTRIBUTE_CC_MINOR))
        self.ctx = self._context()
        self.make_current()
        self.module = ctypes.c_void_p()
        self.load_source = self._load_module()
        self.f_search = ctypes.c_void_p()
        self.f_hash = ctypes.c_void_p()
        cuda.check(self.lib.cuModuleGetFunction(ctypes.byref(self.f_search), self.module, b"unicred_search"),
                   "cuModuleGetFunction")
        cuda.check(self.lib.cuModuleGetFunction(ctypes.byref(self.f_hash), self.module, b"unicred_hash"),
                   "cuModuleGetFunction")
        regs = ctypes.c_int()
        self.lib.cuFuncGetAttribute(ctypes.byref(regs), CU_FUNC_ATTRIBUTE_NUM_REGS, self.f_search)
        local = ctypes.c_int()
        self.lib.cuFuncGetAttribute(ctypes.byref(local), CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.f_search)
        self.regs, self.local_bytes = regs.value, local.value
        occ = ctypes.c_int()
        res = self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(occ), self.f_search, self.BLOCK, ctypes.c_size_t(0))
        self.blocks_per_sm = occ.value if res == 0 and occ.value > 0 else 2
        self.grid = self.sm_count * self.blocks_per_sm
        self.d_out = ctypes.c_uint64()
        cuda.check(self.lib.cuMemAlloc_v2(ctypes.byref(self.d_out), ctypes.c_size_t(16 * 8)), "cuMemAlloc")
        self.h_out = (ctypes.c_uint64 * 16)()
        self._clear()

    def _context(self):
        """Primary context with blocking sync (no CPU spin while waiting for the GPU)."""
        ctx = ctypes.c_void_p()
        try:
            set_flags = getattr(self.lib, "cuDevicePrimaryCtxSetFlags_v2", None) or \
                self.lib.cuDevicePrimaryCtxSetFlags
            set_flags(self.dev, CU_CTX_SCHED_BLOCKING_SYNC)  # may fail if already active: harmless
            self.cuda.check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), self.dev),
                            "cuDevicePrimaryCtxRetain")
            return ctx
        except (AttributeError, CudaError):
            self.cuda.check(self.lib.cuCtxCreate_v2(ctypes.byref(ctx), CU_CTX_SCHED_BLOCKING_SYNC, self.dev),
                            "cuCtxCreate")
            return ctx

    def _load_module(self):
        ptx = embedded_ptx()
        attempts = [("embedded PTX", ptx)]
        for v in ("7.4", "7.0", "6.4"):
            attempts.append(("embedded PTX as ISA " + v, lower_ptx_version(ptx, v)))
        last = None
        for label, text in attempts:
            try:
                self._module_from_image(text.encode() + b"\x00")
                return label
            except CudaError as exc:
                last = exc
                if exc.code not in BAD_PTX_CODES:
                    raise
        cubin = nvcc_build(self.cc)
        if cubin is not None:
            self._module_from_image(cubin)
            return "nvcc sm_%d%d" % self.cc
        raise last

    def _module_from_image(self, image):
        log_buf = ctypes.create_string_buffer(8192)
        opts = (ctypes.c_int * 2)(CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
        vals = (ctypes.c_void_p * 2)(ctypes.cast(log_buf, ctypes.c_void_p), ctypes.c_void_p(8192))
        buf = ctypes.create_string_buffer(image, len(image))
        mod = ctypes.c_void_p()
        res = self.lib.cuModuleLoadDataEx(ctypes.byref(mod), buf, 2, opts, vals)
        if res != 0:
            err = log_buf.value.decode(errors="replace").strip()
            if err:
                log("gpu%d JIT: %s" % (self.index, err[:300]))
            self.cuda.check(res, "cuModuleLoadDataEx")
        self.module = mod

    def make_current(self):
        self.cuda.check(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    def _clear(self):
        self.cuda.check(self.lib.cuMemsetD8_v2(self.d_out, 0, ctypes.c_size_t(16 * 8)), "cuMemsetD8")

    def search(self, params, base, iters):
        """One launch; returns the list of hit counters."""
        base_c = ctypes.c_uint64(base)
        iters_c = ctypes.c_uint32(iters)
        args = (ctypes.c_void_p * 4)(
            ctypes.addressof(params), ctypes.addressof(base_c),
            ctypes.addressof(iters_c), ctypes.addressof(self.d_out))
        self.cuda.check(self.lib.cuLaunchKernel(
            self.f_search, self.grid, 1, 1, self.BLOCK, 1, 1, 0, None, args, None), "cuLaunchKernel")
        self.cuda.check(self.lib.cuMemcpyDtoH_v2(self.h_out, self.d_out, ctypes.c_size_t(16 * 8)),
                        "cuMemcpyDtoH")
        n = int(self.h_out[0])
        if not n:
            return []
        hits = [int(self.h_out[1 + i]) for i in range(min(n, 15))]
        self._clear()
        return hits

    def hash_one(self, params, ctr):
        ctr_c = ctypes.c_uint64(ctr)
        args = (ctypes.c_void_p * 3)(
            ctypes.addressof(params), ctypes.addressof(ctr_c), ctypes.addressof(self.d_out))
        self.cuda.check(self.lib.cuLaunchKernel(
            self.f_hash, 1, 1, 1, 1, 1, 1, 0, None, args, None), "cuLaunchKernel")
        self.cuda.check(self.lib.cuMemcpyDtoH_v2(self.h_out, self.d_out, ctypes.c_size_t(16 * 8)),
                        "cuMemcpyDtoH")
        out = b"".join(int(self.h_out[i]).to_bytes(8, "little") for i in range(4))
        self._clear()
        return out

    def describe(self):
        return "%s (sm_%d%d, %d SM, %d regs, grid %dx%d, %s)" % (
            self.name, self.cc[0], self.cc[1], self.sm_count, self.regs, self.grid, self.BLOCK,
            self.load_source)


def nvcc_build(cc):
    """Fallback: compile kernel.cu next to this file with a local nvcc."""
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.cu")
    if not nvcc or not os.path.exists(src):
        return None
    out = os.path.join(tempfile.gettempdir(), "unicred_sm%d%d.cubin" % cc)
    cmd = [nvcc, "-cubin", "-O3", "-arch=sm_%d%d" % cc, src, "-o", out]
    log("building with nvcc: " + " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        log("nvcc failed: " + res.stdout.decode(errors="replace")[-300:])
        return None
    with open(out, "rb") as fh:
        return fh.read()


class CpuDevice(object):
    """kernel.cu compiled as a plain C++ shared library (tests / no-GPU runs)."""

    def __init__(self, lib_path, index, grid=4, block=256):
        self.lib = ctypes.CDLL(lib_path)
        self.index = index
        self.name = "CPU kernel"
        self.grid, self.BLOCK = grid, block
        self.load_source = "cpu-lib"
        self.out = (ctypes.c_uint64 * 16)()

    def make_current(self):
        pass

    def search(self, params, base, iters):
        for i in range(16):
            self.out[i] = 0
        self.lib.cpu_search(params.cx, ctypes.c_uint64(params.t0), ctypes.c_uint64(params.t1),
                            ctypes.c_uint64(base), ctypes.c_uint32(iters), ctypes.c_uint32(self.grid),
                            ctypes.c_uint32(self.BLOCK), self.out)
        n = int(self.out[0])
        return [int(self.out[1 + i]) for i in range(min(n, 15))]

    def hash_one(self, params, ctr):
        out = (ctypes.c_uint64 * 4)()
        self.lib.cpu_hash(params.cx, ctypes.c_uint64(ctr), out)
        return b"".join(int(out[i]).to_bytes(8, "little") for i in range(4))

    def describe(self):
        return "CPU kernel (grid %dx%d)" % (self.grid, self.BLOCK)


def params_for(job, gpu8):
    p = JobParams()
    for i, v in enumerate(job.cx_for(gpu8)):
        p.cx[i] = v
    p.t0, p.t1 = job.t0, job.t1
    return p


def testvector_job():
    job = Job("selftest", job_prefix(TV_BLOCKHASH, TV_CHALLENGE), TV_MINER,
              int.from_bytes(TV_DIGEST, "big") + 1, TV_NONCE[:16])
    return job, TV_NONCE[16:24], int.from_bytes(TV_NONCE[24:], "big")


def selftest_device(dev, launch_hashes=None):
    """Hash the test vector and find its nonce with the search kernel."""
    dev.make_current()
    job, gpu8, ctr = testvector_job()
    params = params_for(job, gpu8)
    if digest_from_cx(job.cx_for(gpu8), ctr) != TV_DIGEST:
        return False, "python keccak mismatch"
    got = dev.hash_one(params, ctr)
    if got != TV_DIGEST:
        return False, "hash kernel digest %s != expected" % got.hex()
    iters = 2
    per_launch = dev.grid * dev.BLOCK * iters
    offset = random.randrange(per_launch // 4, per_launch - 1)
    hits = dev.search(params, ctr - offset, iters)
    if ctr not in hits:
        return False, "search kernel did not find nonce (hits %s)" % hits
    if len(hits) != 1:
        return False, "unexpected extra hits %s" % hits
    # a nonce just outside the launch range must not be reported
    hits = dev.search(params, ctr + 1, 1)
    if hits:
        return False, "false positive %s" % hits
    return True, "nonce found, digest %s" % TV_DIGEST.hex()


# ---------------------------------------------------------------------------
# Mining threads
# ---------------------------------------------------------------------------
class Shared(object):
    def __init__(self):
        self.job = None           # current Job or None (idle)
        self.version = 0
        self.cond = threading.Condition()
        self.stop = False

    def set_job(self, job):
        with self.cond:
            self.job = job
            self.version += 1
            self.cond.notify_all()


class GpuMiner(threading.Thread):
    LAUNCH_SECONDS = 0.040
    MAX_FOUND_PER_JOB = 8

    def __init__(self, dev, shared, on_found):
        threading.Thread.__init__(self, name="gpu%d" % dev.index)
        self.daemon = True
        self.dev = dev
        self.shared = shared
        self.on_found = on_found
        self.iters = 4
        self.hashes = 0          # total hashes, read by the reporter
        self.error = None

    def run(self):
        try:
            self._run()
        except Exception as exc:  # report and die; the process keeps other GPUs
            self.error = str(exc)
            emit("ERR gpu%d %s" % (self.dev.index, " ".join(str(exc).split())))

    def _run(self):
        self.dev.make_current()
        version = -1
        job = params = gpu8 = None
        ctr = 0
        found = 0
        while not self.shared.stop:
            if self.shared.version != version:
                with self.shared.cond:
                    version, job = self.shared.version, self.shared.job
                if job is not None:
                    gpu8 = bytes([self.dev.index & 0xFF]) + os.urandom(7)
                    params = params_for(job, gpu8)
                    cx = job.cx_for(gpu8)
                    ctr, found = 0, 0
            if job is None:
                with self.shared.cond:
                    if self.shared.version == version and not self.shared.stop:
                        self.shared.cond.wait(0.5)
                continue
            t = time.time()
            hits = self.dev.search(params, ctr, self.iters)
            dt = time.time() - t
            n = self.dev.grid * self.dev.BLOCK * self.iters
            ctr += n
            self.hashes += n
            for hit in hits:
                digest = digest_from_cx(cx, hit)
                if int.from_bytes(digest, "big") < job.target and found < self.MAX_FOUND_PER_JOB:
                    found += 1
                    self.on_found(job, job.nonce(gpu8, hit), digest)
            # autotune launch length toward LAUNCH_SECONDS
            if dt > 0:
                scale = max(0.5, min(2.0, self.LAUNCH_SECONDS / dt))
                self.iters = int(max(1, min(1 << 20, round(self.iters * scale))))


def open_devices(args):
    if args.cpu_lib:
        return [CpuDevice(args.cpu_lib, i) for i in range(args.cpu_devices)]
    cuda = Cuda()
    devs = []
    indices = range(cuda.device_count())
    if args.gpus:
        indices = [int(x) for x in args.gpus.split(",")]
    for i in indices:
        devs.append(CudaDevice(cuda, i))
    return devs


def cmd_selftest(args):
    try:
        devs = open_devices(args)
    except Exception as exc:
        print("SELFTEST FAIL init: %s" % exc)
        return 1
    if not devs:
        print("SELFTEST FAIL no GPUs")
        return 1
    ok_all = True
    for d in devs:
        try:
            ok, msg = selftest_device(d)
        except Exception as exc:
            ok, msg = False, "exception %s" % exc
        ok_all &= ok
        print("SELFTEST %s gpu%d %s: %s" % ("OK" if ok else "FAIL", d.index, d.describe(), msg))
    print("SELFTEST %s" % ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


def cmd_bench(args):
    devs = open_devices(args)
    shared = Shared()
    miners = [GpuMiner(d, shared, lambda *a: None) for d in devs]
    job = Job("bench", os.urandom(128), os.urandom(20), 0, os.urandom(16))
    for m in miners:
        m.start()
    shared.set_job(job)
    time.sleep(min(2.0, args.bench / 3.0))  # warm up + autotune
    start = [m.hashes for m in miners]
    t0 = time.time()
    time.sleep(args.bench)
    dt = time.time() - t0
    rates = [(m.hashes - s) / dt for m, s in zip(miners, start)]
    shared.stop = True
    for d, r, m in zip(devs, rates, miners):
        print("gpu%d %-40s %10.1f MH/s  (iters %d)" % (d.index, d.describe()[:40], r / 1e6, m.iters))
    print("total %.1f MH/s" % (sum(rates) / 1e6))
    print("BENCH %d %s" % (sum(rates), ",".join("%d" % r for r in rates)))
    return 0


def cmd_serve(args):
    try:
        devs = open_devices(args)
    except Exception as exc:
        emit("ERR init %s" % " ".join(str(exc).split()))
        return 2
    good = []
    for d in devs:
        try:
            ok, msg = selftest_device(d)
        except Exception as exc:
            ok, msg = False, str(exc)
        if ok:
            good.append(d)
            log("gpu%d %s selftest ok" % (d.index, d.describe()))
        else:
            emit("ERR gpu%d selftest failed: %s" % (d.index, " ".join(msg.split())))
    if not good:
        emit("ERR no usable GPU")
        return 2

    shared = Shared()

    def on_found(job, nonce, digest):
        emit("FOUND %s %s %s" % (job.id, nonce.hex(), digest.hex()))

    miners = [GpuMiner(d, shared, on_found) for d in good]
    for m in miners:
        m.start()
    emit("READY %d %s" % (len(good), json.dumps([d.name for d in good], separators=(",", ":"))))

    def reporter():
        last = [0] * len(miners)
        last_t = time.time()
        while not shared.stop:
            time.sleep(1.0)
            now = time.time()
            cur = [m.hashes for m in miners]
            dt = max(1e-6, now - last_t)
            rates = [(c - l) / dt for c, l in zip(cur, last)]
            last, last_t = cur, now
            emit("HR %d %s" % (sum(rates), ",".join("%d" % r for r in rates)))

    threading.Thread(target=reporter, daemon=True).start()

    lines = queue.Queue()

    def reader():
        for raw in sys.stdin:
            lines.put(raw)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    last_input = time.time()
    while True:
        try:
            raw = lines.get(timeout=1.0)
        except queue.Empty:
            if args.idle_timeout and time.time() - last_input > args.idle_timeout:
                emit("ERR no input for %ds, exiting" % args.idle_timeout)
                break
            continue
        if raw is None:
            break
        last_input = time.time()
        parts = raw.split()
        if not parts:
            continue
        cmd = parts[0]
        try:
            if cmd == "JOB" and len(parts) == 6:
                job = Job(parts[1], bytes.fromhex(parts[2]), bytes.fromhex(parts[3]),
                          int(parts[4], 16), bytes.fromhex(parts[5]))
                shared.set_job(job)
            elif cmd == "IDLE":
                shared.set_job(None)
            elif cmd == "PING":
                emit("PONG " + (parts[1] if len(parts) > 1 else ""))
            elif cmd == "QUIT":
                break
            else:
                emit("ERR bad command %s" % cmd[:20])
        except Exception as exc:
            emit("ERR %s: %s" % (cmd[:20], " ".join(str(exc).split())))
    shared.stop = True
    shared.set_job(None)
    os._exit(0)


def main():
    ap = argparse.ArgumentParser(description="UNICRED GPU worker")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", type=float, metavar="SECONDS")
    ap.add_argument("--gpus", help="comma separated GPU indices (default: all)")
    ap.add_argument("--idle-timeout", type=float, default=30.0,
                    help="exit if no line from the controller for this long (0 = never)")
    ap.add_argument("--cpu-lib", help=argparse.SUPPRESS)
    ap.add_argument("--cpu-devices", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.selftest:
        return cmd_selftest(args)
    if args.bench:
        return cmd_bench(args)
    return cmd_serve(args)


if __name__ == "__main__":
    sys.exit(main())
