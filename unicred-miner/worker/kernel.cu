// UNICRED PoW search kernel (Keccak-256, second absorb block only).
//
// Message = 192 bytes = abi.encode(TYPEHASH, chainid, UNICRED, inner, sender, nonce).
// Block 1 (bytes 0..135) is fixed per job; the host absorbs it once and sends
// c_x = midstate ^ block2, where block2 already holds the sender tail, the
// nonce words 0..2 (lanes 3..5), zero in lane 6 and the Keccak padding.
// Per nonce the GPU does  a6 ^= bswap64(counter)  and one keccak-f[1600].
//
// The same file is compiled
//   * by NVRTC to PTX (tools/build_ptx.py) and embedded into worker/worker.py;
//   * by nvcc on a server as a fallback;
//   * by a plain C++ compiler (the #ifndef __CUDACC__ shims below) for CPU tests.

#ifndef __CUDACC__
#include <stdint.h>
#include <string.h>
#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(x)
struct dim3_t { unsigned int x, y, z; };
static dim3_t threadIdx, blockIdx, blockDim, gridDim;
static inline unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift) {
    uint64_t v = ((uint64_t)hi << 32) | lo;
    return (unsigned int)((v << (shift & 31)) >> 32);
}
static inline unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s) {
    uint64_t pool = ((uint64_t)y << 32) | x;
    unsigned int r = 0;
    for (int i = 0; i < 4; i++) {
        unsigned int sel = (s >> (4 * i)) & 7;
        r |= (unsigned int)((pool >> (8 * sel)) & 0xff) << (8 * i);
    }
    return r;
}
static inline unsigned long long atomicAdd(unsigned long long *p, unsigned long long v) {
    unsigned long long old = *p; *p = old + v; return old;
}
#endif

typedef unsigned long long u64;
typedef unsigned int u32;

#define MAX_FOUND 15

struct JobParams {
    u64 cx[25];   // midstate ^ block2, lane 6 without the counter
    u64 t0;       // target bits 255..192
    u64 t1;       // target bits 191..128
};

__device__ __forceinline__ u64 rol64_fs(u64 x, const int n) {
    const u32 lo = (u32)x, hi = (u32)(x >> 32);
    u32 nlo, nhi;
    if (n < 32) {
        nhi = __funnelshift_l(lo, hi, n);
        nlo = __funnelshift_l(hi, lo, n);
    } else {
        nhi = __funnelshift_l(hi, lo, n - 32);
        nlo = __funnelshift_l(lo, hi, n - 32);
    }
    return ((u64)nhi << 32) | nlo;
}
#define ROL64(x, n) rol64_fs((x), (n))

__device__ __forceinline__ u64 bswap64(u64 x) {
    const u32 lo = (u32)x, hi = (u32)(x >> 32);
    return ((u64)__byte_perm(lo, 0, 0x0123) << 32) | __byte_perm(hi, 0, 0x0123);
}

#define KECCAK_ROUND(RC) do { \
    c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    d0 = c4 ^ ROL64(c1, 1); \
    d1 = c0 ^ ROL64(c2, 1); \
    d2 = c1 ^ ROL64(c3, 1); \
    d3 = c2 ^ ROL64(c4, 1); \
    d4 = c3 ^ ROL64(c0, 1); \
    b0 = (a0 ^ d0); \
    b1 = ROL64((a6 ^ d1), 44); \
    b2 = ROL64((a12 ^ d2), 43); \
    b3 = ROL64((a18 ^ d3), 21); \
    b4 = ROL64((a24 ^ d4), 14); \
    b5 = ROL64((a3 ^ d3), 28); \
    b6 = ROL64((a9 ^ d4), 20); \
    b7 = ROL64((a10 ^ d0), 3); \
    b8 = ROL64((a16 ^ d1), 45); \
    b9 = ROL64((a22 ^ d2), 61); \
    b10 = ROL64((a1 ^ d1), 1); \
    b11 = ROL64((a7 ^ d2), 6); \
    b12 = ROL64((a13 ^ d3), 25); \
    b13 = ROL64((a19 ^ d4), 8); \
    b14 = ROL64((a20 ^ d0), 18); \
    b15 = ROL64((a4 ^ d4), 27); \
    b16 = ROL64((a5 ^ d0), 36); \
    b17 = ROL64((a11 ^ d1), 10); \
    b18 = ROL64((a17 ^ d2), 15); \
    b19 = ROL64((a23 ^ d3), 56); \
    b20 = ROL64((a2 ^ d2), 62); \
    b21 = ROL64((a8 ^ d3), 55); \
    b22 = ROL64((a14 ^ d4), 39); \
    b23 = ROL64((a15 ^ d0), 41); \
    b24 = ROL64((a21 ^ d1), 2); \
    a0 = b0 ^ (~b1 & b2); \
    a1 = b1 ^ (~b2 & b3); \
    a2 = b2 ^ (~b3 & b4); \
    a3 = b3 ^ (~b4 & b0); \
    a4 = b4 ^ (~b0 & b1); \
    a5 = b5 ^ (~b6 & b7); \
    a6 = b6 ^ (~b7 & b8); \
    a7 = b7 ^ (~b8 & b9); \
    a8 = b8 ^ (~b9 & b5); \
    a9 = b9 ^ (~b5 & b6); \
    a10 = b10 ^ (~b11 & b12); \
    a11 = b11 ^ (~b12 & b13); \
    a12 = b12 ^ (~b13 & b14); \
    a13 = b13 ^ (~b14 & b10); \
    a14 = b14 ^ (~b10 & b11); \
    a15 = b15 ^ (~b16 & b17); \
    a16 = b16 ^ (~b17 & b18); \
    a17 = b17 ^ (~b18 & b19); \
    a18 = b18 ^ (~b19 & b15); \
    a19 = b19 ^ (~b15 & b16); \
    a20 = b20 ^ (~b21 & b22); \
    a21 = b21 ^ (~b22 & b23); \
    a22 = b22 ^ (~b23 & b24); \
    a23 = b23 ^ (~b24 & b20); \
    a24 = b24 ^ (~b20 & b21); \
    a0 ^= (RC); \
} while (0)


// Keccak-f[1600] on c_x with lane 6 ^= bswap64(ctr).
// FULL=false computes only output lanes 0 and 1 in the last round.
template <bool FULL>
__device__ __forceinline__ void unicred_keccak(const JobParams &p, u64 ctr,
                                               u64 &o0, u64 &o1, u64 &o2, u64 &o3) {
    u64 a0 = p.cx[0], a1 = p.cx[1], a2 = p.cx[2], a3 = p.cx[3], a4 = p.cx[4];
    u64 a5 = p.cx[5], a6 = p.cx[6] ^ bswap64(ctr), a7 = p.cx[7], a8 = p.cx[8], a9 = p.cx[9];
    u64 a10 = p.cx[10], a11 = p.cx[11], a12 = p.cx[12], a13 = p.cx[13], a14 = p.cx[14];
    u64 a15 = p.cx[15], a16 = p.cx[16], a17 = p.cx[17], a18 = p.cx[18], a19 = p.cx[19];
    u64 a20 = p.cx[20], a21 = p.cx[21], a22 = p.cx[22], a23 = p.cx[23], a24 = p.cx[24];
    u64 c0, c1, c2, c3, c4, d0, d1, d2, d3, d4;
    u64 b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12;
    u64 b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24;

    KECCAK_ROUND(0x0000000000000001ULL);
    KECCAK_ROUND(0x0000000000008082ULL);
    KECCAK_ROUND(0x800000000000808aULL);
    KECCAK_ROUND(0x8000000080008000ULL);
    KECCAK_ROUND(0x000000000000808bULL);
    KECCAK_ROUND(0x0000000080000001ULL);
    KECCAK_ROUND(0x8000000080008081ULL);
    KECCAK_ROUND(0x8000000000008009ULL);
    KECCAK_ROUND(0x000000000000008aULL);
    KECCAK_ROUND(0x0000000000000088ULL);
    KECCAK_ROUND(0x0000000080008009ULL);
    KECCAK_ROUND(0x000000008000000aULL);
    KECCAK_ROUND(0x000000008000808bULL);
    KECCAK_ROUND(0x800000000000008bULL);
    KECCAK_ROUND(0x8000000000008089ULL);
    KECCAK_ROUND(0x8000000000008003ULL);
    KECCAK_ROUND(0x8000000000008002ULL);
    KECCAK_ROUND(0x8000000000000080ULL);
    KECCAK_ROUND(0x000000000000800aULL);
    KECCAK_ROUND(0x800000008000000aULL);
    KECCAK_ROUND(0x8000000080008081ULL);
    KECCAK_ROUND(0x8000000000008080ULL);
    KECCAK_ROUND(0x0000000080000001ULL);

    // Round 24: theta + rho/pi + chi restricted to row 0, then iota.
    c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;
    c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;
    c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;
    c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;
    c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;
    d0 = c4 ^ ROL64(c1, 1);
    d1 = c0 ^ ROL64(c2, 1);
    d2 = c1 ^ ROL64(c3, 1);
    d3 = c2 ^ ROL64(c4, 1);
    b0 = a0 ^ d0;
    b1 = ROL64(a6 ^ d1, 44);
    b2 = ROL64(a12 ^ d2, 43);
    b3 = ROL64(a18 ^ d3, 21);
    o0 = b0 ^ (~b1 & b2) ^ 0x8000000080008008ULL;
    o1 = b1 ^ (~b2 & b3);
    if (FULL) {
        d4 = c3 ^ ROL64(c0, 1);
        b4 = ROL64(a24 ^ d4, 14);
        o2 = b2 ^ (~b3 & b4);
        o3 = b3 ^ (~b4 & b0);
    } else {
        o2 = 0;
        o3 = 0;
    }
}

// Search: counter = base + i * (gridDim*blockDim) + gid, i < iters.
// A counter is reported when the top 128 bits of the digest are <= target's
// (the host does the exact 256-bit check). out[0] = number of hits,
// out[1..MAX_FOUND] = counters.
extern "C" __global__ void __launch_bounds__(256)
unicred_search(const JobParams p, const u64 base, const u32 iters, u64 *out) {
    const u64 stride = (u64)gridDim.x * blockDim.x;
    u64 ctr = base + (u64)blockIdx.x * blockDim.x + threadIdx.x;
    for (u32 i = 0; i < iters; i++, ctr += stride) {
        u64 o0, o1, o2, o3;
        unicred_keccak<false>(p, ctr, o0, o1, o2, o3);
        const u64 h0 = bswap64(o0);
        if (h0 <= p.t0) {
            const u64 h1 = bswap64(o1);
            if (h0 < p.t0 || h1 <= p.t1) {
                const u64 slot = atomicAdd(&out[0], 1ULL);
                if (slot < MAX_FOUND) out[1 + slot] = ctr;
            }
        }
    }
}

// Full digest of one counter (little-endian lanes 0..3), used by self-checks.
extern "C" __global__ void unicred_hash(const JobParams p, const u64 ctr, u64 *out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    u64 o0, o1, o2, o3;
    unicred_keccak<true>(p, ctr, o0, o1, o2, o3);
    out[0] = o0; out[1] = o1; out[2] = o2; out[3] = o3;
}

#ifndef __CUDACC__
// ---- CPU harness (plain C/C++ build, used by tests) ----
extern "C" void cpu_hash(const u64 *cx, u64 ctr, u64 *out4) {
    JobParams p; memcpy(p.cx, cx, sizeof(p.cx)); p.t0 = p.t1 = 0;
    blockIdx.x = threadIdx.x = 0;
    unicred_hash(p, ctr, out4);
}

extern "C" void cpu_prefix(const u64 *cx, u64 ctr, u64 *out2) {
    JobParams p; memcpy(p.cx, cx, sizeof(p.cx)); p.t0 = p.t1 = 0;
    u64 o2, o3;
    unicred_keccak<false>(p, ctr, out2[0], out2[1], o2, o3);
}

// Emulates a launch of unicred_search with the given geometry.
extern "C" void cpu_search(const u64 *cx, u64 t0, u64 t1, u64 base, u32 iters,
                           u32 grid, u32 block, u64 *out) {
    JobParams p; memcpy(p.cx, cx, sizeof(p.cx)); p.t0 = t0; p.t1 = t1;
    gridDim.x = grid; blockDim.x = block;
    for (u32 b = 0; b < grid; b++)
        for (u32 t = 0; t < block; t++) {
            blockIdx.x = b; threadIdx.x = t;
            unicred_search(p, base, iters, out);
        }
}
#endif
