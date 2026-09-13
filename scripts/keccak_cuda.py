#!/usr/bin/env python3
"""CUDA Keccak-256 search kernel.

The host hands the kernel the padded message as 17 little-endian lanes with a
zero nonce, plus where the two searched 32-bit halves of the nonce sit. Each
thread rewrites those halves and permutes, so nothing in the kernel depends on
the preimage layout.

A preimage under 136 bytes is one Keccak block, so a proof is one permutation.
"""

CUDA_SOURCE = r'''
typedef unsigned long long u64;
typedef unsigned int u32;

__device__ __forceinline__ u64 rol64(u64 x, int n) {
    return n == 0 ? x : (x << n) | (x >> (64 - n));
}

__device__ __forceinline__ u64 bswap64(u64 x) {
    return ((x & 0x00000000000000ffULL) << 56) | ((x & 0x000000000000ff00ULL) << 40) |
           ((x & 0x0000000000ff0000ULL) << 24) | ((x & 0x00000000ff000000ULL) << 8)  |
           ((x & 0x000000ff00000000ULL) >> 8)  | ((x & 0x0000ff0000000000ULL) >> 24) |
           ((x & 0x00ff000000000000ULL) >> 40) | ((x & 0xff00000000000000ULL) >> 56);
}

__device__ __forceinline__ u32 bswap32(u32 x) {
    return (x << 24) | ((x & 0x0000ff00u) << 8) | ((x & 0x00ff0000u) >> 8) | (x >> 24);
}

__device__ __forceinline__ void keccakf(u64 st[25]) {
    const u64 rndc[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
        0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };
    const int rotc[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
                          27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
    const int piln[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
                          15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
    u64 bc[5];
    for (int round = 0; round < 24; ++round) {
        #pragma unroll
        for (int i = 0; i < 5; ++i)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            u64 t = bc[(i + 4) % 5] ^ rol64(bc[(i + 1) % 5], 1);
            #pragma unroll
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        u64 t = st[1];
        #pragma unroll
        for (int i = 0; i < 24; ++i) {
            int j = piln[i];
            u64 next = st[j];
            st[j] = rol64(t, rotc[i]);
            t = next;
        }
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            #pragma unroll
            for (int i = 0; i < 5; ++i) bc[i] = st[j + i];
            #pragma unroll
            for (int i = 0; i < 5; ++i) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        st[0] ^= rndc[round];
    }
}

__device__ __forceinline__ void proof_hash(
    const u64 *lanes, int stream_lane, int stream_shift, int counter_lane, int counter_shift,
    u32 stream, u32 counter, u64 st[25]
) {
    #pragma unroll
    for (int i = 0; i < 17; ++i) st[i] = lanes[i];
    #pragma unroll
    for (int i = 17; i < 25; ++i) st[i] = 0;
    u64 stream_mask = ~(0xffffffffULL << stream_shift);
    st[stream_lane] = (st[stream_lane] & stream_mask) | ((u64)bswap32(stream) << stream_shift);
    u64 counter_mask = ~(0xffffffffULL << counter_shift);
    st[counter_lane] = (st[counter_lane] & counter_mask) | ((u64)bswap32(counter) << counter_shift);
    keccakf(st);
}

extern "C" __global__ void hash_one(
    const u64 *lanes, int stream_lane, int stream_shift, int counter_lane, int counter_shift,
    u32 stream, u32 counter, u64 *out
) {
    if (blockIdx.x || threadIdx.x) return;
    u64 st[25];
    proof_hash(lanes, stream_lane, stream_shift, counter_lane, counter_shift, stream, counter, st);
    out[0] = st[0]; out[1] = st[1]; out[2] = st[2]; out[3] = st[3];
}

extern "C" __global__ void mine_batch(
    const u64 *lanes, int stream_lane, int stream_shift, int counter_lane, int counter_shift,
    u32 stream, u32 counter_base, u32 iterations,
    u64 target0, u64 target1, u64 target2, u64 target3,
    int *found, u32 *found_counter, u64 *found_hash
) {
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u32 counter = counter_base + (u32)(tid * iterations);
    for (u32 i = 0; i < iterations; ++i, ++counter) {
        if (*found) return;
        u64 st[25];
        proof_hash(lanes, stream_lane, stream_shift, counter_lane, counter_shift,
                   stream, counter, st);
        // Keccak output is little-endian lanes; the contract reads it as a big-endian uint256.
        u64 h0 = bswap64(st[0]);
        u64 h1 = bswap64(st[1]);
        u64 h2 = bswap64(st[2]);
        u64 h3 = bswap64(st[3]);
        bool ok = h0 < target0 ||
                  (h0 == target0 && (h1 < target1 ||
                  (h1 == target1 && (h2 < target2 ||
                  (h2 == target2 && h3 < target3)))));
        if (ok && atomicCAS(found, 0, 1) == 0) {
            *found_counter = counter;
            found_hash[0] = st[0]; found_hash[1] = st[1];
            found_hash[2] = st[2]; found_hash[3] = st[3];
            return;
        }
    }
}
'''
