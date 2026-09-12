#!/usr/bin/env python3
"""CUDA SHA-256 search kernel.

The host hands the kernel a padded message (up to two 64-byte blocks) with a
zero nonce, plus the indices of the two 32-bit words that carry the searched
nonce tail. Each thread rewrites those two words and hashes, so the kernel is
independent of where the nonce sits inside the preimage.
"""

CUDA_SOURCE = r'''
typedef unsigned int u32;

__constant__ u32 K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

__device__ __forceinline__ u32 rotr32(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ u32 small0(u32 x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
__device__ __forceinline__ u32 small1(u32 x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }
__device__ __forceinline__ u32 big0(u32 x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
__device__ __forceinline__ u32 big1(u32 x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }

__device__ __forceinline__ void compress(const u32 *block, u32 *state) {
    u32 w[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = block[i];
    u32 a = state[0], b = state[1], c = state[2], d = state[3];
    u32 e = state[4], f = state[5], g = state[6], h = state[7];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        u32 word;
        if (i < 16) {
            word = w[i];
        } else {
            word = w[i & 15] + small0(w[(i + 1) & 15]) + w[(i + 9) & 15] + small1(w[(i + 14) & 15]);
            w[i & 15] = word;
        }
        u32 t1 = h + big1(e) + ((e & f) ^ ((~e) & g)) + K[i] + word;
        u32 t2 = big0(a) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ __forceinline__ void sha256_message(
    const u32 *message, int blocks, int stream_word, int counter_word,
    u32 stream, u32 counter, u32 *state
) {
    state[0] = 0x6a09e667u; state[1] = 0xbb67ae85u;
    state[2] = 0x3c6ef372u; state[3] = 0xa54ff53au;
    state[4] = 0x510e527fu; state[5] = 0x9b05688cu;
    state[6] = 0x1f83d9abu; state[7] = 0x5be0cd19u;
    for (int block = 0; block < blocks; ++block) {
        u32 words[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int index = block * 16 + i;
            u32 value = message[index];
            if (index == stream_word) value = stream;
            if (index == counter_word) value = counter;
            words[i] = value;
        }
        compress(words, state);
    }
}

__device__ __forceinline__ void sha256_state(const u32 *state, u32 *out) {
    u32 block[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) block[i] = state[i];
    block[8] = 0x80000000u;
    #pragma unroll
    for (int i = 9; i < 15; ++i) block[i] = 0u;
    block[15] = 256u;
    out[0] = 0x6a09e667u; out[1] = 0xbb67ae85u;
    out[2] = 0x3c6ef372u; out[3] = 0xa54ff53au;
    out[4] = 0x510e527fu; out[5] = 0x9b05688cu;
    out[6] = 0x1f83d9abu; out[7] = 0x5be0cd19u;
    compress(block, out);
}

__device__ __forceinline__ void proof_hash(
    const u32 *message, int blocks, int doubled, int stream_word, int counter_word,
    u32 stream, u32 counter, u32 *digest
) {
    u32 state[8];
    sha256_message(message, blocks, stream_word, counter_word, stream, counter, state);
    if (doubled) {
        sha256_state(state, digest);
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) digest[i] = state[i];
    }
}

extern "C" __global__ void hash_one(
    const u32 *message, int blocks, int doubled, int stream_word, int counter_word,
    u32 stream, u32 counter, u32 *out
) {
    if (blockIdx.x || threadIdx.x) return;
    u32 digest[8];
    proof_hash(message, blocks, doubled, stream_word, counter_word, stream, counter, digest);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = digest[i];
}

extern "C" __global__ void mine_batch(
    const u32 *message, int blocks, int doubled, int stream_word, int counter_word,
    u32 stream, u32 counter_base, u32 iterations,
    const u32 *target,
    int *found, u32 *found_counter, u32 *found_hash
) {
    __shared__ u32 shared_message[32];
    __shared__ u32 shared_target[8];
    if (threadIdx.x < 32) shared_message[threadIdx.x] = message[threadIdx.x];
    if (threadIdx.x < 8) shared_target[threadIdx.x] = target[threadIdx.x];
    __syncthreads();

    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    u32 counter = counter_base + (u32)(tid * iterations);
    for (u32 i = 0; i < iterations; ++i, ++counter) {
        if (*found) return;
        u32 digest[8];
        proof_hash(shared_message, blocks, doubled, stream_word, counter_word,
                   stream, counter, digest);
        bool ok = false;
        #pragma unroll
        for (int word = 0; word < 8; ++word) {
            if (digest[word] != shared_target[word]) {
                ok = digest[word] < shared_target[word];
                break;
            }
        }
        if (ok && atomicCAS(found, 0, 1) == 0) {
            *found_counter = counter;
            #pragma unroll
            for (int word = 0; word < 8; ++word) found_hash[word] = digest[word];
            return;
        }
    }
}
'''
