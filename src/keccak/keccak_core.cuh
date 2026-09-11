// Keccak-256 core shared by the CUDA kernel and the CPU reference build.
// Compiled as __device__ code by nvcc and as plain C++ by g++ (tests/test_keccak_host.cpp),
// so the GPU and the CPU verifier can never drift apart.
#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
#define HC_FN __device__ __forceinline__
#define HC_RC_MEM __device__ __constant__
#else
#define HC_FN static inline
#define HC_RC_MEM static const
#endif

#define HC_RATE 136 // Keccak-256 rate in bytes: the largest preimage we support is 135 bytes

HC_RC_MEM uint64_t hc_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

HC_FN uint64_t hc_rotl64(uint64_t x, int n) {
  return (x << n) | (x >> (64 - n));
}

// In-place Keccak-f[1600] permutation.
HC_FN void hc_keccak_f(uint64_t st[25]) {
  // Function-local constexpr, not __constant__: these index the state array, so the
  // values must be compile-time known or the unrolled loops spill st[] to local memory.
  constexpr int rotc[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                            27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};
  constexpr int piln[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                            15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};
  uint64_t bc[5], t;

#pragma unroll 1
  for (int round = 0; round < 24; round++) {
    // Theta
#pragma unroll
    for (int i = 0; i < 5; i++)
      bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
#pragma unroll
    for (int i = 0; i < 5; i++) {
      t = bc[(i + 4) % 5] ^ hc_rotl64(bc[(i + 1) % 5], 1);
#pragma unroll
      for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
    }

    // Rho + Pi
    t = st[1];
#pragma unroll
    for (int i = 0; i < 24; i++) {
      int j = piln[i];
      bc[0] = st[j];
      st[j] = hc_rotl64(t, rotc[i]);
      t = bc[0];
    }

    // Chi
#pragma unroll
    for (int j = 0; j < 25; j += 5) {
#pragma unroll
      for (int i = 0; i < 5; i++) bc[i] = st[j + i];
#pragma unroll
      for (int i = 0; i < 5; i++)
        st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }

    // Iota
    st[0] ^= hc_rc[round];
  }
}

// Pad a message of len <= 135 bytes into the 17 rate lanes of a single block.
// Keccak (not SHA3) padding: 0x01 ... 0x80, exactly as used by Solidity's keccak256.
HC_FN void hc_pad_block(const uint8_t *msg, int len, uint64_t lanes[17]) {
  uint8_t buf[HC_RATE];
  for (int i = 0; i < HC_RATE; i++) buf[i] = 0;
  for (int i = 0; i < len; i++) buf[i] = msg[i];
  buf[len] = 0x01;
  buf[HC_RATE - 1] |= 0x80;
#pragma unroll
  for (int i = 0; i < 17; i++) {
    uint64_t v = 0;
#pragma unroll
    for (int b = 7; b >= 0; b--) v = (v << 8) | (uint64_t)buf[i * 8 + b];
    lanes[i] = v;
  }
}

// Absorb one already-padded block and return the 32-byte digest.
HC_FN void hc_keccak256_block(const uint64_t lanes[17], uint8_t out[32]) {
  uint64_t st[25];
#pragma unroll
  for (int i = 0; i < 25; i++) st[i] = 0;
#pragma unroll
  for (int i = 0; i < 17; i++) st[i] = lanes[i];
  hc_keccak_f(st);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint64_t v = st[i];
#pragma unroll
    for (int b = 0; b < 8; b++) out[i * 8 + b] = (uint8_t)(v >> (8 * b));
  }
}

// Same, but returns the first four digest words as big-endian integers so a
// target comparison needs no byte shuffling.
HC_FN void hc_keccak256_block_be(const uint64_t lanes[17], uint64_t out_be[4]) {
  uint64_t st[25];
#pragma unroll
  for (int i = 0; i < 25; i++) st[i] = 0;
#pragma unroll
  for (int i = 0; i < 17; i++) st[i] = lanes[i];
  hc_keccak_f(st);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint64_t v = st[i], r = 0;
#pragma unroll
    for (int b = 0; b < 8; b++) r = (r << 8) | ((v >> (8 * b)) & 0xffULL);
    out_be[i] = r;
  }
}

// Convenience wrapper for messages of at most 135 bytes.
HC_FN void hc_keccak256(const uint8_t *msg, int len, uint8_t out[32]) {
  uint64_t lanes[17];
  hc_pad_block(msg, len, lanes);
  hc_keccak256_block(lanes, out);
}
