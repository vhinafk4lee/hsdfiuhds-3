// CPU build of the same Keccak core the GPU kernel uses.
// Reads hex-encoded messages (one per line) on stdin, prints keccak256 hex per line.
#include "../src/keccak/keccak_core.cuh"

#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>

static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (line.size() % 2 != 0 || line.size() / 2 > 135) {
      std::cerr << "bad input length\n";
      return 1;
    }
    uint8_t msg[135];
    int len = (int)line.size() / 2;
    for (int i = 0; i < len; i++)
      msg[i] = (uint8_t)((hexval(line[2 * i]) << 4) | hexval(line[2 * i + 1]));

    uint8_t out[32];
    hc_keccak256(msg, len, out);
    char hex[65];
    for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", out[i]);
    hex[64] = 0;
    printf("%s\n", hex);
    fflush(stdout);
  }
  return 0;
}
