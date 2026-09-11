// hcminer-gpu: multi-GPU keccak256 nonce search.
//
// The preimage layout is NOT hard-coded here. The supervisor (Python) resolves the
// contract's exact hashing scheme, then hands this process a byte template plus the
// offset of the 8 bytes that the GPU varies. That keeps the kernel valid for any
// keccak256(field, field, ...) scheme with a contiguous nonce.
//
// Protocol: one JSON object per line on stdin, one per line on stdout.
//   in : {"cmd":"job","id":1,"preimage":"<hex, <=135 bytes>","vary_offset":36,
//         "target":"<64 hex>","nonce_start":"0x0"}
//        {"cmd":"bench","seconds":10}   {"cmd":"stop"}
//   out: {"type":"ready","devices":4}
//        {"type":"status","job":1,"hashrate":4.1e9,"total":123456789}
//        {"type":"solution","job":1,"nonce":"0x...","hash":"0x..."}
//        {"type":"error","message":"..."}

#include "../keccak/keccak_core.cuh"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------- device side

__constant__ uint64_t c_lanes[17];      // padded preimage block, nonce bytes zeroed
__constant__ uint64_t c_target[4];      // big-endian target words
__constant__ uint32_t c_nonce_lane0;    // lane holding the first varying nonce bytes
__constant__ uint32_t c_nonce_lane1;    // lane holding the remainder (== lane0 if aligned)
__constant__ uint32_t c_nonce_shift;    // bit shift of the nonce inside lane0

struct Solution {
  unsigned long long nonce;
  uint64_t hash_be[4];
  int found;
};

__device__ __forceinline__ uint64_t hc_bswap64(uint64_t v) {
  return ((v & 0x00000000000000ffULL) << 56) | ((v & 0x000000000000ff00ULL) << 40) |
         ((v & 0x0000000000ff0000ULL) << 24) | ((v & 0x00000000ff000000ULL) << 8) |
         ((v & 0x000000ff00000000ULL) >> 8)  | ((v & 0x0000ff0000000000ULL) >> 24) |
         ((v & 0x00ff000000000000ULL) >> 40) | ((v & 0xff00000000000000ULL) >> 56);
}

__global__ void mine_kernel(unsigned long long nonce_base, uint32_t inner,
                            Solution *sol) {
  const unsigned long long stride =
      (unsigned long long)blockDim.x * gridDim.x;
  const unsigned long long gid =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t lanes[17];
#pragma unroll
  for (int i = 0; i < 17; i++) lanes[i] = c_lanes[i];

  const uint32_t shift = c_nonce_shift;
  const uint32_t lane0 = c_nonce_lane0;
  const uint32_t lane1 = c_nonce_lane1;

  for (uint32_t k = 0; k < inner; k++) {
    const unsigned long long nonce = nonce_base + gid + (unsigned long long)k * stride;

    // The nonce is stored big-endian at a byte offset, i.e. little-endian lane order
    // after a byte swap. Splicing it in with selects rather than an indexed write
    // keeps the whole state in registers.
    const uint64_t w = hc_bswap64(nonce);
    const uint64_t p0 = w << shift;
    const uint64_t p1 = shift ? (w >> (64 - shift)) : 0ULL;

    uint64_t st[25];
#pragma unroll
    for (int i = 0; i < 17; i++) {
      uint64_t v = lanes[i];
      v |= ((uint32_t)i == lane0) ? p0 : 0ULL;
      v |= ((uint32_t)i == lane1) ? p1 : 0ULL;
      st[i] = v;
    }
#pragma unroll
    for (int i = 17; i < 25; i++) st[i] = 0;

    hc_keccak_f(st);

    uint64_t h[4];
#pragma unroll
    for (int i = 0; i < 4; i++) h[i] = hc_bswap64(st[i]);

    // 256-bit big-endian compare: hash < target
    bool below = false;
    if (h[0] != c_target[0]) below = h[0] < c_target[0];
    else if (h[1] != c_target[1]) below = h[1] < c_target[1];
    else if (h[2] != c_target[2]) below = h[2] < c_target[2];
    else below = h[3] < c_target[3];

    if (below) {
      if (atomicCAS(&sol->found, 0, 1) == 0) {
        sol->nonce = nonce;
#pragma unroll
        for (int i = 0; i < 4; i++) sol->hash_be[i] = h[i];
      }
      return;
    }
  }
}

// ------------------------------------------------------------------ host side

static std::string json_str(const std::string &line, const char *key) {
  std::string pat = std::string("\"") + key + "\"";
  size_t p = line.find(pat);
  if (p == std::string::npos) return "";
  p = line.find(':', p + pat.size());
  if (p == std::string::npos) return "";
  size_t a = line.find('"', p);
  if (a == std::string::npos) return "";
  size_t b = line.find('"', a + 1);
  if (b == std::string::npos) return "";
  return line.substr(a + 1, b - a - 1);
}

static unsigned long long json_num(const std::string &line, const char *key,
                                   unsigned long long fallback) {
  std::string pat = std::string("\"") + key + "\"";
  size_t p = line.find(pat);
  if (p == std::string::npos) return fallback;
  p = line.find(':', p + pat.size());
  if (p == std::string::npos) return fallback;
  p++;
  while (p < line.size() && (line[p] == ' ' || line[p] == '"')) p++;
  if (p >= line.size()) return fallback;
  int base = 10;
  if (line.compare(p, 2, "0x") == 0 || line.compare(p, 2, "0X") == 0) {
    base = 16;
    p += 2;
  }
  return strtoull(line.c_str() + p, nullptr, base);
}

static bool hex_to_bytes(const std::string &hex, std::vector<uint8_t> &out) {
  std::string h = hex;
  if (h.compare(0, 2, "0x") == 0 || h.compare(0, 2, "0X") == 0) h = h.substr(2);
  if (h.size() % 2) return false;
  out.clear();
  for (size_t i = 0; i < h.size(); i += 2) {
    auto val = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      return -1;
    };
    int hi = val(h[i]), lo = val(h[i + 1]);
    if (hi < 0 || lo < 0) return false;
    out.push_back((uint8_t)((hi << 4) | lo));
  }
  return true;
}

struct Job {
  uint64_t id = 0;
  std::vector<uint8_t> preimage;
  uint32_t vary_offset = 0;
  uint64_t target[4] = {0, 0, 0, 0};
  unsigned long long nonce_start = 0;
  bool valid = false;
};

static std::mutex g_job_mu;
static Job g_job;
static std::atomic<uint64_t> g_job_gen{0};
static std::atomic<bool> g_stop{false};
static std::atomic<unsigned long long> g_cursor{0};
static std::atomic<unsigned long long> g_hashes{0};
static std::mutex g_out_mu;

static void emit(const std::string &s) {
  std::lock_guard<std::mutex> lk(g_out_mu);
  std::cout << s << "\n" << std::flush;
}

// Build the padded block and the nonce byte map, then push both to the device.
static bool upload_job(const Job &job) {
  if (job.preimage.size() > 135) {
    emit("{\"type\":\"error\",\"message\":\"preimage longer than 135 bytes\"}");
    return false;
  }
  if (job.vary_offset + 8 > job.preimage.size()) {
    emit("{\"type\":\"error\",\"message\":\"vary_offset outside preimage\"}");
    return false;
  }

  std::vector<uint8_t> msg = job.preimage;
  for (int i = 0; i < 8; i++) msg[job.vary_offset + i] = 0; // kernel ORs the nonce in

  uint8_t buf[HC_RATE];
  memset(buf, 0, sizeof(buf));
  memcpy(buf, msg.data(), msg.size());
  buf[msg.size()] = 0x01;
  buf[HC_RATE - 1] |= 0x80;

  uint64_t lanes[17];
  for (int i = 0; i < 17; i++) {
    uint64_t v = 0;
    for (int b = 7; b >= 0; b--) v = (v << 8) | (uint64_t)buf[i * 8 + b];
    lanes[i] = v;
  }

  // The 8 varying bytes span at most two lanes; tell the kernel where they land.
  const uint32_t lane0 = job.vary_offset / 8;
  const uint32_t shift = 8 * (job.vary_offset % 8);
  const uint32_t lane1 = shift ? lane0 + 1 : lane0;
  if (lane1 > 16) {
    emit("{\"type\":\"error\",\"message\":\"nonce crosses the end of the block\"}");
    return false;
  }

  cudaMemcpyToSymbol(c_lanes, lanes, sizeof(lanes));
  cudaMemcpyToSymbol(c_target, job.target, sizeof(job.target));
  cudaMemcpyToSymbol(c_nonce_lane0, &lane0, sizeof(lane0));
  cudaMemcpyToSymbol(c_nonce_lane1, &lane1, sizeof(lane1));
  cudaMemcpyToSymbol(c_nonce_shift, &shift, sizeof(shift));
  return true;
}

struct Tuning {
  uint32_t threads = 256;
  uint32_t blocks = 0;        // 0 = derive from occupancy
  uint32_t blocks_mult = 1;   // extra waves on top of full occupancy
  uint32_t inner = 256;
  uint32_t streams = 4;
  // Windows resets the driver if a kernel on the display GPU runs longer than the
  // TDR timeout (2s by default), so launches are kept comfortably shorter and the
  // work per launch adapts to whatever the card turns out to be capable of.
#ifdef _WIN32
  double max_kernel_ms = 400.0;
#else
  double max_kernel_ms = 0.0;   // 0 = no cap
#endif
};

// Grid size that fills every SM, for the block size actually in use.
static uint32_t occupancy_blocks(uint32_t threads, const cudaDeviceProp &prop) {
  int blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, mine_kernel,
                                                (int)threads, 0);
  if (blocks_per_sm <= 0) blocks_per_sm = 16;
  return (uint32_t)blocks_per_sm * (uint32_t)prop.multiProcessorCount;
}

// One host thread per GPU. Several CUDA streams are kept in flight at once: while
// one batch is being copied back and inspected, the next is already running, so the
// device never waits on the host between launches.
static void device_worker(int device, Tuning tune) {
  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    emit(std::string("{\"type\":\"error\",\"message\":\"cudaSetDevice: ") +
         cudaGetErrorString(err) + "\"}");
    return;
  }

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  const uint32_t threads = tune.threads;
  const uint32_t blocks =
      tune.blocks ? tune.blocks : occupancy_blocks(threads, prop) * tune.blocks_mult;
  const uint32_t nstreams = tune.streams ? tune.streams : 1;

  std::vector<cudaStream_t> streams(nstreams);
  std::vector<Solution *> d_sol(nstreams, nullptr);
  std::vector<Solution *> h_sol(nstreams, nullptr);
  std::vector<cudaEvent_t> ev_start(nstreams), ev_stop(nstreams);
  std::vector<unsigned long long> in_flight_batch(nstreams, 0);
  std::vector<bool> busy(nstreams, false);

  for (uint32_t i = 0; i < nstreams; i++) {
    cudaStreamCreate(&streams[i]);
    cudaMalloc(&d_sol[i], sizeof(Solution));
    // Pinned host memory: the result copy back does not stall the stream.
    cudaHostAlloc(&h_sol[i], sizeof(Solution), cudaHostAllocDefault);
    memset(h_sol[i], 0, sizeof(Solution));
    cudaEventCreate(&ev_start[i]);
    cudaEventCreate(&ev_stop[i]);
  }

  // Adapted at runtime when a kernel duration cap is in force.
  uint32_t inner = tune.inner;
  const uint32_t inner_min = 16, inner_max = 1u << 20;

  uint64_t seen_gen = 0;
  Job local;

  auto launch = [&](uint32_t i) {
    const unsigned long long batch =
        (unsigned long long)threads * blocks * inner;
    const unsigned long long base = local.nonce_start + g_cursor.fetch_add(batch);
    cudaEventRecord(ev_start[i], streams[i]);
    cudaMemsetAsync(d_sol[i], 0, sizeof(Solution), streams[i]);
    mine_kernel<<<blocks, threads, 0, streams[i]>>>(base, inner, d_sol[i]);
    cudaMemcpyAsync(h_sol[i], d_sol[i], sizeof(Solution), cudaMemcpyDeviceToHost,
                    streams[i]);
    cudaEventRecord(ev_stop[i], streams[i]);
    in_flight_batch[i] = batch;
    busy[i] = true;
  };

  while (!g_stop.load()) {
    uint64_t gen = g_job_gen.load();
    if (gen != seen_gen) {
      // In-flight kernels read the job from constant memory, so let them finish
      // before it is overwritten.
      cudaDeviceSynchronize();
      for (uint32_t i = 0; i < nstreams; i++) busy[i] = false;
      {
        std::lock_guard<std::mutex> lk(g_job_mu);
        local = g_job;
      }
      seen_gen = gen;
      if (!local.valid || !upload_job(local)) {
        local.valid = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
    }
    if (!local.valid) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    bool progressed = false;
    for (uint32_t i = 0; i < nstreams && !g_stop.load(); i++) {
      if (!busy[i]) {
        launch(i);
        progressed = true;
        continue;
      }
      cudaError_t state = cudaStreamQuery(streams[i]);
      if (state == cudaErrorNotReady) continue;
      if (state != cudaSuccess) {
        emit(std::string("{\"type\":\"error\",\"message\":\"kernel: ") +
             cudaGetErrorString(state) + "\"}");
        g_stop.store(true);
        break;
      }

      busy[i] = false;
      progressed = true;
      g_hashes.fetch_add(in_flight_batch[i]);

      if (tune.max_kernel_ms > 0) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, ev_start[i], ev_stop[i]) == cudaSuccess && ms > 0) {
          // Halve on overshoot, grow slowly while well inside the budget.
          if (ms > tune.max_kernel_ms && inner > inner_min)
            inner = inner / 2 < inner_min ? inner_min : inner / 2;
          else if (ms < tune.max_kernel_ms * 0.4 && inner < inner_max)
            inner = inner * 2 > inner_max ? inner_max : inner * 2;
        }
      }

      const Solution sol = *h_sol[i];
      if (sol.found && g_job_gen.load() == seen_gen) {
        char hash_hex[67];
        snprintf(hash_hex, sizeof(hash_hex), "0x%016llx%016llx%016llx%016llx",
                 (unsigned long long)sol.hash_be[0], (unsigned long long)sol.hash_be[1],
                 (unsigned long long)sol.hash_be[2], (unsigned long long)sol.hash_be[3]);
        char out[256];
        snprintf(out, sizeof(out),
                 "{\"type\":\"solution\",\"job\":%llu,\"device\":%d,\"nonce\":\"0x%016llx\",\"hash\":\"%s\"}",
                 (unsigned long long)local.id, device, sol.nonce, hash_hex);
        emit(out);
      }
      if (!g_stop.load()) launch(i);
    }

    // Every stream is still running: yield briefly instead of spinning a core.
    if (!progressed) std::this_thread::sleep_for(std::chrono::microseconds(200));
  }

  cudaDeviceSynchronize();
  for (uint32_t i = 0; i < nstreams; i++) {
    cudaEventDestroy(ev_start[i]);
    cudaEventDestroy(ev_stop[i]);
    cudaStreamDestroy(streams[i]);
    cudaFree(d_sol[i]);
    cudaFreeHost(h_sol[i]);
  }
}

static void reporter() {
  auto last = std::chrono::steady_clock::now();
  unsigned long long last_hashes = 0;
  while (!g_stop.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    auto now = std::chrono::steady_clock::now();
    unsigned long long total = g_hashes.load();
    double secs = std::chrono::duration<double>(now - last).count();
    if (secs <= 0) continue;
    double rate = (double)(total - last_hashes) / secs;
    last = now;
    last_hashes = total;
    uint64_t id;
    {
      std::lock_guard<std::mutex> lk(g_job_mu);
      id = g_job.id;
    }
    char out[192];
    snprintf(out, sizeof(out),
             "{\"type\":\"status\",\"job\":%llu,\"hashrate\":%.0f,\"total\":%llu}",
             (unsigned long long)id, rate, total);
    emit(out);
  }
}

// Run one timed search with a given configuration and return hashes per second.
static double run_bench(const std::vector<int> &devices, Tuning tune, double seconds) {
  g_stop.store(false);
  g_hashes.store(0);
  g_cursor.store(0);

  std::vector<std::thread> workers;
  for (int d : devices) workers.emplace_back(device_worker, d, tune);

  // Ignore the first stretch: streams are still filling up.
  const double warmup = seconds * 0.35;
  std::this_thread::sleep_for(std::chrono::duration<double>(warmup));
  const unsigned long long start_hashes = g_hashes.load();
  const auto t0 = std::chrono::steady_clock::now();

  std::this_thread::sleep_for(std::chrono::duration<double>(seconds - warmup));
  const double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count();
  const unsigned long long measured = g_hashes.load() - start_hashes;

  g_stop.store(true);
  for (auto &t : workers) t.join();
  return secs > 0 ? (double)measured / secs : 0.0;
}

static void set_dummy_job() {
  // A target of 1 is unreachable, so a benchmark never stops early on a "solution".
  Job j;
  j.id = 0;
  j.preimage.assign(116, 0xab);
  j.vary_offset = 20;
  j.target[0] = 0; j.target[1] = 0; j.target[2] = 0; j.target[3] = 1;
  j.nonce_start = 0;
  j.valid = true;
  {
    std::lock_guard<std::mutex> lk(g_job_mu);
    g_job = j;
  }
  g_job_gen.fetch_add(1);
}

// Sweep the launch parameters that matter and report the fastest combination.
static int autotune(const std::vector<int> &devices, double per_config) {
  set_dummy_job();

  const uint32_t thread_opts[] = {128, 256, 512};
  const uint32_t mult_opts[] = {1, 2};
  const uint32_t inner_opts[] = {256, 1024};
  const uint32_t stream_opts[] = {2, 4};

  Tuning best;
  double best_rate = 0;
  int tested = 0;

  for (uint32_t threads : thread_opts)
    for (uint32_t mult : mult_opts)
      for (uint32_t inner : inner_opts)
        for (uint32_t streams : stream_opts) {
          Tuning tune;
          tune.threads = threads;
          tune.blocks_mult = mult;
          tune.inner = inner;
          tune.streams = streams;

          const double rate = run_bench(devices, tune, per_config);
          tested++;
          char out[256];
          snprintf(out, sizeof(out),
                   "{\"type\":\"tune_result\",\"threads\":%u,\"blocks_mult\":%u,"
                   "\"inner\":%u,\"streams\":%u,\"hashrate\":%.0f}",
                   threads, mult, inner, streams, rate);
          emit(out);

          if (rate > best_rate) {
            best_rate = rate;
            best = tune;
          }
        }

  char out[256];
  snprintf(out, sizeof(out),
           "{\"type\":\"tune\",\"tested\":%d,\"threads\":%u,\"blocks_mult\":%u,"
           "\"inner\":%u,\"streams\":%u,\"hashrate\":%.0f}",
           tested, best.threads, best.blocks_mult, best.inner, best.streams, best_rate);
  emit(out);
  return best_rate > 0 ? 0 : 1;
}

int main(int argc, char **argv) {
  std::vector<int> devices;
  Tuning tune;
  double bench_seconds = 0;
  double tune_seconds = 0;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--devices") {
      std::string v = next(), cur;
      for (char c : v + ",") {
        if (c == ',') {
          if (!cur.empty()) devices.push_back(atoi(cur.c_str()));
          cur.clear();
        } else cur.push_back(c);
      }
    } else if (a == "--threads") tune.threads = (uint32_t)atoi(next().c_str());
    else if (a == "--blocks") tune.blocks = (uint32_t)atoi(next().c_str());
    else if (a == "--blocks-mult") tune.blocks_mult = (uint32_t)atoi(next().c_str());
    else if (a == "--inner") tune.inner = (uint32_t)atoi(next().c_str());
    else if (a == "--streams") tune.streams = (uint32_t)atoi(next().c_str());
    else if (a == "--max-kernel-ms") tune.max_kernel_ms = atof(next().c_str());
    else if (a == "--bench") bench_seconds = atof(next().c_str());
    else if (a == "--autotune") {
      std::string v = next();
      tune_seconds = v.empty() ? 2.0 : atof(v.c_str());
      if (tune_seconds <= 0) tune_seconds = 2.0;
    } else if (a == "--list-devices") {
      int n = 0;
      cudaGetDeviceCount(&n);
      for (int d = 0; d < n; d++) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, d);
        printf("device %d: %s, sm_%d%d, %d SMs, %.1f GB\n", d, prop.name, prop.major,
               prop.minor, prop.multiProcessorCount,
               (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
      }
      return 0;
    } else if (a == "--help") {
      printf("usage: hcminer-gpu [--devices 0,1,2,3] [--threads N] [--blocks N]\n"
             "                   [--blocks-mult N] [--inner N] [--streams N]\n"
             "                   [--max-kernel-ms MS] [--bench SECONDS]\n"
             "                   [--autotune SECONDS] [--list-devices]\n");
      return 0;
    }
  }

  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    emit("{\"type\":\"error\",\"message\":\"no CUDA devices found\"}");
    return 1;
  }
  if (devices.empty())
    for (int i = 0; i < count; i++) devices.push_back(i);

  char ready[96];
  snprintf(ready, sizeof(ready), "{\"type\":\"ready\",\"devices\":%zu}", devices.size());
  emit(ready);

  if (tune_seconds > 0) return autotune(devices, tune_seconds);

  if (bench_seconds > 0) {
    set_dummy_job();
    const double rate = run_bench(devices, tune, bench_seconds);
    char out[192];
    snprintf(out, sizeof(out),
             "{\"type\":\"bench\",\"seconds\":%.2f,\"hashes\":%llu,\"hashrate\":%.0f}",
             bench_seconds, (unsigned long long)g_hashes.load(), rate);
    emit(out);
    return 0;
  }

  std::vector<std::thread> workers;
  for (int d : devices) workers.emplace_back(device_worker, d, tune);
  std::thread rep(reporter);

  std::string line;
  while (std::getline(std::cin, line)) {
    std::string cmd = json_str(line, "cmd");
    if (cmd == "stop") break;
    if (cmd != "job") continue;

    Job j;
    j.id = json_num(line, "id", 0);
    if (!hex_to_bytes(json_str(line, "preimage"), j.preimage)) {
      emit("{\"type\":\"error\",\"message\":\"bad preimage hex\"}");
      continue;
    }
    j.vary_offset = (uint32_t)json_num(line, "vary_offset", 0);
    std::vector<uint8_t> tgt;
    if (!hex_to_bytes(json_str(line, "target"), tgt) || tgt.size() != 32) {
      emit("{\"type\":\"error\",\"message\":\"target must be 32 bytes\"}");
      continue;
    }
    for (int w = 0; w < 4; w++) {
      uint64_t v = 0;
      for (int b = 0; b < 8; b++) v = (v << 8) | tgt[w * 8 + b];
      j.target[w] = v;
    }
    j.nonce_start = json_num(line, "nonce_start", 0);
    j.valid = true;

    {
      std::lock_guard<std::mutex> lk(g_job_mu);
      g_job = j;
    }
    g_cursor.store(0);
    g_job_gen.fetch_add(1);
  }
  g_stop.store(true);

  for (auto &t : workers) t.join();
  rep.join();
  return 0;
}
