// Fake CUDA driver for tests: implements the subset of the Driver API used by
// worker/worker.py and executes the kernels from worker/kernel.cu on the CPU.
// It checks what the real driver would get: PTX image text, kernel names,
// kernelParams layout (struct by value, u64, u32, CUdeviceptr).
//   FAKE_CUDA_MAX_PTX=7.0   reject newer PTX ISA with CUDA_ERROR_UNSUPPORTED_PTX_VERSION
//   FAKE_CUDA_DEVICES=N     number of devices (default 2)
#include "../worker/kernel.cu"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
enum { F_SEARCH = 1, F_HASH = 2 };
struct FakeFunc { int kind; };
static FakeFunc g_search = {F_SEARCH}, g_hash = {F_HASH};
static int g_module_dummy;
static int g_ctx_dummy[16];

extern "C" {
CUresult cuInit(unsigned int) { return 0; }
CUresult cuGetErrorName(CUresult, const char **s) { *s = "FAKE_ERROR"; return 0; }
CUresult cuDriverGetVersion(int *v) { *v = 12040; return 0; }
CUresult cuDeviceGetCount(int *n) {
    const char *e = getenv("FAKE_CUDA_DEVICES"); *n = e ? atoi(e) : 2; return 0;
}
CUresult cuDeviceGet(int *dev, int ordinal) { *dev = ordinal; return 0; }
CUresult cuDeviceGetName(char *name, int len, int dev) {
    snprintf(name, len, "Fake GPU %d", dev); return 0;
}
CUresult cuDeviceGetAttribute(int *v, int attr, int) {
    switch (attr) {
        case 16: *v = 2; break;   // SM count
        case 75: *v = 8; break;   // cc major
        case 76: *v = 9; break;   // cc minor
        default: *v = 0;
    }
    return 0;
}
CUresult cuDevicePrimaryCtxSetFlags_v2(int, unsigned int flags) { return flags == 4 ? 0 : 1; }
CUresult cuDevicePrimaryCtxRetain(void **ctx, int dev) { *ctx = &g_ctx_dummy[dev & 15]; return 0; }
CUresult cuCtxCreate_v2(void **ctx, unsigned int, int dev) { *ctx = &g_ctx_dummy[dev & 15]; return 0; }
CUresult cuCtxSetCurrent(void *) { return 0; }

CUresult cuModuleLoadDataEx(void **mod, const void *image, unsigned int n, int *opts, void **vals) {
    const char *ptx = (const char *)image;
    const char *ver = strstr(ptx, "\n.version ");
    if (!ver || !strstr(ptx, ".target sm_52") || !strstr(ptx, ".entry unicred_search(")) return 218;
    double v = atof(ver + 10);
    const char *maxv = getenv("FAKE_CUDA_MAX_PTX");
    if (maxv && v > atof(maxv) + 1e-9) {
        for (unsigned i = 0; i < n; i++)
            if (opts[i] == 5) snprintf((char *)vals[i], 64, "ptx version %.1f unsupported", v);
        return 222;
    }
    *mod = &g_module_dummy;
    return 0;
}
CUresult cuModuleGetFunction(void **f, void *, const char *name) {
    if (!strcmp(name, "unicred_search")) { *f = &g_search; return 0; }
    if (!strcmp(name, "unicred_hash")) { *f = &g_hash; return 0; }
    return 500;
}
CUresult cuFuncGetAttribute(int *v, int attr, void *) { *v = attr == 4 ? 78 : 0; return 0; }
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *n, void *, int, size_t) { *n = 1; return 0; }
CUresult cuMemAlloc_v2(unsigned long long *p, size_t size) {
    *p = (unsigned long long)calloc(1, size); return 0;
}
CUresult cuMemsetD8_v2(unsigned long long p, unsigned char c, size_t n) { memset((void *)p, c, n); return 0; }
CUresult cuMemcpyDtoH_v2(void *dst, unsigned long long src, size_t n) { memcpy(dst, (void *)src, n); return 0; }

CUresult cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                        unsigned bz, unsigned shmem, void *stream, void **params, void **extra) {
    if (gy != 1 || gz != 1 || by != 1 || bz != 1 || shmem || extra) return 1;
    FakeFunc *fn = (FakeFunc *)f;
    const JobParams *p = (const JobParams *)params[0];
    gridDim.x = gx; blockDim.x = bx;
    if (fn->kind == F_SEARCH) {
        u64 base = *(u64 *)params[1];
        u32 iters = *(u32 *)params[2];
        u64 *out = (u64 *)*(unsigned long long *)params[3];
        for (unsigned b = 0; b < gx; b++)
            for (unsigned t = 0; t < bx; t++) {
                blockIdx.x = b; threadIdx.x = t;
                unicred_search(*p, base, iters, out);
            }
        return 0;
    }
    if (fn->kind == F_HASH) {
        u64 ctr = *(u64 *)params[1];
        u64 *out = (u64 *)*(unsigned long long *)params[2];
        blockIdx.x = threadIdx.x = 0;
        unicred_hash(*p, ctr, out);
        return 0;
    }
    return 1;
}
}
