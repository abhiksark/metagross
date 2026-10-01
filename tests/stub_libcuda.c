// tests/stub_libcuda.c
// A stand-in for libcuda.so.1 with no GPU behind it. tests/test_stub_live.py
// traces a script that calls these functions, so the live tracing loop can be
// exercised on a machine that has root and BCC but no NVIDIA driver.
//
// Build without optimization: identical function bodies must keep separate
// addresses, or one uprobe would fire for several APIs.
#include <stddef.h>

#define EXPORT __attribute__((visibility("default"), noinline))

// Lets the test refuse to run against a real driver library.
int metagross_stub_libcuda = 1;

static unsigned long long next_device_pointer = 0x700000000000ULL;

EXPORT int cuMemAlloc_v2(unsigned long long *dptr, size_t bytes) {
    *dptr = next_device_pointer;
    next_device_pointer += bytes;
    return 0;
}

EXPORT int cuMemFree_v2(unsigned long long dptr) {
    (void)dptr;
    return 0;
}

EXPORT int cuMemcpyHtoD_v2(unsigned long long dst, const void *src, size_t bytes) {
    (void)dst;
    (void)src;
    (void)bytes;
    return 0;
}

EXPORT int cuModuleGetFunction(void **function, void *module, const char *name) {
    (void)module;
    (void)name;
    *function = (void *)0x5000;
    return 0;
}

EXPORT int cuLaunchKernel(void *function,
                          unsigned grid_x, unsigned grid_y, unsigned grid_z,
                          unsigned block_x, unsigned block_y, unsigned block_z,
                          unsigned shared_bytes, void *stream,
                          void **params, void **extra) {
    (void)function;
    (void)grid_x;
    (void)grid_y;
    (void)grid_z;
    (void)block_x;
    (void)block_y;
    (void)block_z;
    (void)shared_bytes;
    (void)stream;
    (void)params;
    (void)extra;
    return 0;
}

EXPORT int cuGraphLaunch(void *graph_exec, void *stream) {
    (void)graph_exec;
    (void)stream;
    return 0;
}

EXPORT int cuCtxSynchronize(void) {
    return 0;
}
