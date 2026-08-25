# examples/quicklook.py
"""A tiny GPU workload with clearly named steps, so a Metagross trace reads well.

Run it traced to watch each CUDA driver call get attributed to the function
that made it:

    sudo /usr/bin/python3 -m metagross examples/quicklook.py

Or run it bare (no tracing, no root) to check it works:

    /usr/bin/python3 examples/quicklook.py
"""
import ctypes

N = 256
# Minimal PTX kernel: c[i] = a[i] + b[i].
PTX = rb"""
.version 7.0
.target sm_70
.address_size 64

.visible .entry add_vectors(
    .param .u64 a, .param .u64 b, .param .u64 c, .param .u32 n
)
{
    .reg .pred %p1;
    .reg .b32 %r<6>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<11>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [c];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.s32 %p1, %r5, %r1;
    @%p1 bra DONE;
    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;
    mul.wide.s32 %rd7, %r5, 4;
    add.s64 %rd8, %rd4, %rd7;
    add.s64 %rd9, %rd5, %rd7;
    add.s64 %rd10, %rd6, %rd7;
    ld.global.f32 %f1, [%rd8];
    ld.global.f32 %f2, [%rd9];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd10], %f3;
DONE:
    ret;
}
""" + b"\x00"

cuda = ctypes.CDLL("libcuda.so.1")


def check(result, what):
    if result != 0:
        raise RuntimeError(f"{what} failed: CUresult {result}")


def start_gpu():
    check(cuda.cuInit(0), "cuInit")
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")
    module = ctypes.c_void_p()
    check(cuda.cuModuleLoadData(ctypes.byref(module), PTX), "cuModuleLoadData")
    func = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(func), module, b"add_vectors"),
          "cuModuleGetFunction")
    return context, module, func


def send_inputs():
    nbytes = N * 4
    host_a = (ctypes.c_float * N)(*[1.5] * N)
    host_b = (ctypes.c_float * N)(*[2.5] * N)
    buffers = []
    for host in (host_a, host_b, None):
        ptr = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes), "cuMemAlloc")
        if host is not None:
            check(cuda.cuMemcpyHtoD_v2(ptr, host, nbytes), "cuMemcpyHtoD")
        buffers.append(ptr)
    return buffers


def run_kernel(func, buffers):
    d_a, d_b, d_c = buffers
    n = ctypes.c_uint32(N)
    params = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.byref(d_a), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_b), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_c), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n), ctypes.c_void_p))
    block = 64
    grid = (N + block - 1) // block
    check(cuda.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, None,
                              params, None), "cuLaunchKernel")
    check(cuda.cuStreamSynchronize(None), "cuStreamSynchronize")


def fetch_result(buffers):
    out = (ctypes.c_float * N)()
    check(cuda.cuMemcpyDtoH_v2(out, buffers[2], N * 4), "cuMemcpyDtoH")
    assert all(abs(v - 4.0) < 1e-6 for v in out), "add_vectors result wrong"
    for ptr in buffers:
        check(cuda.cuMemFree_v2(ptr), "cuMemFree")


def main():
    context, module, func = start_gpu()
    buffers = send_inputs()
    run_kernel(func, buffers)
    fetch_result(buffers)
    check(cuda.cuModuleUnload(module), "cuModuleUnload")
    check(cuda.cuCtxDestroy_v2(context), "cuCtxDestroy")
    print(f"add_vectors ok: {N} elements, each 1.5 + 2.5 = 4.0")


if __name__ == "__main__":
    main()
