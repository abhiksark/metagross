# examples/gpu_demo.py
"""Metagross demo: drives the CUDA driver API directly via ctypes."""
import ctypes

N = 1024
PTX = rb"""
.version 7.0
.target sm_70
.address_size 64

.visible .entry vec_add(
    .param .u64 pA, .param .u64 pB, .param .u64 pC, .param .u32 pN
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<11>;

    ld.param.u64 %rd1, [pA];
    ld.param.u64 %rd2, [pB];
    ld.param.u64 %rd3, [pC];
    ld.param.u32 %r1, [pN];
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


def setup_gpu():
    check(cuda.cuInit(0), "cuInit")
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")
    return context


def load_kernel():
    module = ctypes.c_void_p()
    check(cuda.cuModuleLoadData(ctypes.byref(module), PTX), "cuModuleLoadData")
    func = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(func), module, b"vec_add"),
          "cuModuleGetFunction")
    return module, func


def upload():
    nbytes = N * 4
    host_a = (ctypes.c_float * N)(*[1.0] * N)
    host_b = (ctypes.c_float * N)(*[2.0] * N)
    dev = []
    for host in (host_a, host_b, None):
        ptr = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes), "cuMemAlloc")
        if host is not None:
            check(cuda.cuMemcpyHtoD_v2(ptr, host, nbytes), "cuMemcpyHtoD")
        dev.append(ptr)
    return dev


def compute(func, dev):
    d_a, d_b, d_c = dev
    n = ctypes.c_uint32(N)
    params = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.byref(d_a), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_b), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_c), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n), ctypes.c_void_p))
    block = 128
    grid = (N + block - 1) // block
    check(cuda.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, None,
                              params, None), "cuLaunchKernel")
    check(cuda.cuStreamSynchronize(None), "cuStreamSynchronize")


def download_and_check(dev):
    out = (ctypes.c_float * N)()
    check(cuda.cuMemcpyDtoH_v2(out, dev[2], N * 4), "cuMemcpyDtoH")
    assert all(abs(v - 3.0) < 1e-6 for v in out), "vec_add result wrong"


def teardown(context, module, dev):
    for ptr in dev:
        check(cuda.cuMemFree_v2(ptr), "cuMemFree")
    check(cuda.cuModuleUnload(module), "cuModuleUnload")
    check(cuda.cuCtxDestroy_v2(context), "cuCtxDestroy")


def main():
    context = setup_gpu()
    module, func = load_kernel()
    dev = upload()
    compute(func, dev)
    download_and_check(dev)
    teardown(context, module, dev)
    print(f"vec_add ok: {N} elements")


if __name__ == "__main__":
    main()
