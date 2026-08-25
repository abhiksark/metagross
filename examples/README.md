# Metagross GPU Demo

This demo illustrates how to drive the CUDA driver API directly via ctypes, launching a vector addition kernel on the GPU.

## Running the Demo

Run the demo bare (no tracing):

```sh
/usr/bin/python3 examples/gpu_demo.py
```

Run the demo under Metagross to trace GPU kernel calls:

```sh
sudo /usr/bin/python3 -m metagross examples/gpu_demo.py
```

## Function Overview

The demo is structured into project functions that serve as integration test assertions:

- **setup_gpu()** exercises `cuInit`, `cuDeviceGet`, and `cuCtxCreate_v2` to initialize the GPU and establish a CUDA context on device 0.
- **load_kernel()** exercises `cuModuleLoadData` and `cuModuleGetFunction` to compile the embedded PTX code and extract the vec_add kernel function.
- **upload()** exercises `cuMemAlloc_v2` and `cuMemcpyHtoD_v2` to allocate device memory and transfer input arrays (A=1.0, B=2.0) from host to GPU, plus allocate output buffer C.
- **compute()** exercises `cuLaunchKernel` and `cuStreamSynchronize` to configure grid and thread blocks (grid = ceil(N/128), block = 128) and launch the kernel, waiting for completion.
- **download_and_check()** exercises `cuMemcpyDtoH_v2` to transfer the output array from device back to host and validates that all elements are 3.0 (A+B).
- **teardown()** exercises `cuMemFree_v2`, `cuModuleUnload`, and `cuCtxDestroy_v2` to release allocated device memory, unload the module, and destroy the CUDA context.
