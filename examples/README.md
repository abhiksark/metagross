# Metagross GPU Demo

This demo drives the CUDA driver API directly through `ctypes` and launches a
vector-addition kernel on the GPU.

## Running the Demo

Run the demo bare (no tracing):

```sh
/usr/bin/python3 examples/gpu_demo.py
```

Run the demo under Metagross to trace GPU kernel calls:

```sh
sudo /usr/bin/python3 -m metagross examples/gpu_demo.py
```

## Other Examples

- **quicklook.py** is a smaller GPU workload with clearly named steps, meant
  to produce a readable trace. Run it bare with
  `PYTHONPATH=. /usr/bin/python3 examples/quicklook.py` (no root required;
  `PYTHONPATH` is unnecessary once Metagross is installed with pip), or traced
  with `sudo /usr/bin/python3 -m metagross examples/quicklook.py`. Its
  `run_kernel()` call runs inside `with metagross.span("compute"):`, so a
  traced run's events for that step carry `"compute"` in their `span` field;
  see [op spans](../docs/reference.md#op-spans).
- **run_integration.sh** (repository root) runs the root-gated live
  integration suite (`test_metagross.LiveTraceTest`) under sudo with
  `RUN_EBPF_INTEGRATION=1` set.
- **docker/** contains a containerized Metagross + PyTorch environment with
  tensor, CNN, ViT, decoder-only transformer, training, and multi-stage pipeline
  workloads, and the `metagross` command that runs it on a script in the
  current directory. See [`docker/README.md`](docker/README.md) for the required
  NVIDIA, host-PID, and BPF flags.

## Function Overview

The demo is structured into project functions that serve as integration-test
assertions:

- **setup_gpu()** exercises `cuInit`, `cuDeviceGet`, and `cuCtxCreate_v2` to
  initialize the GPU and establish a CUDA context on device 0.
- **load_kernel()** exercises `cuModuleLoadData` and `cuModuleGetFunction` to
  compile the embedded PTX and obtain the `vec_add` kernel function.
- **upload()** exercises `cuMemAlloc_v2` and `cuMemcpyHtoD_v2` to allocate device
  memory and transfer inputs from host to GPU.
- **compute()** exercises `cuLaunchKernel` and `cuStreamSynchronize` with a block
  size of 128 and waits for completion.
- **download_and_check()** exercises `cuMemcpyDtoH_v2` and verifies that every
  output element is `3.0`.
- **teardown()** exercises `cuMemFree_v2`, `cuModuleUnload`, and
  `cuCtxDestroy_v2` to release resources.
