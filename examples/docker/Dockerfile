# examples/docker/Dockerfile
# Version-pinned Metagross + PyTorch userspace for a CUDA-capable Linux host.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TARGET_UID=1000
ARG TARGET_GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-bpfcc \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# pip 25.1+ resumes interrupted downloads; the CUDA wheels are hundreds of MB
# each, and without resume a dropped connection fails the build's hash check.
RUN python3 -m pip install --no-cache-dir "pip==26.2.1"
RUN python3 -m pip install --no-cache-dir --resume-retries 50 \
        "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
RUN python3 -m pip install --no-cache-dir --resume-retries 50 "numpy==1.26.4"

# The controller remains root for BPF access. Metagross validates these sudo
# values and drops only the traced target to this unprivileged account.
RUN groupadd --gid "${TARGET_GID}" metagross-target \
    && useradd --uid "${TARGET_UID}" --gid "${TARGET_GID}" \
        --create-home --shell /bin/bash metagross-target

ENV PYTHONUNBUFFERED=1 \
    SUDO_UID=${TARGET_UID} \
    SUDO_GID=${TARGET_GID} \
    SUDO_USER=metagross-target

WORKDIR /workspace
COPY metagross/ /workspace/metagross/
COPY examples/docker/workloads/ /workspace/workloads/
COPY examples/docker/benchmarks/ /workspace/benchmarks/
COPY examples/docker/tests/ /workspace/tests/
RUN chmod -R a+rX /workspace

# -c, not -m: the working directory may be the user's mounted project, and -m
# would put it ahead of this image's Metagross on the root controller's path.
ENTRYPOINT ["/usr/bin/python3", "-c", "import sys; sys.path[0] = '/workspace'; import metagross; sys.exit(metagross.main())"]
CMD ["--project-root", "/workspace/workloads", "/workspace/workloads/basic_tensor_ops.py"]
