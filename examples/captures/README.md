# Sanitized example capture

`basic.jsonl` and `basic-summary.json` preserve the existing local basic-tensor
capture as a reproducible viewer fixture. They contain 17 CUDA driver API events
from three project functions, with only neutral `/workspace/workloads/...` source
paths. The matching version-1 summary reports a complete capture and target exit
status zero. These historical measurements are not a performance benchmark or
proof that live tracing works on the current checkout or host.

Replay without root, BCC, CUDA, or a GPU:

```sh
/usr/bin/python3 -m metagross view --web \
  --summary examples/captures/basic-summary.json \
  examples/captures/basic.jsonl
```

Open the private URL printed by the server. The public capture contains no
viewer or producer token. Stop the server after inspection and do not publish
its private URL.
