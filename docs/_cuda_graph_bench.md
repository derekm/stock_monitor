# _cuda_graph_bench.py

**Developer scratch** — not part of the production pipeline.

Benchmark for CUDA-graph capture during TinyTimeMixer (TTM) training. It builds
AEP-only full-history windows (up to 2000), stacks them into context/target
tensors, loads a Granite model, and times forward/backward passes with and
without `torch.cuda.graph` capture to see whether graph capture helps on the
MX550.

Companion to `_cuda_bench.py`. No persistent outputs; prints timing to stdout.
