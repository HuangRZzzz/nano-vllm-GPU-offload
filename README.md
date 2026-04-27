# Nano-vLLM-OFFLOAD

A lightweight vLLM-style inference engine with CPU-GPU hybrid offload support.

This project is based on `nano-vllm` and keeps the codebase small and readable while adding offload-oriented inference features.

## Key Features

* **Weight offload on `master`** - The `master` branch refers to the `llama.cpp` style weight offload design and supports CPU-GPU hybrid inference.
* **KV cache offload on `GPU_OFFLOAD`** - The `GPU_OFFLOAD` branch supports vLLM-style KV cache offload, moving cold KV blocks between GPU and CPU.
* **Readable inference engine** - A compact Python implementation of vLLM-style offline inference.
* **Continuous batching** - Supports batching requests dynamically during generation.
* **Chunked prefill** - Long prompts can be prefetched in chunks instead of requiring one huge prefill batch.
* **Prefix caching** - Reuses already computed prefix KV blocks when possible.
* **TP=1 KV offload path** - The current KV offload implementation targets tensor parallel size 1.


## Branch Design

### `master`: Weight Offload

The `master` branch focuses on weight offload.

It follows the idea used by `llama.cpp`: only part of the model weights need to stay on GPU, while other layers can remain on CPU and be moved or activated according to the configured offload policy.

This enables CPU-GPU hybrid inference when GPU memory is limited.

Typical goals:

* Reduce static GPU memory usage from model weights.
* Keep hot or currently needed layers on GPU.
* Allow small-memory GPUs to run models that would otherwise exceed VRAM.
* Provide a simple CPU-GPU mixed inference path.

### `GPU_OFFLOAD`: KV Cache Offload

The `GPU_OFFLOAD` branch adds vLLM-style KV cache offload.

Unlike old sequence-level preemption, this implementation does not preempt and recompute whole requests. Instead, it treats KV cache blocks as logical blocks. When GPU KV cache space is not enough, cold GPU-resident blocks are selected by LRU and offloaded to CPU.

Core behavior:

* `seq.block_table` stores logical KV block ids.
* Each logical block may have a GPU physical block id and/or a CPU physical block id.
* Before attention runs, logical block ids are translated into GPU physical block ids.
* If a needed logical block is only on CPU, it is loaded back to GPU first.
* If GPU KV space is full, the least recently used unprotected GPU block is copied to CPU and its GPU slot is released.
* Active blocks required by the current batch are protected and will not be evicted.


install the original upstream project:

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```
| Hardware | Model | Mode | Output Speed |
| --- | --- | --- | ---: |
| RTX 4060 | Qwen3-4B | CPU-GPU offload | 1-2 tok/s |

