<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM Extension for Qwen3.5 Multimodal Model

A compact, readable inference engine based on nano-vllm, extended for Qwen3.5 multimodal model.

This repository is intended for learning how LLM inference engines work: tensor parallelism, KV cache allocation, CUDA Graph decode, Qwen vision encoder are all implemented in a small codebase.

## What Works

- Qwen3.5-2B/Qwen3.5-4B multimodal inference.
- Qwen3.5-4B/Qwen3.5-9B multimodal inference on 2 x RTX 3090.
- Tensor parallelism for attention, MLP, vocabulary embedding/head, gated delta net and Qwen3VL vision encoder when `tensor_parallel_size=2`.
- CUDA Graph decode when `enforce_eager=False`.
- Optimization for some operators based on triton
- Picture and video input

## Model Download

To download the model weights manually, use the following command:
```bash
git clone https://www.modelscope.cn/Qwen/Qwen3.5-9B.git
```

## Installation

Use Python 3.10-3.12 with CUDA-capable PyTorch. Real inference requires GPUs plus
`torch`, `triton`,  `flash-attn`.

## Quick Start

See `example.py` for usage. 
```bash
python example.py
```

## Benchmark

See `bench.py` for benchmark.
```bash
python example.py
```

**Test Configuration:**
- Hardware: 1 x RTX 3090
- Model: Qwen3.5-4B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 165.32    | 810.31              |
| Nano-vLLM      | 133,966     | 236.18    | 567.22               |

**Test Configuration:**
- Hardware: 2 x RTX 3090
- Model: Qwen3.5-9B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 217.35    | 616.34              |
| Nano-vLLM      | 133,966     | 272.56    | 491.52               |

## Acknowledgements

This work builds on the original Nano-vLLM project by Xingkai Yu. The Qwen3.5 model weights are distributed separately by Alibaba under their own
model terms.