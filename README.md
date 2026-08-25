<a name="readme-top"></a>

<div align="center">
  <h1 align="center" style="border-bottom: none">🚀 High-Performance LLM Inference & Serving Engine</h1>
  <p align="center">
    <strong>Ultra-fast, memory-efficient Large Language Model (LLM) serving engine powered by PagedAttention and CUDA kernel optimization.</strong>
  </p>
</div>

<div align="center">
  <a href="https://github.com/NadeemAhmad3"><img src="https://img.shields.io/badge/Author-Nadeem%20Ahmad-blue?style=for-the-badge&logo=github" alt="Author"></a>
  <a href="mailto:engrnadeem26@gmail.com"><img src="https://img.shields.io/badge/Email-engrnadeem26%40gmail.com-red?style=for-the-badge&logo=gmail" alt="Email"></a>
  <a href="https://github.com/NadeemAhmad3/AI-Engineering-Lab"><img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge" alt="Status"></a>
</div>

<hr>

## 📌 Executive Overview

This repository houses a high-throughput, low-latency **Large Language Model (LLM) Inference Engine**. Designed for scale, it solves the fundamental memory fragmentation and throughput bottlenecks of standard PyTorch LLM serving via **PagedAttention**, **Continuous Iteration Batching**, **Prefix Caching**, and **Custom CUDA Kernels**.

```text
               ┌────────────────────────────────────────────────────────┐
               │         OpenAI / REST API Ingress Server               │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │         Async LLM Engine & Scheduler                   │
               │   (Continuous Batching & PagedAttention Virtual Mem)   │
               └───────────────────────────┬────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│ Paged KV Cache   │             │ Custom CUDA      │             │ Quantization     │
│ Block Allocator  │             │ Kernels (csrc/)  │             │ (FP8/INT8/AWQ)   │
└──────────────────┘             └──────────────────┘             └──────────────────┘
```

---

## 🔥 Key Technical Features

### 1. 🧠 PagedAttention Memory Management
Eliminates up to **96% of GPU memory waste** by managing Key-Value (KV) attention caches using OS-style virtual memory paging. It breaks KV-caches into fixed-size block pages, enabling **2×–4× larger batch sizes** on identical VRAM hardware.

### 2. ⚡ Continuous Iteration-Level Batching
Instead of waiting for the slowest sequence in a static batch to finish, requests are batched dynamically at the token iteration level. New requests enter the execution pipeline instantly on the next token step.

### 3. 🎯 Prefix Caching & Chunked Prefill
Automatically caches and reuses Key-Value attention states for recurring prompt prefixes (e.g. system prompts or long document context), cutting prefill latency by up to **80%**.

### 4. 🚀 Quantization & Hardware Acceleration
- **Native Quantization Support**: FP8, INT8, INT4, AWQ, GPTQ, GGUF, and NVFP4.
- **Custom CUDA/C++ Kernels**: Optimized attention and GEMM/MoE kernels (`csrc/`) utilizing FlashAttention-2, FlashInfer, and CUTLASS.

### 5. 🌐 OpenAI-Compatible API Server
Includes a production HTTP server supporting OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`), streaming output, structured JSON generation, and tool-calling parsers.

---

## 📂 Repository Architecture

```text
vllm/
├── vllm/                      # Core Python Engine & Orchestration
│   ├── engine/                # AsyncLLMEngine & scheduler logic
│   ├── entrypoints/           # OpenAI / Anthropic / gRPC Compatible API Servers
│   ├── model_executor/        # PyTorch model definitions & parallel execution
│   └── worker/                # GPU/CPU worker processes & PagedAttention managers
├── csrc/                      # High-Performance C++ & CUDA Kernels
│   ├── attention/             # PagedAttention & FlashAttention CUDA kernels
│   ├── pos_encoding/          # Rotary Positional Embedding (RoPE) CUDA kernels
│   └── quantization/          # AWQ / GPTQ / FP8 GEMM C++ kernels
├── rust/                      # Fast Rust modules for tokenization & server utilities
├── benchmarks/                # Throughput & Latency Benchmarking Scripts
├── examples/                  # Python API & server usage examples
└── docs/                      # Technical documentation & architectural guides
```

---

## ⚡ Quickstart Guide

### 1. Installation

Install via `pip` or `uv`:

```bash
pip install vllm
```

Or build from source for development:

```bash
python setup.py develop
```

### 2. Python API Usage

```python
from vllm import LLM, SamplingParams

# Define prompt batch
prompts = [
    "Explain the concept of PagedAttention in AI engineering:",
    "Write a Python function for binary search:"
]

# Set sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

# Initialize engine
llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Run high-throughput inference
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}\n")
```

### 3. Launching OpenAI-Compatible API Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --port 8000
```

Test the API endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 👤 Maintainer & Author Information

* **Author**: **Nadeem Ahmad**
* **GitHub**: [@NadeemAhmad3](https://github.com/NadeemAhmad3)
* **Email**: [engrnadeem26@gmail.com](mailto:engrnadeem26@gmail.com)
* **Repository**: [https://github.com/NadeemAhmad3/AI-Engineering-Lab](https://github.com/NadeemAhmad3/AI-Engineering-Lab)

---

## 📄 License

Distributed under the Apache 2.0 License. See `LICENSE` for details.
