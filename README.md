# HiRAG

<div align="center">
  <img src="images/HiRAG.png" alt="HiRAG" height="300">
</div>

---

## 📝 Introduction

**HiRAG** (Hierarchical Retrieval-Augmented Generation) is a hierarchical multi-service Retrieval-Augmented Generation framework designed to enhance large language models' tool selection and utilization in complex tasks.  

---

## 🔧 Environment Setup

### 1. Install Conda

Please refer to the official documentation: [Miniconda Installation](https://docs.conda.io/en/latest/miniconda.html)

### 2. Create and Activate Environment

```bash
conda create -n hi_rag -y python=3.12
conda activate hi_rag
pip install -r requirements.txt
```

---

## 🚀 Launch Large Language Models (LLMs)

Use [vLLM](https://github.com/vllm-project/vllm) to start different model services:

```bash
# Qwen3-32B
nohup vllm serve Qwen3-32B \
    --port 38084 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --seed 0 > qwen3_32b.log 2>&1 &

# QwQ-32B
nohup vllm serve QwQ-32B \
    --port 38080 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --seed 0 > qwq.log 2>&1 &

# Qwen3-8B
nohup vllm serve Qwen3_8B \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --seed 0 > qwen3_8b.log 2>&1 &
```

---

## 🧪 Experiments

This project supports three experiment settings: Flat-RAG, Full Service, and HiRAG.

### 1. Full Service

| Scenario | Command |
|----------|---------|
| Single-Service (sig) | `python run_sig.py` |
| Multi-Service (mul)  | `python run_mul.py` |

---

### 2. Flat RAG

| Scenario | Command |
|----------|---------|
| Single-Service (sig) | `python run_sig_rag_top1.py` |
| Multi-Service (mul)  | `python run_mul_rag_top3.py` |

---

### 3. HiRAG (Proposed Method)

| Scenario | Command |
|----------|---------|
| Single-Service (sig, share) | `python run_sig_Hi_rag_top1.py` |
| Multi-Service (mul)         | `python run_mul_Hi_rag_top3.py` |

---
