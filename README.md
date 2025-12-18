
# Hi-RAG: Structure-Aware Tool Selection

<div align="center">
<img src="images/HiRAG.png" alt="HiRAG Framework" height="300">





<em>Official implementation of "Beyond Flat Retrieval: Structure-Aware Tool Selection with Hi-RAG"</em>
</div>

---

## 📖 Introduction

**Hi-RAG** (Hierarchical Retrieval-Augmented Generation) is a novel framework designed to operationalize structure awareness in Large Language Model (LLM) agents. By explicitly leveraging the hierarchical nature of tool protocols (e.g., Model Context Protocol), Hi-RAG creates a high-fidelity, low-noise context for efficient and accurate tool selection.

This repository contains:

1. **Hi-RAG Framework**: The source code for our structure-aware retrieval and reasoning pipeline.
2. **HiMCPBench**: A large-scale benchmark featuring **201 tools** across **40 services**, designed to evaluate complex, multi-service query reasoning.

---

## ⚙️ Environment Setup

### 1. Installation

We recommend using Conda to manage the environment.

```bash
# Create and activate environment
conda create -n rag_py312 -y python=3.12
conda activate rag_py312

# Install dependencies
pip install -r requirements.txt

```

### 2. LLM Backend Serving (vLLM)

Hi-RAG requires an LLM backend for inference and embedding models for retrieval. We utilize [vLLM](https://github.com/vllm-project/vllm) for high-performance serving.

> **Note:** Please verify your GPU memory availability. Adjust `--gpu-memory-utilization` as needed.

```bash
# 1. Start the Generator LLM (e.g., Qwen3-32B)
# Port: 38084
nohup vllm serve Qwen3-32B \
    --port 38084 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --seed 0 > qwen3_32b.log 2>&1 &

# 2. Start the Embedding Model (BGE-Large)
# Port: 8083
vllm serve BAAI/bge-large-en-v1.5 --task embed --port 8083

# 3. Start the Reranker Model (BGE-Reranker)
# Port: 8085
vllm serve BAAI/bge-reranker-base --task score --port 8085

```

---

## 🗂️ HiMCPBench & Dataset

Our benchmark, **HiMCPBench**, is meticulously constructed from real-world **Model Context Protocol (MCP)** specifications.

### Data Organization

* **Service Schemas** (`app/mcp_service/`): Contains definitions for **8 Categories**, **40 Services**, and **201 Tools**.
* **Evaluation Queries** (`data/query_test/`):
* `sig_mcp_test.json`: **Single-Service** queries (Precision focus).
* `mul_mcp_test.json`: **Multi-Service** queries (Complex reasoning focus).



### ⚠️ Note on Deterministic Evaluation

To ensure **scientific reproducibility** and **deterministic evaluation**, the services in this repository are implemented as **High-Fidelity Service Stubs**.

* **Semantic Layer:** We strictly preserve the *original* Pydantic models, function signatures, and semantic docstrings from real-world MCP registries (e.g., YouTube, GitHub, Slack) to maintain the full complexity of the reasoning task.
* **Execution Layer:** The backend execution logic is decoupled and simulated. This isolates the model's **tool selection capability** from external confounders such as network latency, API rate limits, or authentication barriers.

---

## 🚀 Usage & Evaluation

Please follow the steps below to reproduce the results reported in the paper.

### Step 1: Initialize MCP Ecosystem

Launch the simulated MCP server environment. This script initializes all 40 service instances.

```bash
cd scripts
bash server_start.sh
# This will start 40 independent service processes corresponding to the benchmark.

```

### Step 2: Run Evaluation

We provide automated scripts for both single-turn and multi-turn evaluation scenarios.

**Option A: Single-Service Evaluation**
Tests the model's precision in selecting the correct tool from a specific service.

```bash
bash sig_hi_test.sh

```

**Option B: Multi-Service Evaluation (HiMCPBench Main)**
Tests the model's ability to reason across multiple services and tools.

```bash
bash mul_hi_test.sh

```

### Step 3: Web Demonstration (Optional)

Launch a Gradio/Streamlit web interface to interact with Hi-RAG visually.

```bash
bash web.sh

```

### Step 4: Cleanup

After completing the evaluation, ensure all background service processes are terminated.

```bash
bash server_stop.sh

```

---

## 📂 Directory Structure

```text
Hi-RAG/
├── app/
│   └── mcp_service/     # 40 Service definitions (High-Fidelity Stubs)
│       ├── browser-automation/     # Example: Fetch Service
│       ├── calendar-management/      # Example: Time Service
│       └── ...
├── data/
│   └── query_test/      # Annotated Test Datasets
│       ├── sig_mcp_test.json
│       └── mul_mcp_test.json
├── images/              # Assets for README
├── scripts/             # Automation scripts
│   ├── server_start.sh  # Launch MCP ecosystem
│   ├── server_stop.sh   # Terminate MCP ecosystem
│   ├── sig_hi_test.sh   # Run Single-turn Eval
│   ├── mul_hi_test.sh   # Run Multi-turn Eval
│   └── web.sh           # Web Demo
├── requirements.txt     # Python dependencies
└── README.md            # Project Documentation

```

---

## 📧 Contact

For any questions regarding the code or the dataset, please open an issue in this repository.