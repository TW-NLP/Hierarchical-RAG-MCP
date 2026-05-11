"""Run the ColBERT-RAG baseline on single-service MCPBench queries.

Reproduces the ColBERT-RAG row of Table 2 (single-turn) from the Hi-RAG
paper. ColBERT-RAG performs flat late-interaction (MaxSim) retrieval over
the same tool corpus that Flat-RAG and Hi-RAG Stage 1 see; the only
difference is that it scores via per-token MaxSim instead of pooled
cosine.
"""

import os

from app.sig_mcp.sigmcp import SigMCP
from config import LLMConfig, PROJECT_DIR, prompt_en, prompt_zh


if __name__ == '__main__':
    sig_engine = SigMCP()
    model_name = 'Qwen3-32B'

    if 'Qwen3' in model_name:
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'Qwen3-32B')
        os.makedirs(save_parent, exist_ok=True)
        save_path = os.path.join(save_parent, 'sig_mcp_COLBERT.json')
        sig_engine.signal_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_Qwen3,
            rag_type='COLBERT',
            topk=1,
            prompt=prompt_zh,
        )

    elif 'QwQ' in model_name:
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'QwQ')
        os.makedirs(save_parent, exist_ok=True)
        save_path = os.path.join(save_parent, 'sig_mcp_COLBERT.json')
        sig_engine.signal_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_QwQ,
            rag_type='COLBERT',
            topk=1,
            prompt=prompt_zh,
        )

    elif 'qwen3_8b' in model_name:
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'qwen3_8b')
        os.makedirs(save_parent, exist_ok=True)
        save_path = os.path.join(save_parent, 'sig_mcp_COLBERT.json')
        sig_engine.signal_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_qwen3_8b,
            rag_type='COLBERT',
            topk=1,
            prompt=prompt_zh,
        )

    elif 'chatgpt' in model_name:
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'chatgpt')
        os.makedirs(save_parent, exist_ok=True)
        save_path = os.path.join(save_parent, 'sig_mcp_COLBERT.json')
        sig_engine.signal_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_chatgpt,
            rag_type='COLBERT',
            topk=1,
            prompt=prompt_en,
        )
