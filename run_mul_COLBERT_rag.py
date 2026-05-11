"""Run the ColBERT-RAG baseline on multi-service MCPBench queries.

Reproduces the ColBERT-RAG row of Table 2 (multi-turn / Top-3) from the
Hi-RAG paper.
"""

import os

from app.mul_mcp.mulmcp import MulMCP
from config import LLMConfig, PROJECT_DIR, prompt_en, prompt_zh


if __name__ == '__main__':
    mul_engine = MulMCP()
    model_name = 'Qwen3-32B'

    if 'Qwen3' in model_name:
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'Qwen3-32B')
        os.makedirs(save_parent, exist_ok=True)
        save_path = os.path.join(save_parent, 'mul_mcp_COLBERT_top3.json')
        mul_engine.mul_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_Qwen3,
            rag_type='COLBERT',
            topk=3,
            prompt=prompt_zh,
        )

    elif 'QwQ' in model_name:
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'QwQ')
        os.makedirs(save_parent, exist_ok=True)
        save_path = os.path.join(save_parent, 'mul_mcp_COLBERT_top3.json')
        mul_engine.mul_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_QwQ,
            rag_type='COLBERT',
            topk=3,
            prompt=prompt_zh,
        )

    elif 'qwen3_8b' in model_name:
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'qwen3_8b')
        os.makedirs(save_parent, exist_ok=True)
        save_path = os.path.join(save_parent, 'mul_mcp_COLBERT_top3.json')
        mul_engine.mul_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_qwen3_8b,
            rag_type='COLBERT',
            topk=3,
            prompt=prompt_zh,
        )

    elif 'chatgpt' in model_name:
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'chatgpt')
        os.makedirs(save_parent, exist_ok=True)
        save_path = os.path.join(save_parent, 'mul_mcp_COLBERT_top3.json')
        mul_engine.mul_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_chatgpt,
            rag_type='COLBERT',
            topk=3,
            prompt=prompt_en,
        )
