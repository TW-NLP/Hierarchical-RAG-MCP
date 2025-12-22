import os
from app.mul_mcp.mulmcp import MulMCP
from config import PROJECT_DIR, LLMConfig, prompt_zh, prompt_en

if __name__ == '__main__':
    mul_engine = MulMCP()
    model_name = 'Qwen3-32B'

    if 'Qwen3' in model_name:
        
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'Qwen3-32B')
        if not os.path.exists(save_parent):
            os.makedirs(save_parent)
            
        save_path = os.path.join(save_parent, 'mul_mcp_HIRAG_top3.json')
        
        mul_engine.mul_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_Qwen3,
            rag_type='HIRAG',
            topk=3,
            prompt=prompt_zh
        )

    elif 'QwQ' in model_name:
        
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'QwQ')
        if not os.path.exists(save_parent):
            os.makedirs(save_parent)
            
        save_path = os.path.join(save_parent, 'mul_mcp_HIRAG_top3.json')
        
        mul_engine.mul_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_QwQ,
            rag_type='HIRAG',
            topk=3,
            prompt=prompt_zh
        )

    elif 'qwen3_8b' in model_name:
        
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'qwen3_8b')
        if not os.path.exists(save_parent):
            os.makedirs(save_parent)
            
        save_path = os.path.join(save_parent, 'mul_mcp_HIRAG_top3.json')
        
        mul_engine.mul_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_qwen3_8b,
            rag_type='HIRAG',
            topk=3,
            prompt=prompt_zh
        )
    
    elif 'deepseek' in model_name:
        
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'deepseek')
        if not os.path.exists(save_parent):
            os.makedirs(save_parent)
            
        save_path = os.path.join(save_parent, 'mul_mcp_HIRAG_top3.json')
        
        mul_engine.mul_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_deepseek,
            rag_type='HIRAG',
            topk=3,
            prompt=prompt_zh
        )

    elif 'chatgpt' in model_name:
        
        save_parent = os.path.join(PROJECT_DIR, 'data', 'infer', 'chatgpt')
        if not os.path.exists(save_parent):
            os.makedirs(save_parent)
            
        save_path = os.path.join(save_parent, 'mul_mcp_HIRAG_top3.json')
        
        mul_engine.mul_infer(
            save_path,
            llm_set=LLMConfig.LLM_SET_chatgpt,
            rag_type='HIRAG',
            topk=3,
            prompt=prompt_en
        )